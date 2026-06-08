"""Background job queue worker for the EDEL dashboard.

Directory layout:
    {base_path}/jobs/
        pending/   ← submitted jobs waiting for pickup
        running/   ← job currently being executed
        done/      ← completed successfully
        failed/    ← terminated with exception
        logs/      ← one .log file per job_id

Usage (in tmux, with restart wrapper):
    while true; do
        python -m edel.dashboard.worker --base-path artifacts
        echo "Worker crashed — restarting in 2s..."
        sleep 2
    done

The file-move-as-lock pattern (pending/ → running/) is atomic on Linux
filesystems. A worker lock prevents duplicate dashboard-launched workers,
and stale running jobs are returned to pending when their worker PID is gone.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edel.experiments.runner import run_experiments

logger = logging.getLogger("edel.worker")


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _dirs(base: Path) -> dict[str, Path]:
    root = base / "jobs"
    d = {
        "pending": root / "pending",
        "running": root / "running",
        "done":    root / "done",
        "failed":  root / "failed",
        "logs":    root / "logs",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Public: job submission
# ---------------------------------------------------------------------------

def submit_job(config: dict, base_path: str | Path = "artifacts") -> str:
    """Write a new job to the pending queue.

    Returns the job_id (e.g. "job_a3f1c2b0").
    """
    base_path = Path(base_path)
    dirs = _dirs(base_path)

    from edel.experiments.runner import _get_experiment_id
    try:
        experiment_id = _get_experiment_id(config, base_path)
    except Exception:
        experiment_id = "unknown"

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    record = {
        "job_id":        job_id,
        "experiment_id": experiment_id,
        "config":        config,
        "submitted_at":  _now(),
        "started_at":    None,
        "finished_at":   None,
        "error":         None,
    }

    path = dirs["pending"] / f"{job_id}.json"
    path.write_text(json.dumps(record, indent=2))
    logger.info(f"Submitted job: {job_id}")
    return job_id


def submit_sweep(
    base_config: dict,
    sweep_axes: dict[str, list],
    base_path: str | Path = "artifacts",
) -> list[str]:
    """Generate cartesian product of sweep_axes and submit one job per combination.

    Args:
        base_config: The full pipeline config to use as a base.
        sweep_axes: Dict mapping dotted config path → list of values to sweep.
            e.g. {"embedding.model": ["ada-002", "small-3"],
                  "dimensionality_reduction.method": ["diffusion", "umap"]}
        base_path: Root artifact directory.

    Returns:
        List of submitted job_ids.
    """
    import copy
    import itertools

    keys = list(sweep_axes.keys())
    values = list(sweep_axes.values())
    job_ids = []

    for combo in itertools.product(*values):
        config = copy.deepcopy(base_config)
        for path, val in zip(keys, combo):
            _set_nested(config, path, val)
        job_ids.append(submit_job(config, base_path))

    return job_ids


# ---------------------------------------------------------------------------
# Public: queue inspection
# ---------------------------------------------------------------------------

def list_jobs(base_path: str | Path = "artifacts") -> list[dict]:
    """Return all jobs (all states) sorted by submission time (newest first)."""
    base_path = Path(base_path)
    dirs = _dirs(base_path)
    jobs = []

    for state, d in dirs.items():
        if state == "logs":
            continue
        for path in sorted(d.glob("*.json")):
            try:
                record = json.loads(path.read_text())
                record["status"] = state
                if "experiment_id" not in record and "config" in record:
                    from edel.experiments.runner import _get_experiment_id
                    try:
                        record["experiment_id"] = _get_experiment_id(record["config"], base_path)
                    except Exception:
                        record["experiment_id"] = "unknown"
                jobs.append(record)
            except Exception:
                pass

    return sorted(jobs, key=lambda j: j.get("submitted_at", ""), reverse=True)


def get_job_log(job_id: str, base_path: str | Path = "artifacts", tail: int = 50) -> str:
    """Return the last `tail` lines of a job's log file, or empty string."""
    log_path = Path(base_path) / "jobs" / "logs" / f"{job_id}.log"
    if not log_path.exists():
        return ""
    lines = log_path.read_text().splitlines()
    return "\n".join(lines[-tail:])


# ---------------------------------------------------------------------------
# Worker internals
# ---------------------------------------------------------------------------

def _pick_next_job(dirs: dict[str, Path]) -> dict | None:
    """Atomically move the oldest pending job to running/. Returns record or None."""
    candidates = sorted(dirs["pending"].glob("*.json"))
    for path in candidates:
        target = dirs["running"] / path.name
        try:
            # os.rename is atomic on Linux (same filesystem)
            os.rename(path, target)
            return json.loads(target.read_text())
        except FileNotFoundError:
            # Another worker grabbed it first (shouldn't happen with one worker)
            continue
    return None


def _mark_running(record: dict, dirs: dict[str, Path]) -> None:
    record["started_at"] = _now()
    record["worker_pid"] = os.getpid()
    _write_record(record, dirs["running"])


def cancel_job(job_id: str, base_path: str | Path = "artifacts") -> bool:
    """Cancel a job by id.
    
    If pending: delete it / move to failed with error "Cancelled by user".
    If running: terminate the worker process running it, move the job file to failed.
    """
    base_path = Path(base_path)
    dirs = _dirs(base_path)
    
    # Check pending first
    pending_path = dirs["pending"] / f"{job_id}.json"
    if pending_path.exists():
        try:
            record = json.loads(pending_path.read_text())
            record["finished_at"] = _now()
            record["error"] = "Cancelled by user"
            failed_path = dirs["failed"] / f"{job_id}.json"
            failed_path.write_text(json.dumps(record, indent=2))
            pending_path.unlink()
            logger.info(f"Cancelled pending job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel pending job {job_id}: {e}")
            return False

    # Check running
    running_path = dirs["running"] / f"{job_id}.json"
    if running_path.exists():
        try:
            record = json.loads(running_path.read_text())
            pid = record.get("worker_pid")
            
            # 1. Kill the process if PID exists
            if pid:
                import signal
                try:
                    os.kill(pid, signal.SIGTERM)
                    logger.info(f"Sent SIGTERM to worker process {pid} for job {job_id}")
                except ProcessLookupError:
                    pass
            
            # 2. Mark the job as failed (cancelled)
            record["finished_at"] = _now()
            record["error"] = "Cancelled by user"
            failed_path = dirs["failed"] / f"{job_id}.json"
            failed_path.write_text(json.dumps(record, indent=2))
            running_path.unlink()
            logger.info(f"Cancelled running job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel running job {job_id}: {e}")
            return False
            
    return False


def delete_job_record(job_id: str, base_path: str | Path = "artifacts") -> bool:
    """Delete a completed (done or failed) job and all its physical artifacts."""
    base_path = Path(base_path)
    dirs = _dirs(base_path)
    
    # Check done and failed directories
    job_path = None
    if (dirs["done"] / f"{job_id}.json").exists():
        job_path = dirs["done"] / f"{job_id}.json"
    elif (dirs["failed"] / f"{job_id}.json").exists():
        job_path = dirs["failed"] / f"{job_id}.json"
        
    if not job_path:
        logger.warning(f"Job JSON not found for {job_id}")
        return False
        
    try:
        record = json.loads(job_path.read_text())
        experiment_id = record.get("experiment_id")
        config = record.get("config")
        
        # 1. Delete the job JSON file
        job_path.unlink()
        
        # 2. Delete the log file
        log_path = base_path / "jobs" / "logs" / f"{job_id}.log"
        if log_path.exists():
            log_path.unlink()
            
        # 3. Clean up physical artifacts if config is present
        if config:
            from edel.io.artifact import delete_experiment_artifacts
            delete_experiment_artifacts(config, base_path)
            
        # 4. Clean up registry pickle and results.parquet cache if experiment_id is present
        if experiment_id:
            from edel.experiments.runner import delete_registry_record, delete_from_results_cache
            delete_registry_record(experiment_id, base_path)
            delete_from_results_cache(experiment_id, base_path)
            
        logger.info(f"Deleted job record, log, and artifacts for job {job_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete job {job_id}: {e}", exc_info=True)
        return False


def _mark_done(record: dict, dirs: dict[str, Path]) -> None:
    record["finished_at"] = _now()
    src = dirs["running"] / f"{record['job_id']}.json"
    dst = dirs["done"]    / f"{record['job_id']}.json"
    _write_record(record, dirs["running"])  # update with finished_at first
    os.rename(src, dst)


def _mark_failed(record: dict, dirs: dict[str, Path], error: str) -> None:
    record["finished_at"] = _now()
    record["error"] = error
    src = dirs["running"] / f"{record['job_id']}.json"
    dst = dirs["failed"]  / f"{record['job_id']}.json"
    _write_record(record, dirs["running"])
    os.rename(src, dst)


def _write_record(record: dict, directory: Path) -> None:
    path = directory / f"{record['job_id']}.json"
    path.write_text(json.dumps(record, indent=2))


def _setup_job_logger(job_id: str, dirs: dict[str, Path]) -> logging.FileHandler:
    """Attach a per-job FileHandler to the root logger."""
    log_path = dirs["logs"] / f"{job_id}.log"
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _teardown_job_logger(handler: logging.FileHandler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_nested(d: dict, dotted_path: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key path."""
    keys = dotted_path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


class TeeStdout:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.original_stdout = sys.stdout
        self.file = None

    def __enter__(self):
        self.file = open(self.log_path, "a", encoding="utf-8")
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        if self.file:
            self.file.close()

    def write(self, message):
        self.file.write(message)
        self.file.flush()
        self.original_stdout.write(message)
        self.original_stdout.flush()

    def flush(self):
        self.file.flush()
        self.original_stdout.flush()


class TeeStderr:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.original_stderr = sys.stderr
        self.file = None

    def __enter__(self):
        self.file = open(self.log_path, "a", encoding="utf-8")
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self.original_stderr
        if self.file:
            self.file.close()

    def write(self, message):
        self.file.write(message)
        self.file.flush()
        self.original_stderr.write(message)
        self.original_stderr.flush()

    def flush(self):
        self.file.flush()
        self.original_stderr.flush()


def _cleanup_orphaned_jobs(dirs: dict[str, Path]) -> None:
    """Find any running jobs whose worker PIDs are no longer active, and move them back to pending."""
    running_jobs = list(dirs["running"].glob("*.json"))
    for path in running_jobs:
        try:
            record = json.loads(path.read_text())
            pid = record.get("worker_pid")

            # Check if PID is alive
            pid_alive = False
            if pid:
                try:
                    os.kill(pid, 0)
                    pid_alive = True
                except OSError:
                    pid_alive = False

            if not pid_alive:
                # Move back to pending
                target = dirs["pending"] / path.name
                os.rename(path, target)
                logger.info(
                    "Detected orphaned job %s (worker PID %s is dead). Moved back to pending.",
                    record.get("job_id"),
                    pid,
                )
        except Exception as e:
            logger.error(f"Error checking/restoring orphaned job {path.name}: {e}")


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------

def run_worker(base_path: str | Path = "artifacts") -> None:
    """Blocking worker loop. Run in a tmux session with the restart wrapper."""
    base_path = Path(base_path)
    dirs = _dirs(base_path)

    # Ensure only one worker process runs at a time
    lock_path = base_path / "jobs" / "worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Open lock file and keep descriptor alive for the life of this process.
    lock_file = open(lock_path, "w")
    try:
        import fcntl
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("[-] Another worker process is already running. Exiting to avoid duplicate execution.")
        sys.exit(0)
    except ImportError:
        # Fallback for platforms without fcntl
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Worker started. Watching: {base_path / 'jobs' / 'pending'}")
    _cleanup_orphaned_jobs(dirs)

    while True:
        job = _pick_next_job(dirs)

        if job is None:
            time.sleep(5)
            continue

        job_id = job["job_id"]
        logger.info(f"Starting job: {job_id}")

        # Attach per-job file log
        handler = _setup_job_logger(job_id, dirs)
        log_file_path = dirs["logs"] / f"{job_id}.log"

        try:
            _mark_running(job, dirs)
            logger.info(f"[{job_id}] Pipeline starting...")

            with TeeStdout(log_file_path), TeeStderr(log_file_path):
                run_experiments(
                    configs=[job["config"]],
                    base_path=base_path,
                    force=False,
                )

            _mark_done(job, dirs)
            logger.info(f"[{job_id}] ✅ Done.")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"[{job_id}] ❌ Failed: {error_msg}", exc_info=True)
            _mark_failed(job, dirs, error_msg)

        finally:
            _teardown_job_logger(handler)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EDEL background pipeline worker.")
    parser.add_argument(
        "--base-path",
        default="artifacts",
        help="Root artifact directory (default: artifacts)",
    )
    args = parser.parse_args()
    run_worker(base_path=args.base_path)


if __name__ == "__main__":
    main()
