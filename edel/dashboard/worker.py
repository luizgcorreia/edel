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
filesystems, making it safe to run a single worker without additional
locking primitives.
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
    _write_record(record, dirs["running"])


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


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------

def run_worker(base_path: str | Path = "artifacts") -> None:
    """Blocking worker loop. Run in a tmux session with the restart wrapper."""
    base_path = Path(base_path)
    dirs = _dirs(base_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Worker started. Watching: {base_path / 'jobs' / 'pending'}")

    while True:
        job = _pick_next_job(dirs)

        if job is None:
            time.sleep(5)
            continue

        job_id = job["job_id"]
        logger.info(f"Starting job: {job_id}")

        # Attach per-job file log
        handler = _setup_job_logger(job_id, dirs)

        try:
            _mark_running(job, dirs)
            logger.info(f"[{job_id}] Pipeline starting...")

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
