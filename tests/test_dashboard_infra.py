"""Tests for the EDEL Dashboard Infrastructure (Jobs, Worker, Cache)."""

import json
import time
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from edel.dashboard.worker import (
    submit_job,
    submit_sweep,
    list_jobs,
    _pick_next_job,
    _mark_running,
    _mark_done,
    _mark_failed,
    _dirs
)
from edel.dashboard.cache import get_results_df, rebuild_results_cache
from edel.dashboard.utils import parse_config_json, get_nested, df_to_dash_records


# ---------------------------------------------------------------------------
# Job & Worker Tests
# ---------------------------------------------------------------------------

def test_job_submission(tmp_path):
    """Test that submitting a job creates a file in the pending directory."""
    config = {"test": "config"}
    job_id = submit_job(config, base_path=tmp_path)
    
    pending_dir = tmp_path / "jobs" / "pending"
    job_file = pending_dir / f"{job_id}.json"
    
    assert job_file.exists()
    record = json.loads(job_file.read_text())
    assert record["job_id"] == job_id
    assert record["config"] == config
    assert record["submitted_at"] is not None


def test_sweep_submission(tmp_path):
    """Test that submitting a sweep creates the correct number of jobs."""
    base_config = {"a": 1, "b": {"c": 2}}
    sweep_axes = {
        "a": [10, 20],
        "b.c": [3, 4, 5]
    }
    
    job_ids = submit_sweep(base_config, sweep_axes, base_path=tmp_path)
    
    # 2 * 3 = 6 jobs
    assert len(job_ids) == 6
    pending_files = list((tmp_path / "jobs" / "pending").glob("*.json"))
    assert len(pending_files) == 6


def test_worker_state_machine(tmp_path):
    """Test the atomic move and state transitions (pending -> running -> done)."""
    dirs = _dirs(tmp_path)
    job_id = submit_job({"task": "test"}, base_path=tmp_path)
    
    # 1. Pick job (atomic move)
    job = _pick_next_job(dirs)
    assert job is not None
    assert job["job_id"] == job_id
    assert not (dirs["pending"] / f"{job_id}.json").exists()
    assert (dirs["running"] / f"{job_id}.json").exists()
    
    # 2. Mark running
    _mark_running(job, dirs)
    record = json.loads((dirs["running"] / f"{job_id}.json").read_text())
    assert record["started_at"] is not None
    
    # 3. Mark done
    _mark_done(job, dirs)
    assert not (dirs["running"] / f"{job_id}.json").exists()
    assert (dirs["done"] / f"{job_id}.json").exists()
    
    # Verify list_jobs sees it in 'done'
    all_jobs = list_jobs(tmp_path)
    assert any(j["job_id"] == job_id and j["status"] == "done" for j in all_jobs)


def test_worker_failure_state(tmp_path):
    """Test transition to failed state with error message."""
    dirs = _dirs(tmp_path)
    job_id = submit_job({"task": "fail"}, base_path=tmp_path)
    
    job = _pick_next_job(dirs)
    _mark_failed(job, dirs, "Something went wrong")
    
    assert not (dirs["running"] / f"{job_id}.json").exists()
    assert (dirs["failed"] / f"{job_id}.json").exists()
    
    record = json.loads((dirs["failed"] / f"{job_id}.json").read_text())
    assert record["error"] == "Something went wrong"


# ---------------------------------------------------------------------------
# Cache Tests
# ---------------------------------------------------------------------------

@patch("edel.dashboard.cache.analyze_experiments")
@patch("edel.dashboard.cache.load_registry")
def test_cache_rebuild_delta(mock_registry, mock_analyze, tmp_path):
    """Test that delta rebuild only re-analyzes stale experiments."""
    cache_path = tmp_path / "experiments" / "results.parquet"
    
    # Setup mock registry with two experiments
    exp1 = {"experiment_id": "exp1", "artifact_refs": {"landscape": MagicMock()}}
    exp2 = {"experiment_id": "exp2", "artifact_refs": {"landscape": MagicMock()}}
    
    # Mock file paths for artifacts
    exp1["artifact_refs"]["landscape"].parquet_path = tmp_path / "exp1.parquet"
    exp1["artifact_refs"]["landscape"].pkl_path = tmp_path / "exp1.pkl"
    exp2["artifact_refs"]["landscape"].parquet_path = tmp_path / "exp2.parquet"
    exp2["artifact_refs"]["landscape"].pkl_path = tmp_path / "exp2.pkl"
    
    # Create the files
    for exp in [exp1, exp2]:
        exp["artifact_refs"]["landscape"].parquet_path.touch()
    
    mock_registry.return_value = [exp1, exp2]
    
    # 1. Initial full build
    mock_analyze.return_value = pd.DataFrame([
        {"experiment_id": "exp1", "metric": 0.5},
        {"experiment_id": "exp2", "metric": 0.6}
    ])
    
    rebuild_results_cache(tmp_path, delta_only=False)
    assert cache_path.exists()
    
    # 2. Update exp1 artifact mtime to make it stale
    time.sleep(0.1) # ensure mtime difference
    exp1["artifact_refs"]["landscape"].parquet_path.touch()
    
    # 3. Delta rebuild
    mock_analyze.return_value = pd.DataFrame([
        {"experiment_id": "exp1", "metric": 0.9} # Updated metric
    ])
    
    full_df = rebuild_results_cache(tmp_path, delta_only=True)
    
    # Verify analyze was only called with exp1
    assert mock_analyze.call_count == 2
    last_call_args = mock_analyze.call_args[0][0]
    assert len(last_call_args) == 1
    assert last_call_args[0]["experiment_id"] == "exp1"
    
    # Verify upsert logic
    assert len(full_df) == 2
    assert full_df.set_index("experiment_id").loc["exp1", "metric"] == 0.9
    assert full_df.set_index("experiment_id").loc["exp2", "metric"] == 0.6


# ---------------------------------------------------------------------------
# Utils Tests
# ---------------------------------------------------------------------------

def test_utils_config_parsing():
    assert parse_config_json('{"a": 1}') == {"a": 1}
    assert parse_config_json('invalid') is None


def test_utils_nested_get():
    d = {"a": {"b": {"c": 42}}}
    assert get_nested(d, "a.b.c") == 42
    assert get_nested(d, "a.x", default="miss") == "miss"
    assert get_nested(d, "x.y") is None


def test_df_to_dash_records():
    df = pd.DataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    records = df_to_dash_records(df, max_rows=1)
    assert len(records) == 1
    assert records[0] == {"a": 1, "b": 2}


def test_dashboard_table_helpers():
    """Test the split_filter_part and parse_filter_query helpers used for the dynamic DataTable."""
    from edel.dashboard.callbacks.experiments import split_filter_part, parse_filter_query
    
    # 1. Test split_filter_part
    assert split_filter_part('{title} contains "HOL"') == ['{title}', '"HOL"', 'contains']
    assert split_filter_part('{age} ge 18') == ['{age}', '18', '>=']
    assert split_filter_part('{name} eq "John"') == ['{name}', '"John"', '==']
    assert split_filter_part('{status} ne "failed"') == ['{status}', '"failed"', '!=']
    
    # 2. Test parse_filter_query
    df = pd.DataFrame([
        {"title": "HOL-Analysis.Connected", "age": 20, "name": "John"},
        {"title": "HOL-Analysis.Compact", "age": 15, "name": "Jane"},
        {"title": "Interval_Analysis.Interval", "age": 30, "name": "Bob"}
    ])
    
    # Filter: title contains "analysis" (case-insensitive)
    filtered = parse_filter_query('{title} contains "analysis"', df)
    assert len(filtered) == 3
    
    # Filter: title contains "HOL"
    filtered = parse_filter_query('{title} contains "HOL"', df)
    assert len(filtered) == 2
    
    # Filter: age >= 18
    filtered = parse_filter_query('{age} ge 18', df)
    assert len(filtered) == 2
    
    # Filter: name eq "Jane"
    filtered = parse_filter_query('{name} eq "Jane"', df)
    assert len(filtered) == 1
    assert filtered.iloc[0]["name"] == "Jane"
    
    # Complex Filter: title contains "HOL" && age lt 18
    filtered = parse_filter_query('{title} contains "HOL" && {age} lt 18', df)
    assert len(filtered) == 1
    assert filtered.iloc[0]["name"] == "Jane"
