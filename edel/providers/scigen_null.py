"""SCIGen-null synthetic provider using external SCIGen-OpenAlex script."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from edel.providers.base import ensure_schema

SCIGEN_REPO_URL = "https://github.com/luizgcorreia/scigen-openalex.git"


def get_scigen_dir() -> Path:
    """Ensure SCIGen-OpenAlex repo is available and up-to-date in the external/ directory."""
    # Use a location relative to the edel package (project root)
    repo_root = Path(__file__).parents[2]
    target_dir = repo_root / "external" / "scigen-openalex"
    script_path = target_dir / "generate_openalex.pl"

    if not script_path.exists():
        print(f"Cloning SCIGen-OpenAlex repository to {target_dir}...")
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        # Remove partial directory if it exists
        if target_dir.exists():
            import shutil

            shutil.rmtree(target_dir)

        subprocess.run(["git", "clone", SCIGEN_REPO_URL, str(target_dir)], check=True)
    else:
        # Pull latest changes so the remote server always runs the current script
        print(f"Pulling latest SCIGen-OpenAlex updates in {target_dir}...")
        result = subprocess.run(
            ["git", "pull"],
            cwd=str(target_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Warning: git pull failed (will use existing clone): {result.stderr.strip()}")

    return target_dir


def generate_dataset(config: dict) -> tuple[pd.DataFrame, dict]:
    """Generate a SCIGen-based synthetic dataset by running an external Perl script."""
    provider_cfg = config.get("provider", {})
    params = provider_cfg.get("params", {})
    n_docs = int(params.get("n_documents", 10))

    scigen_dir = get_scigen_dir()
    out_file = scigen_dir / "dataset.csv"

    # Pre-flight: verify required data files are present
    required_files = ["scirules.in", "system_names.in", "Autoformat.pm", "scigen.pm"]
    missing = [f for f in required_files if not (scigen_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"SCIGen data files missing in {scigen_dir}: {missing}. "
            "The git clone may be incomplete — delete external/scigen-openalex/ and retry."
        )

    # Run the Perl script; capture output so we can surface any error message
    result = subprocess.run(
        [
            "perl",
            "generate_openalex.pl",
            "--count",
            str(n_docs),
            "--out",
            "dataset.csv",
        ],
        cwd=str(scigen_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"SCIGen Perl script failed (exit {result.returncode}).\n"
            f"STDOUT: {result.stdout.strip()}\n"
            f"STDERR: {result.stderr.strip()}"
        )

    if not out_file.exists():
        raise RuntimeError(f"SCIGen failed to generate output at {out_file}")

    df = pd.read_csv(out_file)

    return ensure_schema(df, provider_name="scigen_null"), {}
