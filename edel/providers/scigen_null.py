"""SCIGen-null synthetic provider using external SCIGen-OpenAlex script."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from edel.providers.base import ensure_schema

SCIGEN_REPO_URL = "https://github.com/luizgcorreia/scigen-openalex.git"


def get_scigen_dir() -> Path:
    """Ensure SCIGen-OpenAlex repo is available in the external/ directory."""
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

    return target_dir


def generate_dataset(config: dict) -> pd.DataFrame:
    """Generate a SCIGen-based synthetic dataset by running an external Perl script."""
    provider_cfg = config.get("provider", {})
    params = provider_cfg.get("params", {})
    n_docs = int(params.get("n_documents", 10))

    scigen_dir = get_scigen_dir()
    out_file = scigen_dir / "dataset.csv"

    # We run the perl script and tell it to output to dataset.csv
    # The script generates OpenAlex-like CSV records
    subprocess.run(
        [
            "perl",
            "generate_openalex.pl",
            "--count",
            str(n_docs),
            "--out",
            "dataset.csv",
        ],
        cwd=str(scigen_dir),
        check=True,
        capture_output=True,
        text=True,
    )

    if not out_file.exists():
        raise RuntimeError(f"SCIGen failed to generate output at {out_file}")

    df = pd.read_csv(out_file)

    return ensure_schema(df, provider_name="scigen_null")
