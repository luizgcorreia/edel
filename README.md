# EDEL

Modular implementation of the embedding-driven epistemic landscapes pipeline.

## Deterministic artifacts

Artifacts are addressed as:

`<base_path>/<stage>/<label>/<name>__<hash_segment>.(parquet|pkl)`

Use `edel.io.artifact.make_stage_artifact(...)` and `save_artifact(...)` / `load_artifact(...)`.

## Install

Install with conda:

conda create -n edel python=3.11
conda activate edel
conda install -c conda-forge pyarrow
pip install -e .

## Run

```bash
python scripts/run_experiment.py --base-path artifacts --make-plots
```

## Use in notebooks

```python
from edel.config.defaults import RUN_CONFIG
from edel.pipeline.run import run_pipeline

artifacts = run_pipeline(RUN_CONFIG, base_path="artifacts")
```

## Testing

To run the automated test stack, ensure you have `pytest` installed and run from the project root:

```bash
pytest tests/
```

For stage-specific testing, you can run:

```bash
pytest tests/test_stage_1_data.py
```
