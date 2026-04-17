# EDEL

Modular refactor of the embedding-driven epistemic landscapes pipeline.

## Deterministic artifacts

Artifacts are addressed as:

`<base_path>/<stage>/<config_hash>/<name>.(parquet|pkl)`

Use `edel.io.artifact.make_stage_artifact(...)` and `save_artifact(...)` / `load_artifact(...)`.

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
