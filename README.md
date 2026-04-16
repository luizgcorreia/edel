# EDEL

Modular refactor of the embedding-driven epistemic landscapes pipeline.

## Run

```bash
python scripts/run_experiment.py --dataset data/cluster_data.csv --field-dataset data/field_cluster_data.csv --make-plots
```

## Use in notebooks

```python
from edel.config.defaults import RUN_CONFIG
from edel.pipeline.run import run_pipeline

artifacts = run_pipeline(RUN_CONFIG)
```
