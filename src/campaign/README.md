# Campaign Package

Thin orchestration layer that converts YAML config files into training runs.

## Modules

| File | Purpose |
|------|---------|
| `experiment_spec.py` | `ExperimentSpec` dataclass — the resolved config for one run |
| `config_loader.py`   | YAML loading, deep-merge, per-dataset model resolution |
| `compatibility.py`   | Trainer × model × dataset compatibility matrix |
| `campaign_builder.py`| Builds `list[ExperimentSpec]` from benchmarking or custom YAML |
| `optuna_helpers.py`  | Resolves Optuna attribute blocks to plain values |
| `training_loop.py`   | `train_one_epoch` and `evaluate` functions |
| `results.py`         | Write per-experiment and campaign-level result files |

## Data flow

```
Input YAML
    │
    ▼
campaign_builder  ──→  list[ExperimentSpec]
                               │
               ┌───────────────┼───────────────┐
               │               │               │
           run_exp_campaign.py spawns experiment.py per spec
                               │
                               ▼
                    experiment.py
                      ├─ get_loader(dataset)
                      ├─ get_network(model)
                      ├─ TRAINER_REGISTRY[trainer]
                      ├─ training_loop.train_one_epoch × epochs
                      ├─ training_loop.evaluate
                      └─ neurobench_eval.run_neurobench
                               │
                               ▼
             experiments/<campaign>/<exp_name>/
               config.yaml   metrics.json   log.txt
```

## Adding a new trainer

1. Create `src/trainers/my_trainer.py` implementing `BaseTrainer`.
2. Add it to `TRAINER_REGISTRY` in `src/trainers/__init__.py`.
3. Create `config/default/trainers/my_trainer.yaml` with its defaults.
4. Add compatibility entries in `compatibility.py` if it requires a specific model.
