# Configuration Guide (Main, Benchmarking, Reproducibility)

This guide explains:

1. Which config files are used by each entrypoint.
2. How config values flow into `src/networks`, `src/trainers`, and `src/datasets`.
3. What is currently validated/handled correctly vs what is currently ignored.

Scope audited here:

- `main.py`
- `benchmarking.py`
- `reproducibility.py`
- `configs/benchmarking.yaml`
- all files under `configs/benchmarking/`
- all files under `configs/networks/`
- all files under `configs/reproducibility/`

---

## 1) Quick Decision: Which Config Style Should I Use?

- Use `main.py --config <file>` when you want **one specific experiment**.
  - Example configs: `configs/benchmarking/bptt/...`, `configs/benchmarking/ottt/...`, `configs/reproducibility/...`, and legacy root configs in `configs/*.yaml`.

- Use `benchmarking.py` when you want a **matrix benchmark**:
  - datasets x network blueprints x trainers
  - global suite config in `configs/benchmarking.yaml`
  - network blueprints in `configs/networks/*.yaml`

- Use `reproducibility.py` when you want to run **multiple single-run configs** from a folder:
  - default folder: `configs/reproducibility/`
  - internally reuses the same training pipeline as `main.py`.

---

## 2) End-to-End Config Flow by Entrypoint

## 2.1 `main.py` (single run)

Config path:

- CLI parsed in `src/utils/parameters.py`
- File loaded in `src/utils/config.py::load_config`
- CLI overrides merged in `src/utils/config.py::merge_config_with_args`
- Validated in `src/utils/config.py::validate_config`
- Training executed in `main.py::trainable`

Runtime wiring:

- `trainer.name` -> trainer class import via `main.py::get_trainer` -> `src/trainers/*`.
- `model.*` + `trainer.name` -> network creation through:
  - custom branches in `main.py` for `recurrent`, `conv`, and DRTP paper conv mode
  - otherwise `src/networks/get_network.py` -> `src/networks/*`.
- `data.dataset`, `training.batch_size`, `data.timesteps`, architecture-dependent flattening -> `src/datasets/get_loader.py` -> `src/datasets/*`.
- `training.optimizer`, `training.learning_rate`, `training.weight_decay` -> optimizer creation in `main.py`.
- Trainer-specific sections (`drtp`, `etlp`, `ostl`, `stop`, `osttp`) -> explicit kwargs in `main.py`.

---

## 2.2 `benchmarking.py` (matrix run)

Config surfaces:

1. Global suite config: `configs/benchmarking.yaml`
2. Network blueprints: `configs/networks/*.yaml`

Planning phase (`BenchmarkingSuite.initialize`):

- load global config and CLI overrides
- discover network YAML files
- resolve trainer classes from `trainers.<name>_trainer`
- build all dataset x network x trainer combinations
- skip invalid combinations based on:
  - network tags (`fully_connected`, `convolutional`, `recurrent`, etc.)
  - required/excluded tags from trainer config
  - required network attributes
  - architecture allow-list

Execution phase (`BenchmarkingSuite.run`):

- instantiate network from network blueprint (`network_kwargs`)
- build trainer kwargs from trainer spec (`training` + `params`)
- train/evaluate each experiment
- write manifest and reports (`experiment_manifest.json`, `results.json`, `results.csv`, `summary.md`)

---

## 2.3 `reproducibility.py` (batch of single-run configs)

Flow:

- discover `*.yaml/*.yml/*.json` under `--configs-dir`
- read raw metadata (trainer/dataset) for scheduling
- for each file:
  - load full typed config with `load_config`
  - apply CLI overrides (`epochs`, `batch-size`, `lr`, `timesteps`, `device`, `seed`)
  - validate with `validate_config`
  - run through `main.trainable` (same runtime path as `main.py`)

Important behavior:

- `reproducibility.py` overrides `config.experiment.log_dir` so outputs are grouped under the suite run directory.

---

## 3) Config Surfaces and Field Usage

## 3.1 `configs/benchmarking.yaml` (global benchmark suite)

### `experiment`

- `output_dir`: used (output root for timestamped benchmark runs).
- `seed`: used (set before each experiment for comparability).
- `deterministic`: used (passed to seed helper).
- `name`: currently not used for output naming (run folder is hardcoded as `full_benchmark_<timestamp>`).

### `execution`

- `epochs`, `batch_size`, `learning_rate`, `timesteps`: used.
- `device`: used by device resolver (`auto`, `cuda`, `cpu`, etc.).
- `show_epoch_progress`: used.
- `max_train_batches`, `max_test_batches`: used.
- `single_process_data_loading`: used (forces `num_workers=0` in wrapped DataLoaders).
- `continue_on_error`: used.
- `run_neurobench`: used.
- `neurobench_include_synaptic_operations`: optional; used if present.

### `data`

- `datasets`: used to build schedule.
- `dataset_defaults`: used to populate per-dataset defaults:
  - `input_shape` / `input_size`
  - `num_classes`
  - per-dataset `timesteps`

### `trainers.<name>`

- `module`, `class_name`: used to import trainer class.
- `enabled`: used.
- `requires_all_tags`, `excludes_any_tags`, `requires_network_attrs`, `allowed_architectures`: used for compatibility filtering.
- `training.requires_grad`, `training.use_optimizer`, `training.optimizer`: used when constructing trainer/optimizer.
- `params`: passed into trainer constructor (keys unsupported by constructor are filtered out).

---

## 3.2 `configs/networks/*.yaml` (network blueprints for matrix benchmarking)

Each file provides:

- `name`: used in experiment IDs and reports.
- `enabled`: used.
- `model.architecture`: used for selecting network class.
- `network_kwargs`: used directly for network instantiation.
- `dataset_overrides`: deep-merged per dataset.

Notes:

- In current implementation, runtime instantiation is driven by `network_kwargs` (plus compatibility metadata and `model.quantization`).
- `model.layer_sizes`, `model.conv_layers`, etc. are mostly metadata in the benchmark pipeline unless your own tooling reads them.
- Keep `model.*` and `network_kwargs.*` consistent manually to avoid confusion.

Supported benchmarking architectures:

- `fc_snn` -> `src/networks/benchmarking/fc_snn.py`
- `r_snn` -> `src/networks/benchmarking/r_snn.py`
- `conv_snn` -> `src/networks/benchmarking/conv_snn.py`
- `vg11_snn` -> `src/networks/benchmarking/vg11_snn.py`

---

## 3.3 `configs/benchmarking/...` and `configs/reproducibility/...` (single-run configs)

These are loaded via `load_config` (typed dataclass `Config`).

Top-level sections and usage:

- `experiment`: used for naming/logging/seed/determinism.
- `model`: used for network construction (`architecture`, `layer_sizes`, `beta`, `tau`, `threshold`, `quantization`, `conv_layers`, `recurrent_type`).
- `training`: used for epochs, batch size, LR, optimizer, weight decay, freeze conv.
- `trainer`: used for trainer class + generic trainer flags (`update_last`, `update_every`, `seq_batch`).
- `drtp` / `etlp` / `ostl` / `stop` / `osttp`: used only when that trainer is selected.
- `data`: `dataset` and `timesteps` are used.
- `hardware.device`: used.
- `checkpoint`: used.

---

## 4) How Configs Reach `src/networks`, `src/trainers`, `src/datasets`

## 4.1 Networks

- `main.py` -> `src/networks/get_network.py` for most architectures.
- `benchmarking.py` -> `_NETWORK_FACTORY` (`FCSNN`, `RSNN`, `ConvSNN`, `VG11SNN`) from `src/networks/benchmarking`.
- DRTP conv paper mode in `main.py` uses `src/networks/reproducibility/DRTP_convolutional_network.py`.
- OTTT reproducibility architecture `ottt_repro` maps to `src/networks/reproducibility/ottt_vgg_sws_snntorch.py`.

## 4.2 Trainers

- `main.py` imports by name using `main.py::get_trainer`.
- `benchmarking.py` imports from `trainers.<module>` using config-specified module/class.
- Trainer constructor kwargs are built from config in:
  - `main.py` (explicit branches)
  - `benchmarking.py::_trainer_kwargs` (filtered against signature)

## 4.3 Datasets

- All paths use `src/datasets/get_loader.py`.
- `get_loader` dispatches to dataset-specific loaders in `src/datasets/*_loader.py`.
- Training loops transpose loader output from `[B, T, ...]` to `[T, B, ...]` before calling trainers.
- Flatten vs image-shape input is chosen by architecture:
  - FC/recurrent-like -> flattened
  - conv/vgg/ottt_repro -> non-flattened spatial tensors

---

## 5) CLI Override Rules

## 5.1 `main.py`

- If `--config` is provided, only explicitly passed CLI flags override config values.
- `--optimizer` is a boolean flag that forces `training.optimizer = "adam"`.
- `--layer-size`/`--n-layers` can reconstruct `model.layer_sizes` based on dataset input/class sizes.

## 5.2 `benchmarking.py`

Overrides available:

- `--epochs`, `--batch-size`, `--lr`, `--timesteps`, `--device`, `--seed`
- `--max-train-batches`, `--max-test-batches`
- filters: `--datasets`, `--algorithms`, `--networks`
- `--run-neurobench`

## 5.3 `reproducibility.py`

Overrides available:

- `--epochs`, `--batch-size`, `--lr`, `--timesteps`, `--device`, `--seed`
- filter: `--algorithms`

---

## 6) Validation and Compatibility

Validation (`validate_config`) checks:

- structural constraints (positive epochs/batch/timesteps/LR, architecture/trainer names, etc.)
- trainer-specific constraints (e.g. OSTL/OSTTP/STOP architecture constraints)

Benchmarking compatibility adds extra runtime scheduling checks:

- network tags vs trainer requirements
- required network attributes
- architecture allow-list

Skipped combinations are explicitly reported in:

- benchmark manifest (`experiment_manifest.json`)
- benchmark summary (`summary.md`)

---

## 7) Audit Results for the Requested Configs

What was checked:

1. `configs/benchmarking/*.yaml` loaded with `load_config` + `validate_config`: all pass.
2. `configs/reproducibility/*.yaml` loaded with `load_config` + `validate_config`: all pass.
3. `benchmarking.py --dry-run` with `configs/benchmarking.yaml` + `configs/networks`: succeeds.
   - planned experiments: 72
   - skipped combinations: 24 (all explained by compatibility rules)
4. `reproducibility.py --dry-run`: discovers 2 configs, both runnable.

Conclusion:

- The requested configs are correctly handled by their intended pipelines.
- The skip behavior in benchmark matrix mode is expected and transparent.

---

## 8) Known Gaps / Caveats (Important)

These are implementation caveats, not failures of your current configs:

1. `data.data_dir` in single-run configs is currently not used by loaders.
   - Dataset root is driven by loader constants / `STSF_DATA` env var.

2. `data.num_workers` in single-run configs is currently not used.
   - Most loaders hardcode `num_workers=4`.
   - Benchmarking mode can force `num_workers=0` only via `execution.single_process_data_loading`.

3. `hardware.mixed_precision` exists in config schema but is currently not used in training code.

4. In benchmarking network blueprints, `model.*` and `network_kwargs.*` can diverge.
   - Runtime uses `network_kwargs` for instantiation.
   - Keep both synchronized for clarity.

5. `experiment.name` in `configs/benchmarking.yaml` is not used to name run directories.

6. `validate_config` accepts trainers like `stllr`/`stdp`, but `main.py::get_trainer` does not currently map those names.
   - Not an issue for the audited config sets, but relevant if you add those trainers to single-run configs.

---

## 9) Practical Usage Recipes

## 9.1 Run one config directly

```bash
python3 main.py --config configs/benchmarking/bptt/bptt-mnist-fc_snn.yaml --epochs 1
```

## 9.2 Run the full matrix benchmark

```bash
python3 benchmarking.py --config configs/benchmarking.yaml --networks-dir configs/networks
```

## 9.3 Run a filtered matrix benchmark

```bash
python3 benchmarking.py \
  --config configs/benchmarking.yaml \
  --networks-dir configs/networks \
  --algorithms bptt,ottt \
  --datasets MNIST,CIFAR10 \
  --networks fc_snn,conv_snn \
  --epochs 1
```

## 9.4 Run reproducibility suite configs

```bash
python3 reproducibility.py --configs-dir configs/reproducibility --epochs 1
```

## 9.5 Preview scheduling only (no training)

```bash
python3 benchmarking.py --dry-run
python3 reproducibility.py --dry-run
```

---

## 10) How to Add a New Experiment Config Safely

For a new single-run config (`main.py` / `reproducibility.py`):

1. Start from a nearby working file in `configs/benchmarking/` or `configs/reproducibility/`.
2. Keep `trainer.name` and `model.architecture` compatible.
3. Validate:
   ```bash
   python3 -c "import sys; sys.path.insert(0,'src'); from utils.config import load_config, validate_config; c=load_config('your.yaml'); print('\\n'.join(validate_config(c)) or 'OK')"
   ```
4. Run a short smoke test with `--epochs 1`.

For benchmarking matrix mode:

1. Add/modify network blueprint in `configs/networks/`.
2. Ensure `network_kwargs` fully describes constructor args for the target class.
3. Add/adjust trainer compatibility rules in `configs/benchmarking.yaml`.
4. Run `python3 benchmarking.py --dry-run` and inspect skipped reasons.

