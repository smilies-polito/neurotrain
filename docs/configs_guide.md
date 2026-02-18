# Config Files Guide (Benchmarking vs Reproducibility)

This repo exposes two config-driven entrypoints:

- `reproducibility.py`: run *one config file per experiment* (typically “per paper/per setting”).
- `benchmarking.py`: run a *matrix benchmark* across **datasets x networks x trainers**.

Both accept YAML (`.yaml`/`.yml`) and (for reproducibility) JSON (`.json`).

---

## 1) Reproducibility mode (`reproducibility.py`)

### 1.1 Where configs live + how they are discovered

- Default directory: `configs/reproducibility/` (override with `--configs-dir`).
- The script recursively scans for `*.yaml`, `*.yml`, `*.json`.
- Each file becomes one runnable experiment **only if** it contains:
  - `trainer.name` (required; configs missing this are skipped)
  - `data.dataset` (strongly recommended; used in reports/IDs; missing becomes `"unknown"` in the suite UI)

Naming note:
- The file stem becomes `config_name` in the summary, and is part of the stable experiment id:
  `"{trainer}__{dataset}__{config_stem}"`.

Logging note:
- `reproducibility.py` overrides `experiment.log_dir` at runtime so all artifacts land under its timestamped
  `--output-dir` folder. You can still set `experiment.log_dir` in the config, but the suite will replace it.

### 1.2 Config schema (what keys are allowed)

Reproducibility configs are parsed by `src/utils/config.py` (`load_config()` -> `Config.from_dict()`).
That means:

- The root must be a mapping (YAML dict / JSON object).
- **Inside each section**, only the keys listed below are allowed. Unknown keys inside a section will raise an error.

Sections and keys:

- `experiment`:
  - `name` (string)
  - `seed` (int)
  - `deterministic` (bool)
  - `log_dir` (string path)
- `model`:
  - `architecture` (string; see “Allowed values”)
  - `layer_sizes` (list of ints; `[input, hidden..., output]`)
  - `conv_layers` (list of dicts; used by some conv architectures)
  - `beta` (float)
  - `tau` (float or null; used by ELL/FELL/BELL and some paper reproductions)
  - `threshold` (float)
  - `recurrent_type` (string; `standard|srnn|snu|ssnu`)
  - `quantization` (bool)
- `training`:
  - `epochs` (int)
  - `batch_size` (int)
  - `learning_rate` (float)
  - `optimizer` (string or null; see “Allowed values”)
  - `weight_decay` (float)
  - `freeze_conv` (bool)
- `trainer`:
  - `name` (string; see “Allowed values”)
  - `update_last` (bool)
  - `update_every` (int)
  - `seq_batch` (int)
- Trainer-specific sections (optional; keep values valid even if unused):
  - `drtp`: `loss`, `output_mode`, `paper_reproduction`, `surrogate_scale`, `surrogate_type`, `feedback_distribution`, `feedback_scale`, `fixed_feedback`
  - `etlp`: `trace_decay`, `surrogate_scale`, `voltage_reg`, `weight_l1`, `weight_l2`, `update_rate_hz`, `dt_ms`, `feedback_distribution`, `feedback_scale`
  - `ostl`: `surrogate_scale`, `grad_clip`, `output_mode`
  - `osttp`: `pseudo_derivative`, `output_loss`, `output_readout`, `feedback_scale`, `feedback_seed`, `target_dim`, `grad_clip`, `debug`
  - `stop`: `loss`, `surrogate`, `learn_weights`, `learn_thresholds`, `learn_leakage`, `lr_weight`, `lr_threshold`, `lr_leakage`, `threshold_min`, `momentum`, `cosine_schedule`, `cosine_t_max`, `static_input_timesteps`
- `data`:
  - `dataset` (string; see “Allowed values”)
  - `timesteps` (int)
  - `data_dir` (string path)
  - `num_workers` (int)
- `hardware`:
  - `device` (string; e.g. `auto|cuda|cuda:0|cpu|mps`)
  - `mixed_precision` (bool)
- `checkpoint`:
  - `save_every` (int; `0` means only best/latest)
  - `save_best` (bool)
  - `save_latest` (bool)
  - `max_keep` (int; `0` means keep all)

### 1.3 Allowed values + important constraints

These come from `validate_config()` in `src/utils/config.py` (reproducibility runs validate before training).

- `data.dataset` must be one of:
  - `MNIST`, `CIFAR10`, `FashionMNIST`, `SVHN`
  - `NMNIST`, `DVSGesture`
  - `SpeechCommands`, `WISDM`, `PrimateReaching`, `MackeyGlass`
- `model.architecture` must be one of:
  - `fc`, `fc_snn`, `r_snn`
  - `conv`, `conv_snn`
  - `local_classifier`, `recurrent`, `stllr`
  - `vgg11`, `vg11_snn`, `resnet18`
  - `ottt_conv_net`
- `trainer.name` must be one of:
  - `stsf`, `bptt`, `decolle`, `eprop`, `esd_rtrl`
  - `drtp`, `etlp`, `ostl`, `osttp`, `ottt`
  - `ell`, `fell`, `bell`, `stllr`, `stop`, `tp`, `stdp`
- `training.optimizer` must be `null` or one of: `adam`, `sgd`, `nag`, `rmsprop`.

Trainer/model coupling (common gotchas):

- `etlp` requires `model.architecture: fc`.
- `osttp` requires `model.architecture: fc`.
- `ostl` requires `model.architecture: fc` or `fc_snn`.
- `stop` requires `model.architecture` in `fc|conv|vgg11|resnet18`.
- `eprop` / `esd_rtrl` require `model.architecture: recurrent` and `model.recurrent_type` in `standard|srnn`.

Note: some trainers override the effective network architecture internally (see `src/networks/get_network.py`), e.g. `ell/fell/bell` use `local_classifier`, and `eprop/esd_rtrl` use `recurrent`.

### 1.4 Minimal reproducibility config example

Save as `configs/reproducibility/my_run.yaml`:

```yaml
experiment:
  name: "MY_REPRO_RUN"
  seed: 123
  deterministic: true
  log_dir: "./experiments"

trainer:
  name: "ottt"
  update_last: false
  update_every: 1
  seq_batch: 1

data:
  dataset: "CIFAR10"
  timesteps: 25
  data_dir: "./src/Data"
  num_workers: 4

model:
  architecture: "fc_snn"
  layer_sizes: [3072, 512, 10]
  beta: 0.9
  tau: null
  threshold: 1.0
  recurrent_type: "standard"
  quantization: false

training:
  epochs: 50
  batch_size: 128
  learning_rate: 0.001
  optimizer: "adam"
  weight_decay: 0.0
  freeze_conv: false

hardware:
  device: "auto"
  mixed_precision: false

checkpoint:
  save_every: 0
  save_best: true
  save_latest: true
  max_keep: 2
```

Quick validation tip (prints issues, if any):

```bash
python -c "from src.utils.config import load_config, validate_config; import sys; cfg=load_config(sys.argv[1]); print('\\n'.join(validate_config(cfg)) or 'OK')" configs/reproducibility/my_run.yaml
```

---

## 2) Benchmarking mode (`benchmarking.py`)

Benchmarking mode uses **two** config surfaces:

1. One *global suite* YAML (default: `configs/benchmarking.yaml`)
2. One *network blueprint* YAML per network (default directory: `configs/networks/*.yaml`)

### 2.1 Global benchmarking config (`configs/benchmarking.yaml`)

The root must be a mapping with these sections:

- `experiment`:
  - `output_dir` (string path; root folder for timestamped runs)
  - `seed` (int; reset before every experiment for comparability)
  - `deterministic` (bool)
- `execution`:
  - `epochs` (int)
  - `batch_size` (int)
  - `learning_rate` (float)
  - `timesteps` (int; default per-dataset value if not overridden in `data.dataset_defaults[...].timesteps`)
  - `device` (string; supports `auto` plus torch device strings)
  - `show_epoch_progress` (bool)
  - `max_train_batches` (int or null)
  - `max_test_batches` (int or null)
  - `single_process_data_loading` (bool; forces `num_workers=0` to avoid multiprocessing issues)
  - `continue_on_error` (bool)
  - `run_neurobench` (bool)
  - `neurobench_include_synaptic_operations` (bool; optional)
- `data`:
  - `datasets` (list of dataset names; if omitted and `dataset_defaults` is present, it defaults to the keys of `dataset_defaults`)
  - `dataset_defaults` (mapping; per-dataset defaults used to fill network shapes/classes and optionally timesteps)
- `trainers`: mapping of `trainer_name -> trainer_spec` (see next section)

Minimal example (single dataset, single trainer):

```yaml
experiment:
  output_dir: ./benchmark_results
  seed: 42
  deterministic: true

execution:
  epochs: 1
  batch_size: 128
  learning_rate: 0.001
  timesteps: 25
  device: auto
  show_epoch_progress: true
  max_train_batches: null
  max_test_batches: null
  single_process_data_loading: true
  continue_on_error: true
  run_neurobench: false

data:
  datasets: [MNIST]
  dataset_defaults:
    MNIST:
      input_shape: [1, 28, 28]
      input_size: 784
      num_classes: 10
      timesteps: 25

trainers:
  bptt:
    module: trainers.bptt_trainer
    class_name: BPTTTrainer
    enabled: true
    requires_all_tags: []
    excludes_any_tags: []
    requires_network_attrs: [reset]
    allowed_architectures: []
    training:
      requires_grad: true
      use_optimizer: true
      optimizer: adam
    params: {}
```

### 2.2 Trainer specs (inside `trainers:`)

Each entry under `trainers:` should look like:

- Required:
  - `module`: Python module import path (with `./src` on `sys.path`), e.g. `trainers.bptt_trainer`
  - `class_name`: class to import from that module, e.g. `BPTTTrainer`
- Optional scheduling/compatibility fields (used to skip invalid combos):
  - `enabled`: bool (default true)
  - `requires_all_tags`: list of BaseSNN tags; available tags: `fully_connected`, `convolutional`, `recurrent`, `single_layer`, `vgg`
  - `excludes_any_tags`: list of tags
  - `requires_network_attrs`: list of attribute names that must exist on the network object
  - `allowed_architectures`: list of allowed network architectures (compared to the network YAML `model.architecture`)
- Optional training behavior:
  - `training.requires_grad`: bool (wraps training in `torch.set_grad_enabled()`)
  - `training.use_optimizer`: bool
  - `training.optimizer`: `adam|sgd|nag|rmsprop` (only used when `use_optimizer: true`)
- Optional trainer constructor kwargs:
  - `params`: mapping passed to the trainer constructor (keys not accepted by the trainer are ignored unless it accepts `**kwargs`)

### 2.3 Network blueprint configs (`configs/networks/*.yaml`)

`benchmarking.py` loads every `*.yaml` / `*.yml` file in `configs/networks/`.

Schema (root mapping):

- `name` (string; defaults to file stem)
- `description` (string; optional)
- `enabled` (bool; default true)
- `model` (mapping):
  - `architecture` (required): one of `fc_snn|r_snn|conv_snn|vg11_snn` (these are the only architectures in `benchmarking.py`'s factory)
  - `layer_sizes` (list[int]; optional but recommended for reporting; may be adjusted by `dataset_defaults`)
  - `quantization` (bool; forwarded to trainers as `quant`)
  - other keys are allowed, but `benchmarking.py` only reads `architecture`, `layer_sizes` (optional), and `quantization`
- `network_kwargs` (mapping; **required**): kwargs passed directly to the network class constructor
- `dataset_overrides` (mapping; optional):
  - keys are dataset names (`MNIST`, `CIFAR10`, ...)
  - each value can contain `model:` and/or `network_kwargs:` which are deep-merged onto the base config
- `tags` (optional; informational only; benchmarking derives actual tags from the instantiated network)

Important: `data.dataset_defaults` in the global config can automatically fill/override:

- `network_kwargs.in_shape` based on `input_size` / `input_shape`
- `network_kwargs.num_classes` based on `num_classes`
- `model.layer_sizes[0]` / `model.layer_sizes[-1]` for `fc_snn` and `r_snn`

Example (simplified `fc_snn` blueprint):

```yaml
name: fc_snn
enabled: true

model:
  architecture: fc_snn
  layer_sizes: [784, 256, 10]
  quantization: false

network_kwargs:
  in_shape: [784]
  num_classes: 10
  hidden_sizes: [256]
  beta: 0.9
  threshold: 1.0

dataset_overrides:
  CIFAR10:
    model:
      layer_sizes: [3072, 512, 10]
    network_kwargs:
      in_shape: [3072]
      hidden_sizes: [512]
```

### 2.4 Quick sanity checks

- Benchmark schedule preview (writes `experiment_manifest.json` with skip reasons):
  ```bash
  python benchmarking.py --dry-run
  ```

- Reproducibility schedule preview (writes `experiment_manifest.json` under `benchmark_results/reproducibility/...`):
  ```bash
  python reproducibility.py --dry-run
  ```
