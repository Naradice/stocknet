# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`stocknet` is a PyTorch-based financial time series prediction framework supporting supervised learning (seq2seq Transformers, LSTMs, GANs) and reinforcement learning (Forex trading agents). Training is configuration-driven via JSON or YAML files.

## Setup

```bash
# Python 3.8.0 (see .python-version)
pip install -r requirements.txt
pip install -e .
```

Uses a git submodule (`finance_client`) for financial data handling:
```bash
git submodule update --init --recursive
```

## Commands

### Run tests
```bash
python -m unittest discover tests

# Single test file
python -m unittest tests.test_datasets

# Single test method
python -m unittest tests.test_datasets.TestDatasets.test_ohlc
```

### Train a model
```bash
# From the predictions/transformer/ directory
python main.py

# Programmatically
python -c "import stocknet; stocknet.train_from_config('path/to/config.json')"
```

### Lint / Format
- Linter: **flake8** with `max-line-length=150`, ignoring E501, E203, E731
- Formatter: **black** with `--line-length=150`

## Architecture

### Configuration-Driven Training Pipeline

`stocknet/main.py:train_from_config()` is the single entry point. Supports JSON and YAML configs. It wires together datasets, models, optimizers, and trainers via a nested generator pattern — each layer (`datasets`, `models`, `batch_sizes`) is a generator, enabling grid search over combinations.

```json
{
  "dataset": { "key": "seq2seq", "source": ["data.csv"], "args": { "columns": [...], "observation_length": 60 } },
  "model":   { "key": "Seq2SeqTransformer", "model_name": "my_model", "configs": "./params/*.json" },
  "training": { "optimizer": { "key": "AdamW", "lr": 0.01 }, "scheduler": { "key": "StepLR", "step_size": 10 },
                "loss": { "key": "MSELoss" }, "batch_size": [16, 32, 64], "epoch": 300, "patience": 3, "device": "cuda" },
  "log":     { "path": "./logs" }
}
```

Config args support variable substitution (`$dataset.columns`) to reference sibling config sections.

**Model params can be specified two ways:**
- `"configs": "./params/*.json"` — glob of separate param files (each yields one model version)
- `"params": {...}` or `"params": [{...}, {...}]` — inline param dict or list of dicts

### Factory Pattern

All major components use factory modules that resolve classes by string key (case-insensitive):
- `stocknet/datasets/factory.py:load()` — routes by key prefix: `seq2seq*` → pandas datasets, `fc*` → finance_client datasets, other → custom
- `stocknet/nets/factory.py:load_models()` — resolves model class from `stocknet.nets` by name
- `stocknet/trainer/factory.py:load_trainers()` — selects trainer by model key; falls back to custom module if `"trainer"` key present in training config

### Dataset Keys

| Key | Class | Description |
|-----|-------|-------------|
| `seq2seq` | `FeatureDataset` | Standard (src, tgt) pairs from CSV |
| `seq2seq_time` | `TimeFeatureDataset` | Adds time-of-day/week features |
| `seq2seq_diff_id` | `DiffIDDS` | Differenced + cluster ID features |
| `fc_*` | `ClientDataset` / `FrameConvertDataset` / etc. | Live data via `finance_client` |

### Available Models (`stocknet/nets/`)

| Class | Trainer |
|-------|---------|
| `Seq2SeqTransformer` | `sltrainer.seq2seq_train/eval` (auto-selected) |
| `LSTM` | `sltrainer.Trainer` callable |
| `MeanVarianceTransformer`, `MeanVarianceLSTM` | custom trainer via config |
| `TimeGAN` | `gantrainer.gan_train/eval` (auto-selected) |
| `AELinearModel`, `SimpleDense`, `ConvDense16` | `sltrainer.Trainer` callable |

`TimeGAN.parameters()` returns only generator parameters — the discriminator optimizer is created lazily inside `gantrainer`.

### Custom Extensions (`predictions/transformer/custom/`)

Place custom modules in subdirectories under `custom/`:
- `custom/dataset/` — custom dataset classes (referenced by key in config)
- `custom/trainer/` — custom train/eval functions (referenced as `"trainer": {"train_key": "trainer.module.func"}`)
- `custom/loss/` — custom loss classes (referenced as `"loss": {"key": "loss.module.ClassName"}`)

Keys use dotted notation: `"trainer.label.train_func"` → imports `custom/trainer/label.py` and gets `train_func`.

### Preprocessing Pipeline

Datasets accept a `processes` list in config. `datasets/utils.py:load_fprocesses()` builds a chain of transforms (e.g. `Diff`, `MinMax`) applied before batching. Processes can be specified per-file or globally in `args`.

### Key Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `stocknet/main.py` | Training orchestration, early stopping, checkpointing |
| `stocknet/datasets/` | Data loading (CSV via pandas), train/val/test split (80/20), preprocessing, batch generation |
| `stocknet/nets/` | Model definitions: LSTM, Dense, AutoEncoder, Seq2SeqTransformer, MeanVariance variants, TimeGAN |
| `stocknet/trainer/sltrainer.py` | Supervised seq2seq training/evaluation |
| `stocknet/trainer/gantrainer.py` | GAN training (manages discriminator optimizer internally) |
| `stocknet/trainer/rltrainer.py` | PFRL-based RL agents |
| `stocknet/envs/` | Gym-compatible Forex/tick trading environments |
| `stocknet/logger.py` | Checkpoint save/load, device detection (`get_device()`) |

### Reinforcement Learning

RL training uses [PFRL](https://github.com/pfnet/pfrl) (`gym==0.22` required). Environments inherit from gym's base and are in `stocknet/envs/`. RL trainer entry point is `stocknet/trainer/rltrainer.py`.

### Example Configs (`predictions/transformer/`)

- `baseline.json` — standard transformer
- `baseline_scaling.json` — with MinMax scaling
- `cd_scaling.json` — cluster differentiation variant
- `cid_scaling.json` — cluster ID variant
- `weeklytime_scaling.json` — weekly time-aware variant
- `meanvar.json` — mean-variance output transformer
