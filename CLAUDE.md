# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`stocknet` is a PyTorch-based financial time series prediction framework supporting both supervised learning (seq2seq Transformers, LSTMs) and reinforcement learning (Forex trading agents). Training is configuration-driven via JSON files.

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

`stocknet/main.py:train_from_config()` is the single entry point. It reads a JSON config and wires together datasets, models, optimizers, and trainers. Config structure:

```json
{
  "dataset": { "key": "seq2seq", "source": ["data.csv"], "args": { "columns": [...], "observation_length": 60 } },
  "model":   { "key": "Seq2SeqTransformer", "model_name": "my_model", "configs": "./params/*.json" },
  "training": { "optimizer": { "key": "AdamW", "lr": 0.01 }, "scheduler": {...}, "loss": { "key": "MSELoss" },
                "batch_size": [16, 32, 64], "epoch": 300, "patience": 3, "device": "cuda" },
  "log":     { "path": "./logs" }
}
```

Config args support variable substitution (`$dataset.columns`) to reference sibling config sections.

### Factory Pattern

All major components use factory modules that load classes by string key:
- `stocknet/datasets/factory.py` — resolves dataset type (e.g. `"seq2seq"`)
- `stocknet/nets/factory.py` — resolves model class by name (case-insensitive, e.g. `"Seq2SeqTransformer"`)
- `stocknet/trainer/factory.py` — resolves trainer/evaluator functions; supports loading custom trainer modules from user-defined files

### Custom Extensions (`predictions/transformer/custom/`)

Users can provide their own dataset, loss, and trainer implementations. The trainer factory dynamically imports these modules. See `predictions/transformer/custom/` for examples.

### Key Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `stocknet/main.py` | Training orchestration, early stopping, checkpointing |
| `stocknet/datasets/` | Data loading (CSV via pandas), train/val/test split (80/20), preprocessing (Diff, MinMax), batch generation |
| `stocknet/nets/` | Model definitions: LSTM, Dense, AutoEncoder, Seq2SeqTransformer with positional encodings |
| `stocknet/trainer/` | `sltrainer.py` for supervised seq2seq; `rltrainer.py` for PFRL-based RL agents |
| `stocknet/envs/` | Gym-compatible Forex/tick trading environments; `render/graph.py` for matplotlib visualization |
| `stocknet/logger.py` | Checkpoint save/load, device detection (`get_device()`) |

### Reinforcement Learning

RL training uses [PFRL](https://github.com/pfnet/pfrl) (`gym==0.22` required). Environments are in `stocknet/envs/` and inherit from gym's base environment. RL trainer entry point is `stocknet/trainer/rltrainer.py`.

### Example Configs

`predictions/transformer/` contains ready-to-run training configs:
- `baseline.json` — standard transformer
- `baseline_scaling.json` — with MinMax scaling
- `cd_scaling.json` — cluster differentiation variant
- `cid_scaling.json` — cluster ID variant
- `weeklytime_scaling.json` — weekly time-aware variant
