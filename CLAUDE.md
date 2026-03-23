# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CryptoMamba is a Mamba-based (State Space Model) architecture for cryptocurrency price prediction. It includes implementations of CryptoMamba and baselines (LSTM, Bi-LSTM, GRU, iTransformer, S-Mamba), two trading algorithms (vanilla and smart), and scripts for training, evaluation, and trading simulation.

## Setup

```bash
pip install -r requirements.txt
```

`mamba_ssm` requires CUDA. If installation fails, follow the [mamba-ssm GitHub instructions](https://github.com/state-spaces/mamba).

## Commands

**Train a model:**
```bash
python3 scripts/training.py --config cmamba_v
```
Config names are filenames (without `.yaml`) from `configs/training/`. Suffix `_v` = with volume, `_nv` = no volume.

**Evaluate a model:**
```bash
python scripts/evaluation.py --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt
```

**Run trading simulation:**
```bash
python scripts/simulate_trade.py --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt --split test --trade_mode smart
```
`--split`: `train`, `val`, or `test`. `--trade_mode`: `smart`, `vanilla`, or `smart_w_short`.

**Predict next day price:**
```bash
python scripts/one_day_pred.py --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt --date 2024-09-16
```

**Monitor training:**
```bash
tensorboard --logdir logs/
```

## Architecture

### Config-driven instantiation
Training configs (`configs/training/*.yaml`) reference a `model` key (e.g., `CMamba_v2`) which maps to a model config path via `configs/models/archs.yaml`. Model configs specify a `target` (Python class path) and `params`, instantiated via `utils/io_tools.instantiate_from_config()`.

### Key layers
- `pl_modules/base_module.py` — `BaseModule(pl.LightningModule)`: shared train/val/test logic, loss selection (rmse/mse/mae/mape), normalization, and optimizer config. All model modules inherit from this.
- `pl_modules/cmamba_module.py` — `CryptoMambaModule`: wraps `CMamba` from `models/cmamba.py`. Requires `window_size == hidden_dims[0]`.
- `models/cmamba.py` — `CMamba`: stacked `CMBlock` layers (each containing a custom `Mamba` SSM + optional MLP), followed by a linear `post_process` that collapses the feature dimension to 1.
- `pl_modules/data_module.py` — `CMambaDataModule`: loads/splits data via `data_utils/dataset.py:DataConverter` and wraps it in `CMambaDataset`.

### Data pipeline
`DataConverter` reads raw OHLCV CSV data, resamples to the target time resolution (`jumps` in seconds), and splits by explicit date intervals defined in `configs/data_configs/mode_1.yaml`. Processed splits are cached as `data/*/train.csv`, `val.csv`, `test.csv`. `DataTransform` normalizes features min-max and produces batches with keys `features`, `Close`, `Close_old`.

### Model input/output
Input shape: `(batch, window_size, num_features)` — a sliding window of OHLCV (+ optional volume/additional features). Output: scalar predicted close price per sample. In `diff` mode, the model predicts a delta added to `Close_old`.

### Config naming conventions
- `_v` / `_nv`: with/without volume as a feature
- `hidden_dims` in model config: list of sequence lengths through SSM blocks (first element must match `window_size`)
- `layer_density`: number of `CMBlock` layers per stage
