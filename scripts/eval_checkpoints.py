"""
Evaluate all checkpoints in the latest (or specified) itransformer version directory
and save a prediction PNG alongside each checkpoint.

Usage:
    python scripts/eval_checkpoints.py
    python scripts/eval_checkpoints.py --config itransformer_v
    python scripts/eval_checkpoints.py --ckpt_dir logs/itransformer/version_5/checkpoints
"""
import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import glob
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from argparse import ArgumentParser
from datetime import datetime

import seaborn as sns
import pytorch_lightning as pl
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import io_tools
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform
from scripts.evaluation import load_model, run_model

sns.set_theme(style='whitegrid', context='paper', font_scale=3)

ROOT = io_tools.get_root(__file__, num_returns=2)


def find_latest_ckpt_dir(config_name: str) -> str:
    base = os.path.join(ROOT, 'logs', config_name)
    versions = sorted(
        [d for d in glob.glob(os.path.join(base, 'version_*')) if os.path.isdir(d)],
        key=lambda d: int(d.rsplit('_', 1)[-1])
    )
    if not versions:
        raise FileNotFoundError(f'No version directories found under {base}')
    return os.path.join(versions[-1], 'checkpoints')


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='itransformer_v')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint directory. Defaults to latest version.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=23)
    return parser.parse_args()


def plot_and_save(model, data_module, normalize, out_path):
    factors = data_module.factors if normalize else None

    train_loader = data_module.train_dataloader()
    val_loader   = data_module.val_dataloader()
    test_loader  = data_module.test_dataloader()

    titles = ['Train', 'Val', 'Test']
    colors = ['red', 'green', 'magenta']

    all_timestamps, all_targets = [], []
    plt.figure(figsize=(20, 10))

    for title, loader, color in zip(titles, [train_loader, val_loader, test_loader], colors):
        timestamps, targets, preds, mse, mape, l1 = run_model(model, loader, factors)
        all_timestamps += timestamps
        all_targets    += list(targets)
        rmse = np.sqrt(mse)
        print(f'  {title:5s}  RMSE={rmse:.3f}  MAPE={mape:.5f}  MAE={l1:.3f}')
        sns.lineplot(x=timestamps, y=preds, color=color, linewidth=2.5, label=title)

    sns.lineplot(x=all_timestamps, y=all_targets, color='blue', zorder=0, linewidth=2.5, label='Target')
    plt.legend()
    plt.xlim([all_timestamps[0], all_timestamps[-1]])
    plt.xticks(rotation=30)
    ax = plt.gca()
    max_val = max(all_targets)
    if max_val > 1000:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}K'.format(x / 1000)))
        plt.ylabel('Price (USD)')
    else:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
        plt.ylabel('Price (BTC)')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}')


if __name__ == '__main__':
    args = get_args()
    pl.seed_everything(args.seed)

    training_config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{args.config.lower()}.yaml')

    log_name = training_config.get('name', args.config)
    ckpt_dir = args.ckpt_dir or find_latest_ckpt_dir(log_name)
    print(f'Checkpoint directory: {ckpt_dir}')

    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, '*.ckpt')))
    if not ckpt_files:
        print('No .ckpt files found.')
        sys.exit(1)
    print(f'Found {len(ckpt_files)} checkpoints.\n')
    data_config     = io_tools.load_config_from_yaml(
        f"{ROOT}/configs/data_configs/{training_config.get('data_config')}.yaml"
    )
    use_volume = training_config.get('use_volume', False)
    train_transform = DataTransform(is_train=True,  use_volume=use_volume,
                                    additional_features=training_config.get('additional_features', []))
    val_transform   = DataTransform(is_train=False, use_volume=use_volume,
                                    additional_features=training_config.get('additional_features', []))
    test_transform  = DataTransform(is_train=False, use_volume=use_volume,
                                    additional_features=training_config.get('additional_features', []))

    for ckpt_path in ckpt_files:
        ckpt_name = pathlib.Path(ckpt_path).stem
        print(f'Evaluating: {ckpt_name}')

        model, normalize = load_model(training_config, ckpt_path)

        data_module = CMambaDataModule(
            data_config,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            batch_size=args.batch_size,
            distributed_sampler=False,
            num_workers=args.num_workers,
            normalize=normalize,
            window_size=model.window_size,
        )

        out_path = os.path.join(ckpt_dir, f'{ckpt_name}.png')
        plot_and_save(model, data_module, normalize, out_path)
        print()
