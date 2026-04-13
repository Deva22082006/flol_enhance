"""
scripts/ablation.py
====================
Ablation study for FLOL-FS.

Trains 4 variants to prove each component contributes:
  1. baseline     — L1 + LOL only (no SSIM, no Freq, no Perc)
  2. +ssim        — L1 + LOL + SSIM
  3. +freq        — L1 + LOL + SSIM + FreqLoss       [our contribution 1]
  4. +freq+perc   — L1 + LOL + SSIM + Freq + Perc    [our full model]

This generates Table II of the paper.

Usage:
  python scripts/ablation.py --train_dir LOL/train --test_dir LOL/test --epochs 100
"""

import sys
import subprocess
import json
import argparse
from pathlib import Path


def run_variant(base_cmd: str, tag: str, extra_flags: str = ''):
    """Run a single training variant."""
    cmd = f"{base_cmd} --tag {tag} {extra_flags}"
    print(f"\n{'='*60}")
    print(f"Running: {tag}")
    print(f"Command: {cmd}")
    print('='*60)
    subprocess.run(cmd, shell=True, check=True)


def read_best(ckpt_dir: str, tag: str) -> dict:
    """Read best result from log.json."""
    log = Path(ckpt_dir) / tag / 'log.json'
    if not log.exists():
        return {'val_psnr': 0.0, 'val_ssim': 0.0}
    with open(log) as f:
        history = json.load(f)
    return max(history, key=lambda r: r['val_psnr'])


def ablation(args):
    base = (
        f"python scripts/train.py"
        f" --train_dir {args.train_dir}"
        f" --test_dir  {args.test_dir}"
        f" --epochs    {args.epochs}"
        f" --batch_size {args.batch_size}"
        f" --patch_size {args.patch_size}"
        f" --ckpt_dir  {args.ckpt_dir}"
    )

    variants = [
        # (tag,           extra_flags)
        ('baseline',      '--no_freq_loss --no_perc_loss --no_color_loss'),
        ('plus_ssim',     '--no_freq_loss --no_perc_loss'),
        ('plus_freq',     '--no_perc_loss'),
        ('plus_freq_perc', ''),   # full model
    ]

    for tag, flags in variants:
        if args.resume:
            flags += ' --resume'
        run_variant(base, tag, flags)

    # Print results table
    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)
    print(f"{'Model':<20} {'PSNR':>8} {'SSIM':>8}")
    print("-" * 40)

    labels = {
        'baseline'      : 'Baseline (L1 only)',
        'plus_ssim'     : '+ SSIM',
        'plus_freq'     : '+ Frequency Loss',
        'plus_freq_perc': '+ Freq + Perceptual',
    }

    for tag, _ in variants:
        best = read_best(args.ckpt_dir, tag)
        label = labels.get(tag, tag)
        print(f"  {label:<20} {best['val_psnr']:>7.2f}  {best['val_ssim']:>7.4f}")

    print("=" * 60)
    print("\nThis table is Table II of your paper.")


def get_args():
    p = argparse.ArgumentParser('FLOL-FS Ablation Study')
    p.add_argument('--train_dir',  default='LOL/train')
    p.add_argument('--test_dir',   default='LOL/test')
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--patch_size', type=int, default=128)
    p.add_argument('--ckpt_dir',   default='checkpoints')
    p.add_argument('--resume',     action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    ablation(get_args())
