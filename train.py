"""
scripts/train.py
=================
Training script for FLOL-FS.

Usage:
  # Standard training:
  python scripts/train.py --train_dir LOL/train --test_dir LOL/test

  # Resume:
  python scripts/train.py --train_dir LOL/train --test_dir LOL/test --resume --tag flol_fs

  # Ablation (no frequency loss):
  python scripts/train.py --train_dir LOL/train --test_dir LOL/test --no_freq_loss --tag no_freq

  # Quick test with synthetic data:
  python scripts/train.py --synthetic --epochs 5
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.flol_enhanced import FLOL_CA
from models.losses        import FLOLLoss
from data.dataset         import LOLDataset, SyntheticLOLDataset, build_dataloaders


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def ssim_fast(pred: torch.Tensor, target: torch.Tensor,
              C1: float = 1e-4, C2: float = 9e-4) -> float:
    """Fast SSIM approximation for logging (no windowing)."""
    mu1, mu2 = pred.mean(), target.mean()
    s1  = pred.var()
    s2  = target.var()
    s12 = ((pred - mu1) * (target - mu2)).mean()
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    return (num / den).item()


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
@torch.no_grad()
def validate(model: torch.nn.Module,
             loader: DataLoader,
             device: torch.device):
    model.eval()
    total_psnr = total_ssim = 0.0
    n = 0
    for low, high in loader:
        low, high = low.to(device), high.to(device)
        x_hat, _ = model(low)
        total_psnr += psnr(x_hat, high)
        total_ssim += ssim_fast(x_hat, high)
        n += 1
    return total_psnr / max(n, 1), total_ssim / max(n, 1)


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train(args):
    # ── Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── Directories
    tag      = args.tag or 'flol_fs'
    ckpt_dir = Path(args.ckpt_dir) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path  = ckpt_dir / 'log.json'
    best_ckpt = ckpt_dir / 'best.pth'
    last_ckpt = ckpt_dir / 'last.pth'

    # ── Data
    if args.synthetic:
        print("Using synthetic dataset (test mode)")
        train_ds = SyntheticLOLDataset(n_samples=args.n_synthetic,
                                       patch_size=args.patch_size)
        val_ds   = SyntheticLOLDataset(n_samples=20,
                                       patch_size=args.patch_size)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=1,
                                  shuffle=False, num_workers=0)
    else:
        train_loader, val_loader = build_dataloaders(
            train_dir   = args.train_dir,
            test_dir    = args.test_dir,
            patch_size  = args.patch_size,
            batch_size  = args.batch_size,
            num_workers = 0,
        )

    # ── Model
    use_ca = not args.no_ca
    model  = FLOL_CA(channels=args.channels, use_ca=use_ca).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model : FLOL-FS | CA={use_ca} | Params: {n_params/1e3:.1f}K")

    # ── Loss
    freq_w  = 0.0  if args.no_freq_loss  else 0.1
    perc_w  = 0.0  if args.no_perc_loss  else 0.05
    color_w = 0.0  if args.no_color_loss else 0.01

    criterion = FLOLLoss(
        lambda_lol   = 1.0,
        lambda_ssim  = 0.1,
        lambda_freq  = freq_w,
        lambda_perc  = perc_w,
        lambda_color = color_w,
    )

    print(f"Loss  : L1 + LOL + SSIM"
          f" + Freq(λ={freq_w})"
          f" + Perc(λ={perc_w})"
          f" + Color(λ={color_w})")

    # ── Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, betas=(0.9, 0.999),
                           weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=args.epochs, eta_min=1e-6)

    # ── Resume
    start_epoch = 0
    best_psnr   = 0.0
    history     = []

    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch = ck['epoch'] + 1
        best_psnr   = ck.get('best_psnr', 0.0)
        history     = ck.get('history', [])
        print(f"Resumed from epoch {start_epoch} "
              f"(best PSNR: {best_psnr:.2f} dB)")

    # ── Training
    print(f"\nStarting training — epochs: {args.epochs} | "
          f"batch: {args.batch_size} | lr: {args.lr}")
    print("─" * 70)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        sums = {k: 0.0 for k in
                ['total', 'l1', 'lol', 'ssim', 'freq', 'perc', 'color']}

        for low, high in train_loader:
            low, high = low.to(device), high.to(device)

            x_hat, x_lol = model(low)
            losses = criterion(x_hat, x_lol, high)

            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in sums:
                v = losses.get(k, 0.0)
                sums[k] += v.item() if hasattr(v, 'item') else float(v)

        scheduler.step()

        n   = len(train_loader)
        avg = {k: v / n for k, v in sums.items()}
        val_psnr, val_ssim = validate(model, val_loader, device)
        elapsed = time.time() - t0

        row = {
            'epoch'    : epoch + 1,
            'loss'     : round(avg['total'], 4),
            'l1'       : round(avg['l1'],    4),
            'ssim_loss': round(avg['ssim'],  4),
            'freq'     : round(avg['freq'],  4),
            'perc'     : round(avg['perc'],  4),
            'color'    : round(avg['color'], 4),
            'val_psnr' : round(val_psnr, 2),
            'val_ssim' : round(val_ssim, 4),
            'lr'       : round(optimizer.param_groups[0]['lr'], 7),
            'time_s'   : round(elapsed, 1),
        }
        history.append(row)

        print(f"Ep {epoch+1:03d}/{args.epochs} | "
              f"Loss {avg['total']:.4f} "
              f"(L1:{avg['l1']:.4f} "
              f"SSIM:{avg['ssim']:.4f} "
              f"Freq:{avg['freq']:.4f} "
              f"Perc:{avg['perc']:.4f} "
              f"Col:{avg['color']:.4f}) | "
              f"PSNR {val_psnr:.2f} dB  "
              f"SSIM {val_ssim:.4f} | "
              f"{elapsed:.0f}s")

        # Save
        ck = {
            'epoch'     : epoch,
            'model'     : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'best_psnr' : best_psnr,
            'history'   : history,
            'config'    : vars(args),
        }
        torch.save(ck, last_ckpt)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            ck['best_psnr'] = best_psnr
            torch.save(ck, best_ckpt)
            print(f"  ✓ New best PSNR: {best_psnr:.2f} dB "
                  f"→ saved to {best_ckpt}")

        with open(log_path, 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")
    return history


# ─────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser('Train FLOL-FS')

    # Data
    p.add_argument('--train_dir',   default='LOL/train')
    p.add_argument('--test_dir',    default='LOL/test')
    p.add_argument('--synthetic',   action='store_true')
    p.add_argument('--n_synthetic', type=int,   default=200)
    p.add_argument('--patch_size',  type=int,   default=256)

    # Training
    p.add_argument('--epochs',      type=int,   default=200)
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--lr',          type=float, default=4e-4)
    p.add_argument('--channels',    type=int,   default=16)

    # Ablation flags
    p.add_argument('--no_ca',         action='store_true',
                   help='Disable Channel Attention')
    p.add_argument('--no_freq_loss',  action='store_true',
                   help='Disable Frequency Loss (ablation)')
    p.add_argument('--no_perc_loss',  action='store_true',
                   help='Disable Perceptual Loss (ablation)')
    p.add_argument('--no_color_loss', action='store_true',
                   help='Disable Color Loss (ablation)')

    # Misc
    p.add_argument('--ckpt_dir', default='checkpoints')
    p.add_argument('--resume',   action='store_true')
    p.add_argument('--tag',      default='flol_fs')

    return p.parse_args()


if __name__ == '__main__':
    train(get_args())
