"""
scripts/evaluate.py
====================
Evaluation script for FLOL-FS.
Computes PSNR and SSIM on full-resolution test images.

Usage:
  python scripts/evaluate.py --test_dir LOL/test --ckpt checkpoints/flol_fs/best.pth
  python scripts/evaluate.py --test_dir LOL/test --ckpt checkpoints/flol_fs/best.pth --save_dir outputs/eval
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.flol_enhanced import FLOL_CA


# ─────────────────────────────────────────────
# METRICS (proper implementations)
# ─────────────────────────────────────────────
def psnr_np(pred: np.ndarray, target: np.ndarray) -> float:
    """PSNR on uint8 numpy arrays [0, 255]."""
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_np(pred: np.ndarray, target: np.ndarray) -> float:
    """SSIM on uint8 numpy arrays [0, 255]. Averaged over RGB channels."""
    from skimage.metrics import structural_similarity as sk_ssim
    # Convert to float [0,1]
    p = pred.astype(np.float64)   / 255.0
    t = target.astype(np.float64) / 255.0
    return sk_ssim(p, t, data_range=1.0, channel_axis=2)


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) float tensor [0,1] to uint8 (H, W, C) numpy."""
    arr = t.detach().cpu().clamp(0, 1).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr.transpose(1, 2, 0)   # (H, W, C)


def load_image(path: Path) -> torch.Tensor:
    """Load image as float tensor (C, H, W) in [0, 1]."""
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# ─────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    ck    = torch.load(args.ckpt, map_location=device)
    cfg   = ck.get('config', {})
    model = FLOL_CA(
        channels = cfg.get('channels', 16),
        use_ca   = not cfg.get('no_ca', False),
    ).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    print(f"Loaded: {args.ckpt}")

    # Test directory
    test_dir  = Path(args.test_dir)
    low_dir   = test_dir / 'low'
    high_dir  = test_dir / 'high'

    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    low_files  = sorted([f for f in low_dir.iterdir()
                         if f.suffix.lower() in exts])
    high_files = sorted([f for f in high_dir.iterdir()
                         if f.suffix.lower() in exts])

    assert len(low_files) == len(high_files)

    # Save directory
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    psnr_list = []
    ssim_list = []
    t0 = time.time()

    for i, (lp, hp) in enumerate(zip(low_files, high_files)):
        low  = load_image(lp).unsqueeze(0).to(device)   # (1, C, H, W)
        high = load_image(hp)                            # (C, H, W)

        x_hat, _ = model(low)
        x_hat = x_hat.squeeze(0)                         # (C, H, W)

        # Convert to uint8 for metrics
        pred_np = tensor_to_uint8(x_hat)
        gt_np   = tensor_to_uint8(high)

        p = psnr_np(pred_np, gt_np)
        s = ssim_np(pred_np, gt_np)
        psnr_list.append(p)
        ssim_list.append(s)

        print(f"  [{i+1:02d}/{len(low_files)}] {lp.name:30s} "
              f"PSNR: {p:.2f} dB   SSIM: {s:.4f}")

        if save_dir:
            Image.fromarray(pred_np).save(save_dir / lp.name)

    elapsed = time.time() - t0
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print("\n" + "─" * 50)
    print(f"  Avg PSNR : {avg_psnr:.2f} dB")
    print(f"  Avg SSIM : {avg_ssim:.4f}")
    print(f"  Images   : {len(low_files)}  |  Total time: {elapsed:.1f}s")
    if save_dir:
        print(f"  Saved to : {save_dir}")
    print("─" * 50)

    return avg_psnr, avg_ssim


def get_args():
    p = argparse.ArgumentParser('Evaluate FLOL-FS')
    p.add_argument('--test_dir', required=True)
    p.add_argument('--ckpt',     required=True)
    p.add_argument('--save_dir', default=None)
    return p.parse_args()


if __name__ == '__main__':
    evaluate(get_args())
