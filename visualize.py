"""
scripts/visualize.py
=====================
Generates comparison images for the paper.
Creates a side-by-side grid: Input | Enhanced | Ground Truth

Usage:
  python scripts/visualize.py --ckpt checkpoints/flol_fs/best.pth --test_dir LOL/test
"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.flol_enhanced import FLOL_CA


def load_image(path: Path, max_size: int = 400) -> torch.Tensor:
    """Load and optionally resize image."""
    img = Image.open(path).convert('RGB')
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.detach().cpu().clamp(0, 1).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr.transpose(1, 2, 0))


def add_label(img: Image.Image, text: str,
              color: tuple = (255, 255, 255)) -> Image.Image:
    """Add text label to bottom of image."""
    out = img.copy()
    d   = ImageDraw.Draw(out)
    w, h = out.size
    # Simple text at bottom
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        fnt = ImageFont.load_default()
    d.rectangle([0, h - 22, w, h], fill=(0, 0, 0))
    d.text((4, h - 20), text, fill=color, font=fnt)
    return out


@torch.no_grad()
def visualize(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    ck    = torch.load(args.ckpt, map_location=device)
    cfg   = ck.get('config', {})
    model = FLOL_CA(
        channels = cfg.get('channels', 16),
        use_ca   = not cfg.get('no_ca', False),
    ).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    print(f"Model loaded: {args.ckpt}")

    # Find test images
    test_dir  = Path(args.test_dir)
    low_dir   = test_dir / 'low'
    high_dir  = test_dir / 'high'

    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    low_files  = sorted([f for f in low_dir.iterdir()
                         if f.suffix.lower() in exts])[:args.n_images]
    high_files = sorted([f for f in high_dir.iterdir()
                         if f.suffix.lower() in exts])[:args.n_images]

    # Generate comparison grid
    rows = []
    patch = args.patch_size

    for lp, hp in zip(low_files, high_files):
        low  = load_image(lp, max_size=patch)
        high = load_image(hp, max_size=patch)

        # Inference
        x_hat, _ = model(low.unsqueeze(0).to(device))
        x_hat = x_hat.squeeze(0)

        # Resize all to same size for grid
        def resize_tensor(t, size):
            return torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(size, size), mode='bilinear',
                align_corners=False
            ).squeeze(0)

        low   = resize_tensor(low,   patch)
        x_hat = resize_tensor(x_hat, patch)
        high  = resize_tensor(high,  patch)

        # To PIL with labels
        img_low  = add_label(tensor_to_pil(low),   'Input (Dark)')
        img_enh  = add_label(tensor_to_pil(x_hat), 'FLOL-FS (Ours)')
        img_gt   = add_label(tensor_to_pil(high),  'Ground Truth')

        # Horizontal strip for this image
        gap   = 4
        strip = Image.new('RGB', (patch * 3 + gap * 2, patch), (40, 40, 40))
        strip.paste(img_low,  (0,                 0))
        strip.paste(img_enh,  (patch + gap,        0))
        strip.paste(img_gt,   (patch * 2 + gap*2, 0))
        rows.append(strip)

    # Stack rows vertically
    row_gap  = 6
    total_h  = len(rows) * patch + (len(rows) - 1) * row_gap
    total_w  = patch * 3 + 4 * 2

    canvas = Image.new('RGB', (total_w, total_h), (20, 20, 20))
    for i, row in enumerate(rows):
        canvas.paste(row, (0, i * (patch + row_gap)))

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'comparison.png'
    canvas.save(out_path)
    print(f"Comparison saved: {out_path}")
    print(f"  {len(rows)} images | "
          f"Size: {total_w}×{total_h}px")


def get_args():
    p = argparse.ArgumentParser('Visualize FLOL-FS')
    p.add_argument('--ckpt',       required=True)
    p.add_argument('--test_dir',   default='LOL/test')
    p.add_argument('--out_dir',    default='outputs')
    p.add_argument('--n_images',   type=int, default=6)
    p.add_argument('--patch_size', type=int, default=256)
    return p.parse_args()


if __name__ == '__main__':
    visualize(get_args())
