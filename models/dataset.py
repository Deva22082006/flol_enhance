"""
data/dataset.py
================
Dataset loader for LOL (Low-Light) dataset.

LOL dataset structure:
  LOL/
  ├── train/
  │   ├── low/    ← dark input images
  │   └── high/   ← ground truth normal-light images
  └── test/
      ├── low/
      └── high/

Supports: LOLv1 (485 train / 15 test)
          LOLv2-real (689 train / 100 test) — recommended
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────
# REAL DATASET
# ─────────────────────────────────────────────────────────────
class LOLDataset(Dataset):
    """
    Loads paired low/high images from LOL dataset directory.

    Args:
        root_dir   : path to LOL/train or LOL/test
        patch_size : random crop size during training (None = full image)
        augment    : enable random flip/rotation augmentation
    """

    def __init__(self,
                 root_dir:   str,
                 patch_size: Optional[int] = 128,
                 augment:    bool = True):
        super().__init__()
        self.root_dir   = Path(root_dir)
        self.patch_size = patch_size
        self.augment    = augment

        self.low_dir  = self.root_dir / 'low'
        self.high_dir = self.root_dir / 'high'

        assert self.low_dir.exists(),  f"Missing: {self.low_dir}"
        assert self.high_dir.exists(), f"Missing: {self.high_dir}"

        # Match filenames between low and high directories
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        low_files  = sorted([f for f in self.low_dir.iterdir()
                              if f.suffix.lower() in exts])
        high_files = sorted([f for f in self.high_dir.iterdir()
                              if f.suffix.lower() in exts])

        assert len(low_files) == len(high_files), \
            f"Mismatch: {len(low_files)} low vs {len(high_files)} high"

        self.pairs = list(zip(low_files, high_files))

    def __len__(self) -> int:
        return len(self.pairs)

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL image to float tensor [0, 1]."""
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)   # (C, H, W)

    def _random_crop(self, low: torch.Tensor,
                     high: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crop both images at the same random location."""
        _, H, W = low.shape
        p = self.patch_size
        if H < p or W < p:
            # Pad if image is smaller than patch size
            pad_h = max(0, p - H)
            pad_w = max(0, p - W)
            low  = TF.pad(low,  [0, 0, pad_w, pad_h])
            high = TF.pad(high, [0, 0, pad_w, pad_h])
            _, H, W = low.shape

        i = random.randint(0, H - p)
        j = random.randint(0, W - p)
        return low[:, i:i+p, j:j+p], high[:, i:i+p, j:j+p]

    def _augment(self, low: torch.Tensor,
                 high: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply identical random augmentation to both images."""
        # Random horizontal flip
        if random.random() > 0.5:
            low  = TF.hflip(low)
            high = TF.hflip(high)
        # Random vertical flip
        if random.random() > 0.5:
            low  = TF.vflip(low)
            high = TF.vflip(high)
        # Random 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            low  = torch.rot90(low,  k, dims=[1, 2])
            high = torch.rot90(high, k, dims=[1, 2])
        return low, high

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        low_path, high_path = self.pairs[idx]

        low  = self._to_tensor(Image.open(low_path).convert('RGB'))
        high = self._to_tensor(Image.open(high_path).convert('RGB'))

        if self.patch_size is not None:
            low, high = self._random_crop(low, high)

        if self.augment:
            low, high = self._augment(low, high)

        return low, high


# ─────────────────────────────────────────────────────────────
# SYNTHETIC DATASET (for quick testing without real data)
# ─────────────────────────────────────────────────────────────
class SyntheticLOLDataset(Dataset):
    """
    Synthetic low-light dataset for quick testing.
    Creates paired (dark, bright) images without downloading real data.
    """

    def __init__(self, n_samples: int = 200, patch_size: int = 128):
        self.n_samples  = n_samples
        self.patch_size = patch_size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.patch_size

        # Generate a realistic-looking bright image
        torch.manual_seed(idx)
        high = torch.rand(3, p, p) * 0.6 + 0.2   # values in [0.2, 0.8]

        # Simulate low-light: reduce brightness + add noise
        gamma = random.uniform(2.0, 4.0)           # random darkness level
        low   = high ** gamma                       # gamma darkening
        noise = torch.randn_like(low) * 0.02        # Gaussian noise
        low   = torch.clamp(low + noise, 0.0, 1.0)

        return low, high


# ─────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────
def build_dataloaders(train_dir:   str,
                      test_dir:    str,
                      patch_size:  int = 128,
                      batch_size:  int = 4,
                      num_workers: int = 0):
    """
    Build train and validation dataloaders.

    Args:
        train_dir   : path to LOL/train
        test_dir    : path to LOL/test
        patch_size  : crop size for training
        batch_size  : training batch size
        num_workers : dataloader worker processes

    Returns:
        train_loader, val_loader
    """

    train_ds = LOLDataset(train_dir, patch_size=patch_size, augment=True)
    val_ds   = LOLDataset(test_dir,  patch_size=None,       augment=False)

    print(f"[LOLDataset] {train_dir} → {len(train_ds)} pairs "
          f"| patch={patch_size} | aug=True")
    print(f"[LOLDataset] {test_dir}  → {len(val_ds)} pairs "
          f"| patch={patch_size} | aug=False")

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = 1,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = False,
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ds = SyntheticLOLDataset(n_samples=10, patch_size=128)
    low, high = ds[0]
    print(f"Synthetic sample — low: {low.shape}, high: {high.shape}")
    print(f"  low  range: [{low.min():.3f}, {low.max():.3f}]")
    print(f"  high range: [{high.min():.3f}, {high.max():.3f}]")
