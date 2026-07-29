from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


from .dataset import ChannelStats, compute_channel_stats, load_tile, normalize, spatial_split





class PalmSSLDataset(Dataset):
    """Dataset for MAE-style SSL. Returns image only tensors with no target"""
    
    def __init__(self,
                 tile_paths: list[Path],
                 stats: ChannelStats,
                 crop_size: int
                 ) -> None:
        self.tile_paths = tile_paths
        self.stats = stats
        self.crop_size = crop_size
        self.rng = None
        
    def __len__(self):
        return len(self.tile_paths)
    
    def __getitem__(self, idx):
        
        path = self.tile_paths[idx]
        arr, _, _ = load_tile(path)
        arr = normalize(arr, self.stats)
        
        if self.rng is None:
            worker_info = get_worker_info()
            seed = worker_info.id if worker_info is not None else 0
            self.rng = np.random.default_rng(seed)
        
        H, W = arr.shape[1], arr.shape[2]
        row_offset = self.rng.integers(0, H - self.crop_size, endpoint=True)
        col_offset = self.rng.integers(0, W - self.crop_size, endpoint=True)
        crop = arr[:, row_offset:row_offset + self.crop_size, col_offset:col_offset + self.crop_size]
        
        return torch.from_numpy(crop).float()
    
    
if __name__ == "__main__":
    tile_dir = Path("data/processed/lugano_example/feature_stack")
    tile_paths = list(tile_dir.glob("*_rgbchm.tif"))

    train, val, test = spatial_split(tile_paths, val_frac=0.15, test_frac=0.15, block_size_m=150, seed=42)
    stats = compute_channel_stats(train)

    dataset = PalmSSLDataset(train, stats, crop_size=224)
    print(f"dataset length: {len(dataset)}")

    x = dataset[0]
    print(f"shape={x.shape}, dtype={x.dtype}, min={x.min():.2f}, max={x.max():.2f}")
    
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, num_workers=2, shuffle=True)
    batch = next(iter(loader))
    print(f"batch shape: {batch.shape}, dtype: {batch.dtype}")
    
    for x in batch:
        print(x.mean())