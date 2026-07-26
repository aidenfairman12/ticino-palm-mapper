"""
PyTorch Dataset over the Phase 0 feature stacks.

STUBBED ON PURPOSE — this file is yours to implement. Every function below has
a docstring describing what it needs to do, its expected inputs/outputs, and
relevant context from Phase 0. No tensor logic is filled in.

Inputs available on disk (see data/processed/lugano_example/):
  - feature_stack/*_rgbchm.tif      4 bands float32 [R, G, B, CHM],   1024x1024
  - feature_stack_rs/*_nirchm.tif   6 bands float32 [NIR,R,G,B,NDVI,CHM], 1024x1024
    (NIR/R/G/B here are RS-native, NOT the same acquisition as feature_stack's
    R/G/B — see 07_build_nir_stack.py's docstring before mixing the two)
  - <aoi>/*_mask.npy                weak positive masks, rasterized from GBIF
    points (02_align_labels.py) — presence-only, NOT pixel-accurate. Fine for
    pretraining/coarse supervision, NOT for evaluation.
  - data/interim/labels/lugano_MASTER_confirmed_palms.geojson
    39 hand-verified points (21 distinct/5 weak/13 none NDVI signal), the
    only real evaluation-grade ground truth. Small — treat as a sanity-check
    set, not a statistically powered test set.

Design decisions already made this session, worth keeping in mind:
  - Task formulation: density/counting favoured over precise instance
    detection (crowns run 20-30px at 10cm; adjacent palms merge; positional
    precision on hand-verified points is ~1-4m, comparable to crown size).
  - Splits MUST be spatially blocked, not randomly shuffled — nearby tiles
    are correlated (same lighting/imagery/palm clusters), a random split
    leaks information and inflates validation metrics.
  - Channel scales differ wildly (R/G/B in ~0-255, CHM in metres ~0-40,
    NDVI in [-1,1]) — normalize per-channel, not globally.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import rasterio
from rasterio import Affine
from rasterio.crs import CRS

from shapely.geometry import box
from collections import defaultdict
import geopandas as gpd

from config import PROJECT_CRS



@dataclass
class ChannelStats:
    """Per-channel normalization statistics (e.g. mean/std or min/max).

    Shape of each field should be (C,) matching the number of channels in
    whichever stack this was computed over (4 for feature_stack, 6 for
    feature_stack_rs).
    """
    mean: np.ndarray
    std: np.ndarray


def compute_channel_stats(tile_paths: list[Path]) -> ChannelStats:
    """Compute per-channel mean/std across a set of tiles, for normalization.

    Should stream through `tile_paths` (don't load them all into memory at
    once — these are 1024x1024xC float32, several MB each) and accumulate
    per-channel statistics.

    Decide: compute over the TRAINING split only (standard practice — stats
    from val/test tiles shouldn't leak into normalization), or over everything?

    Returns
    -------
    ChannelStats with .mean and .std arrays of shape (C,).
    """
    raise NotImplementedError


def load_tile(path: Path) -> tuple[np.ndarray, Affine, CRS] :
    """Read one feature-stack GeoTIFF into a (C, H, W) float32 array, also loads tile's 
    affine transformation, as well as CRS.
    """
    with rasterio.open(path) as src: 
        arr = src.read()
        transform = src.transform
        crs = src.crs
    
    assert arr.dtype == np.float32, f"expected float32, got {arr.dtype} in {path}"
    assert arr.shape[0] in (4,6), f"expected 4 or 6 channels but got {arr.shape[0]} from {path}"
    assert crs.to_epsg() == int(PROJECT_CRS.split(":")[1]), f"expected EPSG:2056, got {crs} in {path}"
    
    return arr, transform, crs


def normalize(array: np.ndarray, stats: ChannelStats) -> np.ndarray:
    """Apply per-channel normalization to a (C, H, W) array.

    array[c] = (array[c] - stats.mean[c]) / stats.std[c], broadcasting
    correctly over the (H, W) spatial dims. Watch for divide-by-zero if any
    channel has zero variance in a degenerate case.
    """
    raise NotImplementedError


def load_confirmed_points(geojson_path: Path):
    """Load the hand-verified ground-truth points for evaluation.

    Returns a GeoDataFrame (geopandas) in EPSG:2056, or convert to whatever
    structure you want to consume downstream (e.g. a list of (x, y, signal)
    tuples). Relevant columns: geometry, ndvi_signal_current, source,
    batch_file. Remember: 13/39 are "none" — decide whether "none" points
    count as confirmed-negative evaluation examples or get excluded.
    """
    raise NotImplementedError


def points_to_density_target(
    points, tile_transform, tile_shape: tuple[int, int], sigma_px: float
) -> np.ndarray:
    """Rasterize point locations into a density map target for one tile.

    Typical approach: place a delta/1 at each point's pixel location (via
    the tile's affine transform, rowcol()), then Gaussian-blur (scipy or
    skimage) with sigma_px so the target is a smooth density map whose
    integral over the tile approximates the palm count. This is the
    standard crowd-counting-style target — matches the density/counting
    task formulation.

    Only rasterize points that actually fall within this tile's bounds.

    Returns
    -------
    (H, W) float32 array.
    """
    raise NotImplementedError


def spatial_split(
    tile_paths: list[Path], val_frac: float, test_frac: float, block_size_m: float, seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split tiles into train/val/test by SPATIAL block, not randomly.

    Why: tiles that are geographically close are correlated (same lighting
    conditions, same imagery date, may share/straddle the same palm cluster).
    A random shuffle leaks that correlation across the split and inflates
    validation/test metrics — this was flagged explicitly for Phase 4's
    spatial modelling in the project proposal, and applies here too.

    One approach: bin tile centroids into `block_size_m` grid cells, assign
    whole cells (not individual tiles) to train/val/test so no two tiles from
    the same cell end up in different splits.

    Returns
    -------
    (train_paths, val_paths, test_paths)
    """
    coordMap = {}
    
    for x in tile_paths:
        with rasterio.open(x) as src:
            coords = src.bounds

        centroid = box(*coords).centroid
        coordMap[x] = centroid

    paths = list(coordMap.keys())
    centroids = list(coordMap.values())
    
    gdf = gpd.GeoDataFrame({"path": paths, "geometry": centroids}, crs=PROJECT_CRS)
    
    gdf["cell_x"] = (gdf.geometry.x // block_size_m).astype(int)
    gdf["cell_y"] = (gdf.geometry.y // block_size_m).astype(int)
    
    cellMap = gdf.groupby(["cell_x", "cell_y"])["path"].apply(list).to_dict()
    
    cells = list(cellMap.keys())
    rng = np.random.default_rng(seed)

    shuffled_indices = rng.permutation(cells)
    shuffled_cells = [cells[i] for i in shuffled_indices]
    
    n = len(shuffled_cells)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    
    val_cells = shuffled_cells[:n_val]
    test_cells = shuffled_cells[n_val:n_val + n_test]
    train_cells = shuffled_cells[n_val + n_test:]
    
    train_paths = [p for cell in train_cells for p in cellMap[cell]]
    val_paths = [p for cell in val_cells for p in cellMap[cell]]
    test_paths = [p for cell in test_cells for p in cellMap[cell]]
    
    return train_paths, val_paths, test_paths
    
def random_crop(
    image: np.ndarray, target: np.ndarray, crop_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly crop a (crop_size, crop_size) window from a tile + its target.

    image is (C, H, W), target is (H, W) — crop both at the SAME offset so
    they stay aligned. Useful for training-time augmentation; at 1024x1024
    per tile you'll likely want a smaller crop per training step (decide the
    size based on your model's expected input and available compute).
    """
    raise NotImplementedError


class PalmTileDataset(Dataset):
    """Dataset over feature-stack tiles + density targets.

    Decide up front (and document your choice here once made):
      - which stack to use: feature_stack (4ch, canton-wide RGB+CHM) or
        feature_stack_rs (6ch, NIR-covered subset only) or a merge of both.
      - which labels to train against: the weak GBIF masks (plentiful, noisy)
        vs. the 39-point confirmed set (accurate, tiny) — probably weak masks
        for training, confirmed set held out entirely for evaluation.
    """

    def __init__(
        self,
        tile_paths: list[Path],
        stats: ChannelStats,
        crop_size: int | None = None,
        augment: bool = False,
    ) -> None:
        """Store whatever the dataset needs: the tile list, normalization
        stats, and any config for cropping/augmentation. Don't load tile
        data here — do that lazily in __getitem__ so multiple DataLoader
        workers don't duplicate everything in memory upfront.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load tile `idx`, normalize, build its target, optionally crop/augment,
        and return (image_tensor, target_tensor) as torch.float32 tensors.

        image_tensor shape: (C, H, W). target_tensor shape: (H, W) for a
        density map, or whatever shape your task formulation needs.
        """
        raise NotImplementedError


if __name__ == "__main__":
    # Sanity-check scaffold — fill in once the pieces above work.
    # Suggested first check (see project notes): load ONE tile through the
    # full pipeline and print shape/dtype/min/max, e.g.:
    #
    #   ds = PalmTileDataset(tile_paths=[...], stats=...)
    #   x, y = ds[0]
    #   print(x.shape, x.dtype, x.min().item(), x.max().item())
    #   print(y.shape, y.dtype, y.sum().item())  # sum ~ implied palm count
    #
    # Get this boring plumbing verified correct before touching a model.
    spatial_split(["data/processed/lugano_example/feature_stack/lugano_example_2024_c000_r000_rgbchm.tif"], .15, .15, .2, 12)
