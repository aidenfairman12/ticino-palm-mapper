from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import scipy
from scipy import ndimage

import rasterio
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.transform import array_bounds, rowcol


from shapely.geometry import box
from collections import defaultdict
import geopandas as gpd

from .config import PROJECT_CRS



@dataclass
class ChannelStats:
    """Per-channel normalization statistics (e.g. mean/std or min/max).

    Shape of each field should be (C,) matching the number of channels in
    whichever stack this was computed over (4 for feature_stack, 6 for
    feature_stack_rs).
    """
    mean: np.ndarray
    std: np.ndarray


def compute_channel_stats(train_paths: list[Path]) -> ChannelStats:
    """Compute per-channel mean/std across a set of tiles, for normalization using Welford's algorithm
    so each tile only needs to be loaded once.
    """
    n_a = 0
    mean_a = None
    M2_a = None
    
    for path in train_paths:
        arr, _, _ = load_tile(path)
        if mean_a is None:
            mean_a = np.zeros(arr.shape[0])
            M2_a = np.zeros(arr.shape[0])
            
        n_b = arr.shape[1] * arr.shape[2]
        mean_b = arr.mean(axis=(1,2))
        M2_b = arr.var(axis=(1,2)) * n_b

        n_ab = n_a + n_b
        delta = mean_b - mean_a
        mean_a = mean_a + delta * (n_b / n_ab)
        M2_a = M2_a + M2_b + delta**2 * (n_a * n_b / n_ab)
        n_a = n_ab

    std = np.sqrt(M2_a / n_a)
    
    return ChannelStats(mean_a, std)

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
    """
    C = array.shape[0]
    mean = stats.mean.reshape((C,1,1))
    std = stats.std.reshape((C,1,1))
    
    
    return (array - mean) / (std + 1e-8)


def load_confirmed_points(geojson_path: Path) -> gpd.GeoDataFrame:
    """Load the hand-verified ground-truth points for evaluation.

    Returns a GeoDataFrame (geopandas) in EPSG:2056, or convert to whatever
    structure you want to consume downstream (e.g. a list of (x, y, signal)
    tuples). Relevant columns: geometry, ndvi_signal_current, source,
    batch_file. Remember: 13/39 are "none" — decide whether "none" points
    count as confirmed-negative evaluation examples or get excluded.
    """
    gdf = gpd.read_file(geojson_path)
    
    
    return gdf


def points_to_density_target(
    points, tile_transform, tile_shape: tuple[int, int], sigma_px: float
) -> np.ndarray:
    """Creates density map from points using gaussian filter.
    """
    H, W = tile_shape[0], tile_shape[1]
    left, bottom, right, top = array_bounds(H, W, tile_transform)
    mask = points.geometry.within(box(left,bottom,right, top))

    tile_points = points[mask]
    rows, cols = rowcol(tile_transform, tile_points.geometry.x, tile_points.geometry.y)
    
    arr = np.zeros((H,W))
    arr[rows,cols] = 1
    
    arr = scipy.ndimage.gaussian_filter(arr, sigma=sigma_px)
    
    return arr

def spatial_split(
    tile_paths: list[Path], val_frac: float, test_frac: float, block_size_m: float, seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split tiles into train/val/test by SPATIAL block, not randomly. Function computes a centroid
    for each tile, and puts them in a coarse grid. tiles who belong to the same cell in this coarse grid
    are not used in seperate splits
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

    shuffled_indices = rng.permutation(len(cells))
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

def get_cells(paths, block_size_m):
    cells = set()
    for p in paths:
        with rasterio.open(p) as src:
            coords = src.bounds
        centroid = box(*coords).centroid
        cell = (int(centroid.x // block_size_m), int(centroid.y // block_size_m))
        cells.add(cell)
    return cells
    
def random_crop(
    image: np.ndarray, target: np.ndarray, crop_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Crops the image and density map to the same size.
    """
    H, W = target.shape
    
    row_offset = rng.integers(0, H - crop_size, endpoint=True)
    col_offset = rng.integers(0, W - crop_size, endpoint=True)
    
    image_crop = image[:, row_offset: row_offset + crop_size, col_offset:col_offset + crop_size]
    target_crop = target[row_offset: row_offset + crop_size, col_offset: col_offset + crop_size]
    
    return image_crop, target_crop

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
    tile_dir = Path("data/processed/lugano_example/feature_stack")
    tile_paths = list(tile_dir.glob("*_rgbchm.tif"))
    stats = compute_channel_stats(tile_paths)
    #print(stats.mean, stats.std)
    
    some_path = tile_paths[0]
    #arr, _, _ = load_tile(some_path)    
    #normed = normalize(arr, stats)
    #print(normed.min(), normed.max(), normed.mean(axis=(1,2)))
    
    #geojson = "data/interim/labels/lugano_MASTER_confirmed_palms.geojson"
    #load_confirmed_points(geojson)
    
    points = load_confirmed_points(Path("data/interim/labels/lugano_MASTER_confirmed_palms.geojson"))

    tile_dir = Path("data/processed/lugano_example/feature_stack")
    for tile_path in tile_dir.glob("*_rgbchm.tif"):
        arr, transform, crs = load_tile(tile_path)
        tile_shape = arr.shape[1:]  # (H, W) — drop the channel dim
        target = points_to_density_target(points, transform, tile_shape, sigma_px=10)
        if target.sum() > 0:
            print(f"found points in: {tile_path.name}")
            print(f"target shape: {target.shape}, sum: {target.sum():.2f}, max: {target.max():.4f}")
            break
    else:
        print("no tile in this AOI contained a confirmed point")