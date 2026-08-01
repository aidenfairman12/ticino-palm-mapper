"""
Point-centered crop dataset for the downstream linear-probe evaluation.

STUBBED ON PURPOSE — this file is yours to implement. Every function/method
below has a docstring describing what it needs to do. No logic is filled in.

This is a different cropping strategy from PalmSSLDataset (which takes a
random crop anywhere in a tile, for unsupervised pretraining). Here, every
crop must be centered on a specific labeled point — positive (confirmed
palm) or negative (confirmed/assumed non-palm) — so the linear probe has a
meaningful (crop, label) pair to train and evaluate on.

Design decisions made this session, worth keeping in mind:
  - A single point may fall inside more than one tile if multiple NIR-date
    stacks cover that location (e.g. both feature_stack_rs_20210811 and
    feature_stack_rs_20240821). Rather than picking just one, EACH covering
    tile becomes its own separate (crop, label) example — free extra
    training/eval diversity for a label-starved dataset, not a conflict to
    resolve down to one choice.
  - Positives: the "distinct"-signal points from the MASTER confirmed file
    (load_confirmed_points in src/data/dataset.py), plus your scouted
    candidates once you've decided how much to trust each confidence tier.
  - Negatives: locations with NO confirmed/candidate point nearby — see
    sample_negative_points below. Must not accidentally sample from an
    unreviewed GBIF occurrence point's location (could be a true positive
    nobody's checked) — sample from genuinely empty space instead.
  - Evaluation later uses leave-one-tile-out CV given how few positive
    TILES exist (not just points) — this dataset just needs to expose
    enough structure (which tile each crop came from) for that CV loop to
    group examples by tile.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import torch, rasterio, shapely
from rasterio.transform import rowcol
from torch.utils.data import Dataset
from shapely.geometry import box, Point

from .dataset import ChannelStats, load_tile, normalize


def load_tile_bounds(tile_paths: list[Path]) -> list[tuple[Path, "shapely.geometry.base.BaseGeometry"]]:
    """Open every tile in `tile_paths` exactly once and return (path, bounds-box)
    pairs, for repeated in-memory point-in-tile lookups via find_covering_tiles.
    """
    ret = []
    for path in tile_paths:
      with rasterio.open(path) as src:
        coords = src.bounds
      ret.append((path, box(*coords)))
    return ret


def find_covering_tiles(point, tile_boxes: list[tuple[Path, "shapely.geometry.base.BaseGeometry"]]) -> list[Path]:
    """Return every tile whose bounds (from `tile_boxes`, see load_tile_bounds)
    contain `point`.
    """
    return [path for path, tile_box in tile_boxes if point.within(tile_box)]
        
        


def crop_centered_on_point(
    arr: np.ndarray, transform, point, crop_size: int
) -> np.ndarray:
    """Extract a (C, crop_size, crop_size) window from `arr`, centered on
    `point` as closely as possible, if not clamp the point.
    """
    H,W = arr.shape[1], arr.shape[2]
    rows_px, cols_px = rowcol(transform, point.x, point.y)
    row_offset_ideal = rows_px - crop_size // 2
    col_offset_ideal = cols_px - crop_size // 2
    
    rows, cols = max(0, min(row_offset_ideal, H - crop_size)), max(0, min(col_offset_ideal, W - crop_size))
    arr = arr[:, rows:rows+crop_size, cols:cols+crop_size]
    
    return arr
    
    


def sample_negative_points(
    positive_points: gpd.GeoSeries,
    candidate_points: gpd.GeoSeries,
    tile_paths: list[Path],
    n_negatives: int,
    min_distance_m: float,
    seed: int,
) -> list[tuple[float, float]]:
    """Sample negative points, ensuring they are far enough away from positive and candidate points.
    return a list of tuples (x,y)
    """
    
    results = []
    rng = np.random.default_rng(seed)
    
    while len(results) < n_negatives:
      idx = rng.integers(0, len(tile_paths))
      tile_path = tile_paths[idx]
      with rasterio.open(tile_path) as src:
        coords = src.bounds
      
      x, y = rng.uniform(coords.left, coords.right), rng.uniform(coords.bottom, coords.top)
      point = shapely.geometry.Point(x, y)
      
      if positive_points.distance(point).min() >= min_distance_m and candidate_points.distance(point).min() >= min_distance_m:
        results.append((x,y))
    
    return results


class PalmProbeDataset(Dataset):
    """(crop, label, tile_path) triples for linear-probe training/eval.

    Unlike PalmSSLDataset, __init__ does the expensive work of building the
    full example list up front (not lazily in __getitem__) — every (point,
    covering tile) pair is enumerated once here, since that requires
    checking tile bounds against every point, not something to repeat on
    every __getitem__ call. Loading actual pixel data still happens lazily
    in __getitem__, same reasoning as PalmSSLDataset (avoid duplicating
    tile data across DataLoader workers).
    """

    def __init__(
        self,
        positive_points: gpd.GeoSeries,
        negative_points: list[tuple[float, float]],
        tile_paths: list[Path],
        stats: ChannelStats,
        crop_size: int,
    ) -> None:
        """Build self.examples: a list of (tile_path, x, y, label) tuples,
        label=1 for positives, label=0 for negatives. For each point in
        positive_points, call find_covering_tiles and add one example per
        covering tile (per the multi-date design decision above). Do the
        same for negative_points (each negative location likely covered by
        just one tile, but check — no reason to assume otherwise). Store
        stats and crop_size for use in __getitem__.
        """
        self.stats = stats
        self.crop_size = crop_size
        self.examples = []

        tile_boxes = load_tile_bounds(tile_paths)

        for point in positive_points:
          cov_tiles = find_covering_tiles(point, tile_boxes)

          for tile in cov_tiles:
            self.examples.append((tile, point.x, point.y, 1.0))

        for point in negative_points:
          x, y = point
          coord = shapely.geometry.Point(x, y)

          cov_tiles = find_covering_tiles(coord, tile_boxes)

          for tile in cov_tiles:
            self.examples.append((tile, coord.x, coord.y, 0.0))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load the tile for self.examples[idx], normalize, crop centered
        on the example's point (crop_centered_on_point), and return
        (image_tensor, label_tensor) — image_tensor (C, crop_size,
        crop_size) float32, label_tensor a scalar float32 (0. or 1.) ready
        for BCEWithLogitsLoss.
        """
        tile_path, x, y, label = self.examples[idx]

        arr, transform, _ = load_tile(tile_path)
        arr = normalize(arr, self.stats)
        
        point = shapely.geometry.Point(x, y)
        crop = crop_centered_on_point(arr, transform, point, self.crop_size)
        image_tensor = torch.from_numpy(crop).float()
        
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return (image_tensor, label_tensor)
        
        
if __name__ == "__main__":
    # Sanity-check scaffold — fill in once the pieces above work. Suggested
    # first check: build the dataset over a small tile subset + your
    # confirmed points, print len(dataset), and pull one positive and one
    # negative example to confirm shapes/labels look right, e.g.:
    #
    #   x_pos, y_pos = next((x, y) for x, y in dataset if y.item() == 1.0)
    #   print(x_pos.shape, y_pos.item())
    pass
