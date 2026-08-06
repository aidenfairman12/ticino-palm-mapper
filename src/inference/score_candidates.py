"""
Score random locations across the full NIR footprint with a trained linear
probe, surfacing candidates for manual review — an active-learning pass to
find both new positives (palms the current label set missed) and hard
negatives (locations that fool the model despite not being palms).

Workflow:
  1. Train one "production" probe on ALL currently available labeled data
     (confirmed + scouted positives, sampled negatives) — no held-out fold,
     unlike leave_one_tile_out_cv, since this isn't an evaluation, it's the
     actual classifier we're about to use.
  2. Generate a systematic grid of candidate locations (--spacing-m apart)
     across the given --tile-dirs' combined footprint, rather than i.i.d.
     random draws. Palms are spatially sparse and clustered, so a modest
     number of random samples essentially never lands near an undiscovered
     one — coverage needs to be deliberate, not left to chance. Spacing
     defaults to half the crop width so every location is within centering
     distance of some grid point (a palm near the edge of a crop, rather
     than centered, gets diluted by surrounding context in the pooled
     feature — see score_locations). Excludes a --min-distance-m buffer
     around every already-reviewed point: the original confirmed/scouted
     positives, plus (if provided) prior active-learning rounds' confirmed
     positives and hard negatives — otherwise a repeat run keeps
     re-surfacing spots you've already manually judged. If the grid
     produces more points than --n-samples, a seeded shuffle picks which
     subset to score this run, so --seed still controls per-run scope.
  3. Score every location with the production probe, batched
     (--batch-size) rather than one at a time — necessary for a dense grid
     to finish in a reasonable time; a few hundred random samples didn't
     need this, hundreds of thousands of grid points do.
  4. Write out every scored location (sorted by predicted probability,
     highest first) to a GeoJSON for manual review — high-probability
     ones are the actual candidates worth looking at: either genuine
     unlabeled palms, or hard negatives that fooled the model.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from shapely.geometry import Point

from src.data.config import PROJECT_CRS
from src.data.dataset import load_confirmed_points, load_tile, normalize
from src.data.probe_dataset import (
    PalmProbeDataset,
    crop_centered_on_point,
    find_covering_tiles,
    load_tile_bounds,
    sample_negative_points,
)
from src.models.mae import MaskedAutoencoder, adapt_patch_embed
from src.training.linear_probe import extract_all_features, train_probe
from src.training.pretrain_ssl import glob_tiles


def generate_grid_points(
    tile_paths: list[Path],
    spacing_m: float,
    exclude_points: gpd.GeoSeries,
    min_distance_m: float,
) -> list[tuple[float, float]]:
    """Systematic grid of (x, y) points, spacing_m apart, covering the
    combined footprint of tile_paths — deliberate coverage instead of
    leaving it to i.i.d. random draws, which essentially never land near a
    spatially sparse, clustered feature like a palm across a large AOI.

    Tile bounding envelopes combined can include gaps where no tile
    actually exists (irregular AOI shape), so grid points are filtered
    down to only those actually covered by at least one tile — same
    reasoning as find_covering_tiles, done as a spatial join here since
    it's over far more points than that per-point helper is meant for.

    exclude_points/min_distance_m drop any grid point within min_distance_m
    of an already-reviewed location (confirmed, scouted, prior
    active-learning positives/hard-negatives), so repeat runs don't
    re-surface spots already manually judged.
    """
    tile_boxes = load_tile_bounds(tile_paths)
    boxes_gdf = gpd.GeoDataFrame(geometry=[b for _, b in tile_boxes], crs=PROJECT_CRS)
    minx, miny, maxx, maxy = boxes_gdf.total_bounds

    xs = np.arange(minx, maxx, spacing_m)
    ys = np.arange(miny, maxy, spacing_m)
    xx, yy = np.meshgrid(xs, ys)
    grid_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs=PROJECT_CRS)

    covered = gpd.sjoin(grid_gdf, boxes_gdf, predicate="within", how="inner")
    covered = covered[~covered.index.duplicated()]  # one row per point, even if it's within >1 tile

    if len(exclude_points) > 0:
        exclusion_zone = exclude_points.buffer(min_distance_m).union_all()
        covered = covered[~covered.geometry.within(exclusion_zone)]

    return [(pt.x, pt.y) for pt in covered.geometry]


def build_production_probe(
    model: MaskedAutoencoder,
    dataset: PalmProbeDataset,
    device: torch.device,
    embed_dim: int,
    epochs: int,
    lr: float,
    pos_weight_multiplier: float = 1.0,
    hidden_dim: int = 0,
) -> nn.Module:
    """Train one probe on ALL of `dataset` — no held-out fold. This is the
    real classifier used to score new candidates, as opposed to the CV
    probes in leave_one_tile_out_cv (which only exist to measure how well
    this approach generalizes, and are discarded after evaluation)."""
    features, labels, _ = extract_all_features(model, dataset, device)
    return train_probe(features, labels, embed_dim, epochs, lr, pos_weight_multiplier, hidden_dim)


def _write_results(results: list[dict], output_path: Path) -> None:
    """Write via a temp file + atomic rename, not directly to output_path.
    This is called repeatedly through a run that can be time-limit-killed at
    any moment — a plain to_file() could get interrupted mid-write and leave
    output_path holding truncated/corrupt GeoJSON, which would then break
    _load_prior_results() on the next resume attempt. os.replace (what
    Path.replace uses) is atomic on POSIX as long as both paths are on the
    same filesystem, which they are here.
    """
    gdf = gpd.GeoDataFrame(
        results,
        geometry=[Point(r["x"], r["y"]) for r in results],
        crs="EPSG:2056",
    )
    gdf = gdf.sort_values("predicted_prob", ascending=False)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    gdf.to_file(tmp_path, driver="GeoJSON")
    tmp_path.replace(output_path)


def _load_prior_results(output_path: Path) -> tuple[list[dict], set[tuple[float, float]]]:
    """Load a previous (possibly truncated, e.g. by a time-limit kill)
    --output file, if one exists, so a resumed run can skip locations
    already scored instead of starting the whole grid over. Coordinates are
    rounded before going into the skip-set since they round-trip through
    GeoJSON text serialization, not compared as exact floats.
    """
    if not output_path.exists():
        return [], set()

    prior_gdf = gpd.read_file(output_path)
    prior_results = prior_gdf.drop(columns="geometry").to_dict("records")
    already_scored = {(round(r["x"], 3), round(r["y"], 3)) for r in prior_results}
    print(f"resuming from {output_path}: {len(already_scored)} location(s) already scored")
    return prior_results, already_scored


def score_locations(
    model: MaskedAutoencoder,
    probe: nn.Linear,
    locations: list[tuple[float, float]],
    tile_paths: list[Path],
    stats,
    crop_size: int,
    device: torch.device,
    batch_size: int = 32,
    prior_results: list[dict] | None = None,
    output_path: Path | None = None,
    flush_every: int = 5000,
) -> list[dict]:
    """Score every (x, y) in `locations` with `probe`, one result per
    location. Unlike PalmProbeDataset (which deliberately fans a labeled
    point out across every covering NIR-date tile for training diversity),
    here a location covered by multiple dates is scored on only the most
    recent date's tile that actually has real coverage there — this is a
    review/candidate list, so a single real-world spot should appear as ONE
    card, not once per date it happens to have extra coverage for. (Tile
    filenames are identical across date directories — only the parent
    directory differs — so sorting covering tiles by parent dir name picks
    dates newest-first, since feature_stack_rs_<YYYYMMDD> sorts
    chronologically as a string.)

    Tile bounding boxes are rectangular, but the actual flight-strip/
    delivery coverage within them can have real nodata gaps — a point can
    fall inside a covering tile's bounds yet land on a patch of nothing.
    Reading that produces an all-zero crop, which is both an unreviewable
    black card AND a meaningless prediction (z-score normalizing all-zero
    input just yields a constant, not "the model looked and found
    nothing"). So each covering tile is tried newest-first, skipping ahead
    to an older date if the newest one turns out to be nodata at this exact
    point, and the location itself is skipped only if every covering tile
    is nodata there.

    Two performance changes vs. scoring one location at a time (fine for a
    few hundred random samples, impractical for a systematic grid that can
    run into the hundreds of thousands): tiles are cached in memory as
    they're loaded (a dense grid revisits the same handful of tiles
    repeatedly — modest tile sizes here make caching all of them cheap
    relative to re-reading from disk per point), and crops are batched
    through the model instead of one forward pass per location.

    Returns predicted_prob via sigmoid(logit), so results can be
    sorted/thresholded meaningfully rather than just a hard 0/1.

    If `output_path` is given, `prior_results` (already-scored locations from
    an earlier, possibly time-limit-truncated run against this same output
    file — see _load_prior_results) plus everything scored so far this run
    are flushed to `output_path` every `flush_every` locations. Unlike
    pretrain_ssl.py's epoch checkpoints, there's no separate resumable state
    to save — the output file itself IS the checkpoint, since `locations` is
    regenerated deterministically from the same CLI args on every run (see
    main()) and resuming just means skipping whatever's already in it. A job
    killed mid-run this way loses at most `flush_every` locations of
    progress, not the whole run.
    """
    model.eval()
    t_start = time.time()

    # Precompute once — reopening every tile's bounds per location (as the
    # old find_covering_tiles(point, tile_paths) signature required) would
    # be wasteful at this scale.
    tile_boxes = load_tile_bounds(tile_paths)

    # Bounded LRU, NOT a plain dict — each 6-band 1024x1024 float32 tile is
    # ~25MB, and a full systematic grid over this AOI touches most/all of the
    # ~11.8k tiles across all date dirs combined. An unbounded cache here is
    # what OOM-killed every attempt at the real full run (32GB job, ~290GB if
    # every tile got cached). 150 tiles is a ~3.75GB ceiling — a large margin
    # under 32GB even alongside the model/probe/batch tensors, while still
    # capturing the locality benefit: the grid is walked in raster-scan
    # order (see generate_grid_points — not shuffled unless --n-samples
    # subsamples it), so nearby grid points revisit the same handful of
    # tiles in clusters, not randomly across the whole AOI.
    max_cached_tiles = 150
    tile_cache: OrderedDict[Path, tuple[np.ndarray, object]] = OrderedDict()

    def _load_tile_cached(path: Path) -> tuple[np.ndarray, object]:
        if path in tile_cache:
            tile_cache.move_to_end(path)
            return tile_cache[path]
        arr, transform, _ = load_tile(path)
        tile_cache[path] = (arr, transform)
        if len(tile_cache) > max_cached_tiles:
            tile_cache.popitem(last=False)
        return tile_cache[path]

    # Single streaming pass: resolve each location to a tile + crop, buffer
    # up to batch_size of them, then run inference on that batch immediately
    # — NOT "resolve every location in `locations` into a `resolved` list,
    # then batch through it," which used to hold one full (6, 224, 224)
    # float32 crop (~1.2MB) per PENDING location simultaneously. At full
    # systematic-grid scale (~500k locations) that's ~600GB of crops sitting
    # in memory before a single inference batch ran — this, combined with
    # the unbounded tile cache below, is what OOM-killed every attempt at
    # the real full run before it ever printed a progress line. Bounding the
    # in-flight buffer to batch_size keeps peak memory for crops at
    # batch_size * ~1.2MB regardless of how large `locations` is.
    results = list(prior_results) if prior_results else []
    since_last_flush = 0
    n_skipped_nodata = 0
    n_skipped_no_tile = 0
    pending: list[tuple[float, float, Path, np.ndarray]] = []

    def _run_pending_batch() -> None:
        nonlocal since_last_flush
        if not pending:
            return
        imgs = torch.stack([torch.from_numpy(crop).float() for _, _, _, crop in pending]).to(device)
        with torch.no_grad():
            encoded = model.encode_full(imgs)
            features = encoded[:, 0]  # CLS token, (B, D)
            logits = probe(features).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().tolist()
        for (x, y, tile_path, _), prob in zip(pending, probs):
            results.append({"x": x, "y": y, "tile": tile_path.name, "predicted_prob": prob})
        since_last_flush += len(pending)
        pending.clear()

    for x, y in locations:
        point = Point(x, y)
        covering = find_covering_tiles(point, tile_boxes)
        if not covering:
            n_skipped_no_tile += 1
            continue

        tile_path, raw_crop = None, None
        for candidate_tile in sorted(covering, key=lambda p: p.parent.name, reverse=True):
            arr, transform = _load_tile_cached(candidate_tile)
            candidate_crop = crop_centered_on_point(arr, transform, point, crop_size)
            if (candidate_crop == 0).all(axis=0).mean() > 0.5:
                continue  # mostly nodata at this point in this date's tile — try an older one
            tile_path, raw_crop = candidate_tile, candidate_crop
            break

        if tile_path is None:
            n_skipped_nodata += 1
            continue  # every covering tile is nodata at this point — unscoreable

        pending.append((x, y, tile_path, normalize(raw_crop, stats)))

        if len(pending) >= batch_size:
            _run_pending_batch()
            if output_path is not None and since_last_flush >= flush_every:
                _write_results(results, output_path)
                elapsed = time.time() - t_start
                rate = (len(results) - len(prior_results or [])) / elapsed if elapsed > 0 else 0.0
                print(
                    f"flushed {len(results)} total scored ({len(results) - len(prior_results or [])} this run, "
                    f"{elapsed/60:.1f} min elapsed, {rate:.1f} locations/s) -> {output_path}"
                )
                since_last_flush = 0

    _run_pending_batch()  # leftover partial batch smaller than batch_size

    if n_skipped_nodata:
        print(f"[warn] skipped {n_skipped_nodata} location(s) — nodata in every covering tile")
    if n_skipped_no_tile:
        print(f"[warn] skipped {n_skipped_no_tile} location(s) — no covering tile")

    if output_path is not None and since_last_flush > 0:
        _write_results(results, output_path)  # final partial flush so the last <flush_every don't get lost

    return results


def parse_score_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score random locations with a trained probe, for active-learning-style manual review."
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tile-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--confirmed-points", type=Path, required=True)
    p.add_argument("--scouted-points", type=Path, default=None)
    p.add_argument(
        "--hard-negatives", type=Path, default=None,
        help="Optional path to active_learning_hard_negatives.geojson (from "
             "12_merge_review_verdicts.py) — added ON TOP of the --n-negatives random "
             "negatives when training the production probe (same reasoning as "
             "linear_probe.py's --hard-negatives), AND excluded from the random "
             "locations sampled for scoring, so a repeat run doesn't re-surface "
             "spots already confirmed as not-palm for review.",
    )
    p.add_argument(
        "--reviewed-positives", type=Path, default=None,
        help="Optional path to active_learning_confirmed_palms.geojson (from "
             "12_merge_review_verdicts.py) — excluded from the random locations "
             "sampled for scoring, so a repeat run doesn't re-surface spots "
             "already confirmed as palms for review.",
    )
    p.add_argument(
        "--n-samples", type=int, default=500,
        help="Max number of grid locations to score this run. The systematic grid "
             "(see --spacing-m) is usually far larger than this — if so, a seeded "
             "shuffle picks which subset to score, so --seed still controls scope "
             "per run without needing multiple separate grids on disk.",
    )
    p.add_argument("--n-negatives", type=int, default=30, help="Negatives for training the production probe.")
    p.add_argument("--min-distance-m", type=float, default=20.0)
    p.add_argument(
        "--spacing-m", type=float, default=11.0,
        help="Grid spacing for the systematic candidate-location scan. Default is "
             "half the crop width (crop covers ~22m at ~0.1m/px), so every location "
             "is within centering distance of some grid point — a palm near the "
             "edge of a crop rather than centered gets diluted by surrounding "
             "context in the pooled feature, so under-spacing costs real recall.",
    )
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for scoring inference.")
    p.add_argument(
        "--pos-weight-multiplier", type=float, default=1.0,
        help="Scales pos_weight relative to the full n_neg/n_pos ratio — see "
             "linear_probe.py's flag of the same name for the reasoning/tradeoff.",
    )
    p.add_argument(
        "--hidden-dim", type=int, default=0,
        help="0 (default) trains a plain linear probe. >0 trains a small MLP head "
             "instead — see linear_probe.py's flag of the same name for the reasoning.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("candidate_scores.geojson"))
    return p.parse_args()


def main() -> None:
    # SLURM output is redirected to a file, so Python's default is fully
    # buffered, not line-buffered — every print() below (including inside
    # score_locations's periodic flushes) could sit unwritten until process
    # exit. That's exactly what made the OOM kills that motivated this
    # function hard to diagnose: three ~1h runs, zero progress lines in any
    # of their logs, even though a lot had almost certainly already
    # happened by the time each was killed.
    sys.stdout.reconfigure(line_buffering=True)

    args = parse_score_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Phase timing — printed at the end, so a small trial run's numbers can be used
    # to extrapolate a full/near-full grid pass's runtime. Wall-clock alone conflates
    # fixed costs (checkpoint load, production-probe training) with the part that
    # actually scales with --n-samples (scoring), which would make a small trial's
    # extrapolation pessimistic — so each phase is timed separately.
    t_load_start = time.time()

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stats = checkpoint["stats"]
    in_chans = len(stats.mean)

    # backbone_name/embed_dim are read from the checkpoint (saved by pretrain_ssl.py's
    # save_checkpoint) rather than hardcoded, since --model-size now lets a run use
    # vit_small or vit_base. Fallback covers checkpoints saved before this field
    # existed (all were vit_small).
    backbone_name = checkpoint.get("backbone_name", "vit_small_patch14_dinov2.lvd142m")
    img_size = 224
    patch_size = 14
    embed_dim = checkpoint.get("embed_dim", 384)
    decoder_embed_dim = 192
    decoder_depth = 4
    decoder_num_heads = 6
    mask_ratio = 0.75

    backbone = timm.create_model(backbone_name, pretrained=True)
    backbone = adapt_patch_embed(backbone, in_chans)
    model = MaskedAutoencoder(
        backbone, img_size, patch_size, in_chans, embed_dim,
        decoder_embed_dim, decoder_depth, decoder_num_heads, mask_ratio,
    )
    model.backbone.patch_embed.strict_img_size = False
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    points = load_confirmed_points(args.confirmed_points)
    distinct = points[points["ndvi_signal_current"] == "distinct"]
    none_signal = points[points["ndvi_signal_current"] == "none"]

    positive_geoms = distinct.geometry
    if args.scouted_points is not None:
        scouted = gpd.read_file(args.scouted_points)
        positive_geoms = pd.concat([positive_geoms, scouted.geometry], ignore_index=True)

    # Active-learning positives confirmed in a PRIOR round feed back in as training
    # positives too (same reasoning as --scouted-points) — without this, they'd only
    # ever be used to exclude already-reviewed spots below, never to actually teach
    # the production probe about them.
    if args.reviewed_positives is not None:
        reviewed_pos = gpd.read_file(args.reviewed_positives)
        positive_geoms = pd.concat([positive_geoms, reviewed_pos.geometry], ignore_index=True)

    tile_paths = glob_tiles(args.tile_dirs, in_chans)

    negatives = sample_negative_points(
        positive_points=positive_geoms,
        candidate_points=none_signal.geometry,
        tile_paths=tile_paths,
        n_negatives=args.n_negatives,
        min_distance_m=args.min_distance_m,
        seed=args.seed,
    )

    # Empty by default so the random_locations exclusion below always has something
    # to concatenate against, whether or not --hard-negatives was passed.
    hard_neg_geoms = gpd.GeoSeries([], crs="EPSG:2056")
    if args.hard_negatives is not None:
        hard_neg = gpd.read_file(args.hard_negatives)
        hard_neg_geoms = hard_neg.geometry
        negatives = negatives + [(pt.x, pt.y) for pt in hard_neg_geoms]
        print(f"negatives: {len(negatives) - len(hard_neg)} random + {len(hard_neg)} hard = {len(negatives)} total")

    t_load_end = time.time()

    dataset = PalmProbeDataset(positive_geoms, negatives, tile_paths, stats, args.crop_size)
    print(f"training production probe on {len(dataset)} examples "
          f"({sum(1 for e in dataset.examples if e[3] == 1.0)} positive, "
          f"{sum(1 for e in dataset.examples if e[3] == 0.0)} negative)")

    probe = build_production_probe(
        model, dataset, device, embed_dim, args.epochs, args.lr, args.pos_weight_multiplier, args.hidden_dim
    )

    t_probe_end = time.time()

    # Systematic grid instead of random draws — see module docstring for why.
    # Excludes a --min-distance-m buffer around every already-reviewed point
    # (positive_geoms already includes confirmed + scouted + reviewed-positives;
    # none_signal + hard_neg_geoms covers GBIF-none and prior hard negatives).
    already_reviewed = pd.concat([positive_geoms, none_signal.geometry, hard_neg_geoms], ignore_index=True)
    grid_locations = generate_grid_points(tile_paths, args.spacing_m, already_reviewed, args.min_distance_m)
    print(f"grid: {len(grid_locations)} candidate locations at {args.spacing_m}m spacing")

    if len(grid_locations) > args.n_samples:
        rng = np.random.default_rng(args.seed + 1)  # different seed than the training negatives
        chosen_idx = rng.choice(len(grid_locations), size=args.n_samples, replace=False)
        scored_locations = [grid_locations[i] for i in chosen_idx]
    else:
        scored_locations = grid_locations
    print(f"scoring {len(scored_locations)} of them this run...")

    t_grid_end = time.time()

    # Resume support: score_locations/main() are re-run wholesale on a chained
    # follow-up job rather than restarting a saved model/optimizer like
    # pretrain_ssl.py — but scored_locations is deterministic given the same
    # CLI args (grid + seeded subsample), so a resumed run reproduces the
    # exact same target list and just needs to skip whatever's already in
    # args.output from a prior, possibly time-limit-truncated attempt.
    prior_results, already_scored = _load_prior_results(args.output)
    locations_to_score = [
        loc for loc in scored_locations
        if (round(loc[0], 3), round(loc[1], 3)) not in already_scored
    ]
    n_already_done = len(scored_locations) - len(locations_to_score)
    if n_already_done:
        print(f"{n_already_done} of {len(scored_locations)} target location(s) already scored, "
              f"{len(locations_to_score)} remaining this run")

    results = score_locations(
        model, probe, locations_to_score, tile_paths, stats, args.crop_size, device, args.batch_size,
        prior_results=prior_results, output_path=args.output,
    )

    t_score_end = time.time()
    ms_per_location = (
        (t_score_end - t_grid_end) / len(locations_to_score) * 1000 if locations_to_score else 0.0
    )
    print(
        f"timing — load: {t_load_end - t_load_start:.1f}s, probe train: {t_probe_end - t_load_end:.1f}s, "
        f"grid gen: {t_grid_end - t_probe_end:.1f}s, scoring: {t_score_end - t_grid_end:.1f}s "
        f"({ms_per_location:.2f} ms/location, fixed overhead excluded, {len(locations_to_score)} newly scored)"
    )

    # score_locations already flushed results (prior + new) to args.output as it
    # went — nothing left to write here.
    n_flagged = sum(1 for r in results if r["predicted_prob"] > 0.5)
    print(f"wrote {len(results)} scored locations -> {args.output}")
    print(f"{n_flagged} flagged as predicted-positive (prob > 0.5) — review these first")


if __name__ == "__main__":
    main()
