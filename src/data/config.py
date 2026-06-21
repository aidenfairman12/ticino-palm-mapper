"""Small shared helpers for the Phase 0 pipeline scripts."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args(description: str) -> argparse.Namespace:
    """Standard CLI: every Phase 0 script takes --config."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--config",
        required=True,
        help="Path to AOI/run YAML config (e.g. configs/aoi_example.yaml)",
    )
    return p.parse_args()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# The single source of truth for the project CRS.
PROJECT_CRS = "EPSG:2056"  # CH1903+ / LV95
