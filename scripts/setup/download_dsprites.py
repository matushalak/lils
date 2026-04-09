#!/usr/bin/env python3
"""Download the canonical dSprites archive into the repo data directory."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np


DATASET_URL = (
    "https://github.com/deepmind/dsprites-dataset/raw/master/"
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "raw" / "dsprites" / "dsprite_train.npz",
        help="Target dataset path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing file.",
    )
    return parser.parse_args()


def validate_dataset(path: Path) -> None:
    with np.load(path, allow_pickle=True) as data:
        required = {"imgs", "latents_values", "latents_classes"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"dataset file is missing arrays: {sorted(missing)}")


def main() -> int:
    args = parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.force:
        validate_dataset(output)
        print(f"dSprites already present at {output}")
        return 0

    tmp_path = output.with_suffix(output.suffix + ".part")
    try:
        print(f"Downloading dSprites to {output}")
        with urllib.request.urlopen(DATASET_URL) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(output)
        validate_dataset(output)
        print("dSprites download complete")
        return 0
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        print(f"Failed to download dSprites: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
