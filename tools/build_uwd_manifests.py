#!/usr/bin/env python3
"""Audit UW-D and create deterministic scene-disjoint JSONL manifests."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


WATER_TYPES = {
    "Type-1",
    "Type-3",
    "Type-5",
    "Type-7",
    "Type-9",
    "Type-IB",
    "Type-II",
    "Type-III",
}


def discover_samples(root: Path) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    seen_depths = set()

    for rgb_path in sorted((root / "uw").rglob("uw_*.png")):
        if rgb_path.name.startswith("._"):
            continue

        water_type = rgb_path.parent.name
        scene_id = rgb_path.parents[1].name
        frame_id = rgb_path.stem.removeprefix("uw_")
        if water_type not in WATER_TYPES:
            raise ValueError(f"unexpected water type in {rgb_path}")

        depth_path = root / "depth" / scene_id / f"depth_{frame_id}.hdf5"
        if not depth_path.is_file():
            raise FileNotFoundError(f"missing depth for {rgb_path}: {depth_path}")
        if depth_path in seen_depths:
            raise ValueError(f"depth is paired more than once: {depth_path}")
        seen_depths.add(depth_path)

        samples.append(
            {
                "rgb": rgb_path.relative_to(root).as_posix(),
                "depth": depth_path.relative_to(root).as_posix(),
                "scene_id": scene_id,
                "frame_id": frame_id,
                "water_type": water_type,
            }
        )

    depth_files = set((root / "depth").rglob("depth_*.hdf5"))
    unpaired = depth_files.difference(seen_depths)
    if unpaired:
        example = next(iter(unpaired))
        raise ValueError(f"found {len(unpaired)} unpaired depth files, e.g. {example}")
    return samples


def split_scenes(
    scene_ids: Iterable[str], train_fraction: float, val_fraction: float, seed: int
) -> Dict[str, set[str]]:
    scenes = sorted(set(scene_ids))
    if not scenes:
        raise ValueError("no scenes found")
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("fractions must satisfy train > 0, val >= 0, train + val < 1")

    random.Random(seed).shuffle(scenes)
    train_end = int(len(scenes) * train_fraction)
    val_end = train_end + int(len(scenes) * val_fraction)
    return {
        "train": set(scenes[:train_end]),
        "val": set(scenes[train_end:val_end]),
        "test": set(scenes[val_end:]),
    }


def write_manifests(
    samples: List[Dict[str, object]], splits: Dict[str, set[str]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, scene_ids in splits.items():
        split_samples = [s for s in samples if s["scene_id"] in scene_ids]
        with (output_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as f:
            for sample in split_samples:
                f.write(json.dumps(sample, sort_keys=True) + "\n")
        with (output_dir / f"{split_name}_scenes.txt").open(
            "w", encoding="utf-8"
        ) as f:
            for scene_id in sorted(scene_ids):
                f.write(scene_id + "\n")

        counts = Counter(str(s["water_type"]) for s in split_samples)
        print(
            f"{split_name}: {len(scene_ids)} scenes, {len(split_samples)} samples, "
            f"water_types={dict(sorted(counts.items()))}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("splits/uwd"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = discover_samples(args.root)
    splits = split_scenes(
        (str(sample["scene_id"]) for sample in samples),
        args.train_fraction,
        args.val_fraction,
        args.seed,
    )
    print(f"audited {len(samples)} one-to-one RGB/depth pairs")
    write_manifests(samples, splits, args.output_dir)


if __name__ == "__main__":
    main()
