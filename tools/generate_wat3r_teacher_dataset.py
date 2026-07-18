#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


PATH_COLUMNS = ("image", "teacher_depth", "teacher_confidence", "static_mask", "camera")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Wat3R teacher windows for every FLSea scene in a training split."
    )
    parser.add_argument("--train-list", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--wat3r-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-size", default=518, type=int)
    parser.add_argument("--window-size", default=24, type=int)
    parser.add_argument("--overlap", default=12, type=int)
    parser.add_argument("--frames-chunk-size", default=4, type=int)
    parser.add_argument("--min-visible-views", default=3, type=int)
    parser.add_argument("--teacher-weights", action="store_true")
    parser.add_argument("--save-visualization", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalized(path):
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def scene_key(image_path, dataset_root):
    image_path = image_path.expanduser().resolve()
    if image_path.parent.name != "imgs":
        raise ValueError(f"Expected image inside an imgs directory: {image_path}")
    scene_dir = image_path.parent.parent
    try:
        relative = scene_dir.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"Image is outside --dataset-root: {image_path}") from error
    return relative


def merge_manifests(manifests, output_path):
    rows = []
    fieldnames = None
    for source_index, manifest in enumerate(manifests):
        with manifest.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fieldnames:
                raise ValueError(f"Manifest schema mismatch: {manifest}")
            scene_name = manifest.parent.relative_to(output_path.parent).as_posix()
            for row in reader:
                row = dict(row)
                row["window"] = f"scene_{source_index:04d}/{scene_name}/{row['window']}"
                for key in PATH_COLUMNS:
                    path = Path(row[key]).expanduser()
                    if not path.is_absolute():
                        path = manifest.parent / path
                    row[key] = str(path.resolve())
                rows.append(row)
    if not rows:
        raise RuntimeError("No Wat3R scene manifests were generated")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(rows)} window-frame rows into {output_path}")


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    list_root = output_root / "_scene_lists"
    list_root.mkdir(parents=True, exist_ok=True)

    grouped_images = defaultdict(set)
    with args.train_list.expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            image_path = Path(line.split()[0]).expanduser().resolve()
            key = scene_key(image_path, dataset_root)
            grouped_images[key].add(image_path)
    if not grouped_images:
        raise RuntimeError("The training split contains no images")

    generator = Path(__file__).with_name("generate_wat3r_teacher_windows.py")
    manifests = []
    print(f"Discovered {len(grouped_images)} training scenes")
    for scene_index, key in enumerate(sorted(grouped_images, key=lambda value: value.as_posix())):
        images = sorted(grouped_images[key], key=lambda path: path.name)
        if len(images) < 3:
            print(f"Skipping {key}: only {len(images)} training frames")
            continue
        safe_name = "__".join(key.parts)
        image_list = list_root / f"{safe_name}.txt"
        image_list.write_text("".join(f"{path}\n" for path in images), encoding="utf-8")
        scene_output = output_root / key
        manifest = scene_output / "manifest.csv"
        print(f"[{scene_index + 1}/{len(grouped_images)}] {key}: {len(images)} train frames")
        if manifest.exists() and not args.overwrite:
            print(f"  Reusing existing manifest: {manifest}")
            manifests.append(manifest)
            continue
        command = [
            sys.executable,
            str(generator),
            "--wat3r-root",
            str(args.wat3r_root.expanduser().resolve()),
            "--checkpoint",
            str(args.checkpoint.expanduser().resolve()),
            "--image-list",
            str(image_list),
            "--output-dir",
            str(scene_output),
            "--device",
            args.device,
            "--target-size",
            str(args.target_size),
            "--window-size",
            str(args.window_size),
            "--overlap",
            str(args.overlap),
            "--frames-chunk-size",
            str(args.frames_chunk_size),
            "--min-visible-views",
            str(args.min_visible_views),
            "--relative-depth-threshold",
            "0.05",
            "--boundary",
            "4",
        ]
        if args.teacher_weights:
            command.append("--teacher-weights")
        if args.save_visualization:
            command.append("--save-visualization")
        subprocess.run(command, check=True)
        manifests.append(manifest)

    merged_manifest = output_root / "manifest_all.csv"
    merge_manifests(manifests, merged_manifest)
    print("Teacher dataset generation complete")
    print(f"Merged manifest: {merged_manifest}")


if __name__ == "__main__":
    main()
