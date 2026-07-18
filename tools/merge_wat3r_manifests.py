#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


PATH_COLUMNS = ("image", "teacher_depth", "teacher_confidence", "static_mask", "camera")


def main():
    parser = argparse.ArgumentParser(description="Merge scene-level Wat3R manifests.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()

    rows = []
    fieldnames = None
    for manifest in args.manifests:
        manifest = manifest.expanduser().resolve()
        with manifest.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fieldnames:
                raise ValueError(f"Manifest schema mismatch: {manifest}")
            scene_name = manifest.parent.name
            for row in reader:
                row = dict(row)
                row["window"] = f"{scene_name}/{row['window']}"
                for key in PATH_COLUMNS:
                    path = Path(row[key]).expanduser()
                    if not path.is_absolute():
                        path = manifest.parent / path
                    row[key] = str(path.resolve())
                rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(rows)} rows from {len(args.manifests)} manifests into {args.output}")


if __name__ == "__main__":
    main()
