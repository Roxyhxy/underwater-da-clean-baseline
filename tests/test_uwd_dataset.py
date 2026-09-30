import json

import h5py
import numpy as np
import torch
from PIL import Image

from dataset.uwd import UWD, WATER_TYPE_TO_LABEL
from tools.build_uwd_manifests import discover_samples, split_scenes


def _write_sample(root, scene, water_type, frame, value):
    rgb_dir = root / "uw" / scene / water_type
    depth_dir = root / "depth" / scene
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.full((6, 8, 3), value, dtype=np.uint8)
    Image.fromarray(rgb).save(rgb_dir / f"uw_{frame}.png")
    with h5py.File(depth_dir / f"depth_{frame}.hdf5", "w") as handle:
        handle.create_dataset("dataset", data=np.full((6, 8), value / 10, np.float16))


def test_manifest_discovery_filters_appledouble_and_pairs_depth(tmp_path):
    _write_sample(tmp_path, "ai_001_001", "Type-1", "00_00", 10)
    metadata = tmp_path / "uw" / "ai_001_001" / "Type-1" / "._uw_00_00.png"
    metadata.write_bytes(b"not an image")

    samples = discover_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0]["depth"] == "depth/ai_001_001/depth_00_00.hdf5"


def test_scene_split_is_disjoint_and_deterministic():
    scenes = [f"ai_{index:03d}_001" for index in range(20)]
    first = split_scenes(scenes, 0.8, 0.1, seed=42)
    second = split_scenes(scenes, 0.8, 0.1, seed=42)
    assert first == second
    assert len(first["train"]) == 16
    assert len(first["val"]) == 2
    assert len(first["test"]) == 2
    assert not (first["train"] & first["val"])
    assert not (first["train"] & first["test"])
    assert not (first["val"] & first["test"])


def test_dataset_returns_aligned_float32_tensors(tmp_path):
    _write_sample(tmp_path, "ai_001_001", "Type-II", "00_00", 20)
    record = discover_samples(tmp_path)[0]
    manifest = tmp_path / "val.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    sample = UWD(tmp_path, manifest, mode="val", size=(4, 4))[0]
    assert sample["image"].shape == (3, 4, 4)
    assert sample["depth"].shape == (4, 4)
    assert sample["image"].dtype == torch.float32
    assert sample["depth"].dtype == torch.float32
    assert sample["valid_mask"].all()
    assert sample["water_label"].item() == WATER_TYPE_TO_LABEL["Type-II"]
