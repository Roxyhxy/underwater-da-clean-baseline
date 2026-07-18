import csv
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.flsea import FLSea


def _normalized_path(path):
    return os.path.normcase(os.path.abspath(os.path.expanduser(path)))


def _resolve_path(value, manifest_dir):
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return str(path.resolve())


class FLSeaWat3R(Dataset):
    """FLSea center-frame supervision with privileged Wat3R window geometry."""

    def __init__(self, filelist_path, teacher_manifest, size=(518, 518), frame_stride=1):
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        self.size = tuple(size)
        self.frame_stride = int(frame_stride)
        self.gt_by_image = {}
        self.gt_by_stem = defaultdict(list)
        with open(filelist_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 2:
                    raise ValueError("Each FLSea split line must contain image_path and depth_path")
                image_path, depth_path = parts[:2]
                self.gt_by_image[_normalized_path(image_path)] = depth_path
                self.gt_by_stem[Path(image_path).stem].append(depth_path)

        manifest_path = Path(teacher_manifest).expanduser().resolve()
        manifest_dir = manifest_path.parent
        groups = defaultdict(list)
        required = {
            "window",
            "local_index",
            "image",
            "teacher_depth",
            "teacher_confidence",
            "static_mask",
            "camera",
        }
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Wat3R manifest is missing columns: {sorted(missing)}")
            for row in reader:
                record = dict(row)
                record["local_index"] = int(record["local_index"])
                for key in ("image", "teacher_depth", "teacher_confidence", "static_mask", "camera"):
                    record[key] = _resolve_path(record[key], manifest_dir)
                groups[record["window"]].append(record)

        self.samples = []
        for window, records in groups.items():
            records.sort(key=lambda item: item["local_index"])
            by_index = {item["local_index"]: item for item in records}
            for center in records:
                center_index = center["local_index"]
                neighbor_indices = (
                    center_index - self.frame_stride,
                    center_index,
                    center_index + self.frame_stride,
                )
                if any(index not in by_index for index in neighbor_indices):
                    continue
                depth_path = self.gt_by_image.get(_normalized_path(center["image"]))
                if depth_path is None:
                    candidates = self.gt_by_stem.get(Path(center["image"]).stem, [])
                    if len(candidates) == 1:
                        depth_path = candidates[0]
                if depth_path is None:
                    continue
                self.samples.append(
                    {
                        "window": window,
                        "views": [by_index[index] for index in neighbor_indices],
                        "depth_path": depth_path,
                    }
                )
        if not self.samples:
            raise RuntimeError(
                "No Wat3R triplets matched the FLSea split. Check image paths and frame stride."
            )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read_map(path, name):
        if Path(path).suffix.lower() == ".npy":
            value = np.load(path)
        else:
            value = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if value is None:
            raise FileNotFoundError(f"Failed to read {name}: {path}")
        if value.ndim == 3:
            value = value[..., 0]
        return value.astype(np.float32)

    @staticmethod
    def _resize_shape(height, width, target_height, target_width, multiple=14):
        scale = max(target_height / height, target_width / width)
        new_height = max(target_height, int(np.round(height * scale / multiple) * multiple))
        new_width = max(target_width, int(np.round(width * scale / multiple) * multiple))
        return new_height, new_width

    def __getitem__(self, index):
        entry = self.samples[index]
        raw_images = []
        teacher_depths = []
        teacher_confidences = []
        static_masks = []
        intrinsics = []
        extrinsics = []

        original_shape = None
        for view in entry["views"]:
            image = cv2.imread(view["image"], cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {view['image']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            depth = self._read_map(view["teacher_depth"], "teacher depth")
            confidence = self._read_map(view["teacher_confidence"], "teacher confidence")
            static = self._read_map(view["static_mask"], "static mask") > 0
            if image.shape[:2] != depth.shape or depth.shape != confidence.shape or depth.shape != static.shape:
                raise ValueError(f"Teacher maps do not match RGB shape for {view['image']}")
            if original_shape is None:
                original_shape = image.shape[:2]
            elif original_shape != image.shape[:2]:
                raise ValueError("All views in a Wat3R window triplet must have the same shape")
            camera = np.load(view["camera"])
            raw_images.append(image)
            teacher_depths.append(depth)
            teacher_confidences.append(confidence)
            static_masks.append(static)
            intrinsics.append(np.asarray(camera["intrinsics"], dtype=np.float32))
            extrinsics.append(np.asarray(camera["extrinsics"], dtype=np.float32))

        gt_depth = FLSea._read_depth(entry["depth_path"])
        if gt_depth.shape != original_shape:
            raise ValueError(f"GT depth does not match center RGB: {entry['depth_path']}")

        target_width, target_height = self.size
        height, width = original_shape
        new_height, new_width = self._resize_shape(
            height, width, target_height, target_width
        )
        crop_y = np.random.randint(0, new_height - target_height + 1)
        crop_x = np.random.randint(0, new_width - target_width + 1)
        crop = np.s_[crop_y : crop_y + target_height, crop_x : crop_x + target_width]
        scale_x = new_width / float(width)
        scale_y = new_height / float(height)

        transformed_images = []
        transformed_depths = []
        transformed_confidences = []
        transformed_static = []
        transformed_intrinsics = []
        for image, depth, confidence, static, intrinsic in zip(
            raw_images,
            teacher_depths,
            teacher_confidences,
            static_masks,
            intrinsics,
        ):
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)[crop]
            depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_LINEAR)[crop]
            confidence = cv2.resize(
                confidence, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            )[crop]
            static = cv2.resize(
                static.astype(np.uint8),
                (new_width, new_height),
                interpolation=cv2.INTER_NEAREST,
            )[crop]
            intrinsic = intrinsic.copy()
            intrinsic[0] *= scale_x
            intrinsic[1] *= scale_y
            intrinsic[0, 2] -= crop_x
            intrinsic[1, 2] -= crop_y
            image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
                [0.229, 0.224, 0.225], dtype=np.float32
            )
            transformed_images.append(np.ascontiguousarray(image.transpose(2, 0, 1)))
            transformed_depths.append(np.ascontiguousarray(depth.astype(np.float32)))
            transformed_confidences.append(np.ascontiguousarray(confidence.astype(np.float32)))
            transformed_static.append(np.ascontiguousarray(static.astype(bool)))
            transformed_intrinsics.append(intrinsic)

        gt_depth = cv2.resize(
            gt_depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )[crop].astype(np.float32)
        valid_mask = np.isfinite(gt_depth) & (gt_depth > 0)
        gt_depth[~valid_mask] = 0.0

        return {
            "images": torch.from_numpy(np.stack(transformed_images)),
            "image": torch.from_numpy(transformed_images[1]),
            "depth": torch.from_numpy(np.ascontiguousarray(gt_depth)),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask)),
            "teacher_depth": torch.from_numpy(np.stack(transformed_depths)),
            "teacher_confidence": torch.from_numpy(np.stack(transformed_confidences)),
            "teacher_static_mask": torch.from_numpy(np.stack(transformed_static)),
            "intrinsics": torch.from_numpy(np.stack(transformed_intrinsics)),
            "extrinsics": torch.from_numpy(np.stack(extrinsics).astype(np.float32)),
            "image_path": entry["views"][1]["image"],
            "depth_path": entry["depth_path"],
            "teacher_window": entry["window"],
        }
