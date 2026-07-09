import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Crop, NormalizeImage, PrepareForNet, Resize


class FLSea(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        self.mode = mode
        self.size = size
        with open(filelist_path, "r", encoding="utf-8") as handle:
            self.filelist = [line.strip() for line in handle if line.strip()]

        net_w, net_h = size
        transforms = [
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
        if self.mode == "train":
            transforms.append(Crop(size[0]))
        self.transform = Compose(transforms)

    def __len__(self):
        return len(self.filelist)

    @staticmethod
    def _read_depth(depth_path):
        ext = os.path.splitext(depth_path)[1].lower()
        if ext == ".npy":
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"Failed to read depth: {depth_path}")
            depth = depth.astype(np.float32)
            if ext in {".png", ".tif", ".tiff"} and depth.max() > 255:
                depth = depth / 1000.0
        return depth

    def __getitem__(self, index):
        parts = self.filelist[index].split()
        if len(parts) < 2:
            raise ValueError(
                "Each FLSea split line must contain at least image_path and depth_path."
            )

        image_path, depth_path = parts[0], parts[1]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        depth = self._read_depth(depth_path)

        sample = self.transform({"image": image, "depth": depth})
        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])

        valid_mask = torch.isfinite(sample["depth"]) & (sample["depth"] > 0)
        sample["valid_mask"] = valid_mask
        sample["depth"][~valid_mask] = 0.0

        sample["image_path"] = image_path
        sample["depth_path"] = depth_path
        return sample

