"""
MVTec AD dataset loader with few-shot normal-sample sampling.

Directory layout expected:
  data/mvtec/<category>/train/good/*.png
  data/mvtec/<category>/test/<defect_type>/*.png
  data/mvtec/<category>/ground_truth/<defect_type>/*.png   (absent for 'good')

For few-shot training we randomly sample k normal images per category.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

# Canonical resize used throughout the paper (ViT-B/16 CLIP input)
IMAGE_SIZE = 224


def _default_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),  # CLIP stats
    ])


def _mask_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])


class MVTecTrainDataset(Dataset):
    """
    Training split: only normal ('good') images.

    few_shot_k: if set, randomly pick k images from all good images.
    seed:       controls which k images are picked (for reproducibility).
    """

    def __init__(
        self,
        root: str,
        category: str,
        few_shot_k: Optional[int] = None,
        seed: int = 42,
        image_size: int = IMAGE_SIZE,
    ):
        self.root = Path(root)
        self.category = category
        self.transform = _default_transform(image_size)

        good_dir = self.root / category / "train" / "good"
        all_images = sorted(good_dir.glob("*.png")) + sorted(good_dir.glob("*.jpg"))
        if not all_images:
            raise FileNotFoundError(f"No training images found at {good_dir}")

        if few_shot_k is not None and few_shot_k < len(all_images):
            rng = random.Random(seed)
            self.image_paths = rng.sample(all_images, few_shot_k)
        else:
            self.image_paths = all_images

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), 0, str(path)  # label 0 = normal


class MVTecTestDataset(Dataset):
    """
    Test split: all images (normal + defective) with pixel-level GT masks.
    Returns (image, label, mask, defect_type, path).
      label: 0 = normal, 1 = anomalous
      mask:  binary tensor H×W (1 where anomaly pixels are)
    """

    def __init__(
        self,
        root: str,
        category: str,
        image_size: int = IMAGE_SIZE,
    ):
        self.root = Path(root)
        self.category = category
        self.transform = _default_transform(image_size)
        self.mask_transform = _mask_transform(image_size)

        self.samples: List[Tuple[Path, int, Optional[Path], str]] = []

        test_root = self.root / category / "test"
        for defect_dir in sorted(test_root.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect_type = defect_dir.name
            is_anomalous = defect_type != "good"

            for img_path in sorted(defect_dir.glob("*.png")) + sorted(defect_dir.glob("*.jpg")):
                if is_anomalous:
                    # ground truth mask lives under ground_truth/<defect_type>/<stem>_mask.png
                    mask_path = (
                        self.root / category / "ground_truth" / defect_type
                        / (img_path.stem + "_mask.png")
                    )
                    if not mask_path.exists():
                        mask_path = None
                else:
                    mask_path = None

                self.samples.append((img_path, int(is_anomalous), mask_path, defect_type))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label, mask_path, defect_type = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if mask_path is not None and mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float().squeeze(0)  # H×W binary
        else:
            mask = torch.zeros(image.shape[1], image.shape[2])

        return image, label, mask, defect_type, str(img_path)


