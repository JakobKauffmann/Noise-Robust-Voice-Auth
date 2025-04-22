#!/usr/bin/env python3
# datasets/spectrogram_dataset.py

import csv
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class MelSpectrogramPairDataset(Dataset):
    """
    Loads pairs of spectrogram images (PNG or JPEG) from a CSV.
    Returns (img_tensor1, img_tensor2, label).
    """

    def __init__(
        self,
        csv_path: str,
        img_size: int = 224,
        normalize: bool = True,
    ):
        """
        csv_path: CSV with columns file1,file2,label
        img_size: resize images to (img_size x img_size)
        normalize: apply ImageNet normalization if True
        """
        self.pairs = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.pairs.append((row[0], row[1], int(row[2])))

        transforms = [T.Resize((img_size, img_size)), T.ToTensor()]
        if normalize:
            transforms.append(
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225],
                )
            )
        self.transform = T.Compose(transforms)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        x1 = self.transform(img1)
        x2 = self.transform(img2)
        return x1, x2, label