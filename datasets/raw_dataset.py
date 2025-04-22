# datasets/raw_dataset.py

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchaudio


class RawAudioPairDataset(Dataset):
    """
    Loads pairs of .wav files and labels from a CSV.
    Returns (waveform1, waveform2, label).
    """

    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 16000,
        duration: float = 3.0,
        normalize: bool = True,
        resample: bool = True,
    ):
        self.pairs = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                self.pairs.append((row[0], row[1], int(row[2])))

        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.normalize = normalize
        self.resample = resample

    def __len__(self):
        return len(self.pairs)

    def _load_clip(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)  # [channels, L]

        # resample if needed
        if self.resample and sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )

        # mix to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # pad or truncate
        if waveform.shape[1] < self.num_samples:
            pad_amt = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
        else:
            waveform = waveform[:, : self.num_samples]

        # normalize
        if self.normalize:
            mean = waveform.mean()
            std = waveform.std(unbiased=False) + 1e-9
            waveform = (waveform - mean) / std

        return waveform  # [1, num_samples]

    def __getitem__(self, idx):
        f1, f2, label = self.pairs[idx]
        x1 = self._load_clip(f1)
        x2 = self._load_clip(f2)
        return x1, x2, label
# # datasets/raw_dataset.py
# from pathlib import Path
# import csv
# import torch
# from torch.utils.data import Dataset
# import torchaudio

# class RawAudioPairDataset(Dataset):
#     """
#     Loads pairs of .wav files and labels from a CSV.
#     Returns (waveform1, waveform2, label).
#     """
#     def __init__(
#         self,
#         csv_path: str,
#         sample_rate: int = 16000,
#         duration: float = 3.0,
#         normalize: bool = True,
#         resample: bool = True,
#     ):
#         self.pairs = []
#         with open(csv_path, newline="") as f:
#             reader = csv.reader(f)
#             next(reader)  # skip header
#             for row in reader:
#                 self.pairs.append((row[0], row[1], int(row[2])))

#         self.sample_rate = sample_rate
#         self.num_samples = int(sample_rate * duration)
#         self.normalize = normalize

#         if resample:
#             self.resampler = torchaudio.transforms.Resample(orig_freq=0, new_freq=sample_rate)
#         else:
#             self.resampler = None

#     def __len__(self):
#         return len(self.pairs)

#     def _load_clip(self, path: str) -> torch.Tensor:
#         waveform, sr = torchaudio.load(path)
#         if self.resampler and sr != self.sample_rate:
#             self.resampler.orig_freq = sr
#             waveform = self.resampler(waveform)

#         # mono
#         if waveform.size(0) > 1:
#             waveform = waveform.mean(dim=0, keepdim=True)

#         # pad/truncate
#         if waveform.size(1) < self.num_samples:
#             pad = self.num_samples - waveform.size(1)
#             waveform = torch.nn.functional.pad(waveform, (0, pad))
#         else:
#             waveform = waveform[:, : self.num_samples]

#         if self.normalize:
#             mean = waveform.mean()
#             std = waveform.std(unbiased=False) + 1e-9
#             waveform = (waveform - mean) / std

#         return waveform

#     def __getitem__(self, idx):
#         f1, f2, label = self.pairs[idx]
#         x1 = self._load_clip(f1)
#         x2 = self._load_clip(f2)
#         return x1, x2, label
