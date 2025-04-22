# datasets/preprocessed_raw_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import csv

class RawAudioDatasetPreprocessed(Dataset):
    """
    Dataset for loading pairs of PREPROCESSED raw audio files (saved as .npy).
    Assumes audio is already resampled and padded/truncated to a fixed length.
    """
    def __init__(self, csv_file_path):
        """
        Args:
            csv_file_path (string or Path): Path to the csv file with pairs
                                            listing paths to .npy files.
                                            Format: file1.npy,file2.npy,label
        """
        self.csv_file_path = Path(csv_file_path)
        if not self.csv_file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")

        self.pairs = []
        print(f"Loading pairs from preprocessed CSV: {self.csv_file_path}")
        try:
            with open(self.csv_file_path, 'r', newline='') as infile:
                reader = csv.reader(infile)
                header = next(reader) # Skip header
                for i, row in enumerate(reader):
                    if len(row) == 3:
                        npy_path1, npy_path2, label_str = row
                        # Basic check if paths seem valid (end with .npy)
                        if not npy_path1.endswith(".npy") or not npy_path2.endswith(".npy"):
                             print(f"Warning: Row {i+1} in {self.csv_file_path} does not contain expected .npy paths: {row}. Skipping.")
                             continue
                        try:
                             label = int(label_str)
                             self.pairs.append((npy_path1, npy_path2, label))
                        except ValueError:
                             print(f"Warning: Invalid label in row {i+1} of {self.csv_file_path}: {label_str}. Skipping row.")
                    else:
                         print(f"Warning: Skipping malformed row {i+1} in {self.csv_file_path}: {row}")
            print(f"Loaded {len(self.pairs)} pairs.")
            if not self.pairs:
                 print("Warning: No valid pairs loaded from CSV.")

        except Exception as e:
            print(f"Error reading CSV file {self.csv_file_path}: {e}")
            raise

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the pair to retrieve.

        Returns:
            tuple: (audio1, audio2, label) where audio1 and audio2 are
                   torch tensors of the preprocessed audio data.
        """
        if idx >= len(self.pairs):
            raise IndexError("Index out of bounds")

        npy_path1, npy_path2, label = self.pairs[idx]

        try:
            # Load the preprocessed numpy arrays
            audio1_np = np.load(npy_path1)
            audio2_np = np.load(npy_path2)

            # Convert to torch tensors (ensure float32)
            audio1 = torch.from_numpy(audio1_np.astype(np.float32))
            audio2 = torch.from_numpy(audio2_np.astype(np.float32))

            # Add channel dimension if model expects (B, 1, L) - SincNet does
            if audio1.ndim == 1:
                audio1 = audio1.unsqueeze(0)
            if audio2.ndim == 1:
                audio2 = audio2.unsqueeze(0)

            return audio1, audio2, label

        except FileNotFoundError as e:
            print(f"ERROR in dataset: Cannot find file {e.filename} for index {idx}. Check CSV and preprocessed files.")
            # Return dummy data or raise error? Let's raise an error.
            raise FileNotFoundError(f"Missing preprocessed file: {e.filename}") from e
        except Exception as e:
            print(f"ERROR loading data for index {idx} (Paths: {npy_path1}, {npy_path2}): {e}")
            # Return dummy data or raise error?
            raise RuntimeError(f"Failed to load data for index {idx}") from e

