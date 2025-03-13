#!/usr/bin/env python
"""
pair_train_test.py
Loops through all matching train-test pairs based on unique IDs.
Usage:
    python pair_train_test.py <train_folder> <test_folder> <file_extension>
Example:
    python pair_train_test.py ./data/train ./data/test .wav
"""

import os
import sys

def pair_train_test(train_dir, test_dir, ext=".wav"):
    """
    Returns a list of (train_file, test_file) tuples for files that have the same basename.
    """
    train_files = {os.path.splitext(f)[0]: os.path.join(train_dir, f)
                   for f in os.listdir(train_dir) if f.endswith(ext)}
    test_files = {os.path.splitext(f)[0]: os.path.join(test_dir, f)
                  for f in os.listdir(test_dir) if f.endswith(ext)}
    
    pairs = []
    missing_in_test = []
    for uid, train_path in train_files.items():
        if uid in test_files:
            pairs.append((train_path, test_files[uid]))
        else:
            missing_in_test.append(uid)
    
    if missing_in_test:
        print("Warning: The following IDs were found in train but not in test:")
        for uid in missing_in_test:
            print(f" - {uid}")
    
    return pairs

def main():
    if len(sys.argv) < 3:
        print("Usage: python pair_train_test.py <train_folder> <test_folder> [file_extension]")
        sys.exit(1)
    
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    ext = sys.argv[3] if len(sys.argv) > 3 else ".wav"
    
    pairs = pair_train_test(train_dir, test_dir, ext)
    print(f"Found {len(pairs)} matching pairs:")
    for train_file, test_file in pairs:
        print(f"Train: {train_file}  |  Test: {test_file}")

if __name__ == "__main__":
    main()
