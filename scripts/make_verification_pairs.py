# scripts/make_verification_pairs.py
import argparse
import random
import csv
from itertools import combinations
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Generate verification pairs CSV")
    p.add_argument("dataset_root", type=Path,
                   help="Root folder (e.g. noisy_train/)")
    p.add_argument("output_csv", type=Path,
                   help="CSV output path")
    p.add_argument("--imposter_ratio", type=float, default=3.0,
                   help="Neg/pos ratio per speaker")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    speaker_files = {}
    for spk_dir in args.dataset_root.iterdir():
        if not spk_dir.is_dir(): continue
        files = list(spk_dir.rglob("*.wav"))
        if len(files) < 2: continue
        speaker_files[spk_dir.name] = [str(f) for f in files]

    genuine = []
    for spk, files in speaker_files.items():
        for f1, f2 in combinations(files, 2):
            genuine.append((f1, f2, 1))

    imposter = []
    for spk, files in speaker_files.items():
        pos_count = len(list(combinations(files, 2)))
        neg_count = int(args.imposter_ratio * pos_count)
        others = list(speaker_files.keys())
        others.remove(spk)
        for _ in range(neg_count):
            f1 = random.choice(files)
            f2 = random.choice(speaker_files[random.choice(others)])
            imposter.append((f1, f2, 0))

    all_pairs = genuine + imposter
    random.shuffle(all_pairs)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file1","file2","label"])
        for a,b,l in all_pairs:
            w.writerow([a,b,l])

if __name__=="__main__":
    main()
