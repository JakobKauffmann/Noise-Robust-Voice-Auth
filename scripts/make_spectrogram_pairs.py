#!/usr/bin/env python3
# scripts/make_spectrogram_pairs.py

import argparse, csv, os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(
        description="Make mel‑spectrogram images (once per WAV) + pair CSV"
    )
    p.add_argument("input_csv", type=Path,
                   help="CSV of raw WAV pairs: wav1,wav2,label")
    p.add_argument("spec_dir", type=Path,
                   help="Directory to save spectrogram images")
    p.add_argument("output_csv", type=Path,
                   help="Output CSV: img1,img2,label")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--n_mels",      type=int, default=80)
    p.add_argument("--n_fft",       type=int, default=512)
    p.add_argument("--hop_length",  type=int, default=160)
    p.add_argument("--win_length",  type=int, default=400)
    p.add_argument("--duration",    type=float, default=3.0)
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--img_format",  choices=["png", "jpg"], default="png")
    p.add_argument("--quality",     type=int, default=60,
                   help="JPEG quality (if img_format=jpg)")
    return p.parse_args()


def make_spec(
    wav_path: Path,
    img_path: Path,
    mel_tf, db_tf,
    num_samples: int,
    img_size: int,
    img_format: str,
    quality: int,
):
    # load & mono
    wav, sr = torchaudio.load(wav_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # resample
    if sr != mel_tf.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, mel_tf.sample_rate)
    # pad/trunc
    if wav.size(1) < num_samples:
        wav = torch.nn.functional.pad(wav, (0, num_samples - wav.size(1)))
    else:
        wav = wav[:, :num_samples]
    # mel → dB
    mel = mel_tf(wav)
    mel_db = db_tf(mel).squeeze(0).cpu().numpy()
    # normalize 0–255
    arr = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    img_arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(img_arr) \
               .resize((img_size, img_size), Image.BILINEAR) \
               .convert("RGB")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    if img_format == "png":
        img.save(img_path)
    else:
        img.save(img_path.with_suffix(".jpg"), "JPEG", quality=quality)


def main():
    args = parse_args()

    # load pairs, collect unique WAVs
    pairs = []
    wav_set = set()
    with args.input_csv.open(newline="") as f:
        rdr = csv.reader(f)
        next(rdr)
        for a, b, lbl in rdr:
            pairs.append((Path(a), Path(b), lbl))
            wav_set.add(Path(a))
            wav_set.add(Path(b))

    # determine common root to mirror directory structure
    all_paths = [str(p) for p in wav_set]
    orig_root = Path(os.path.commonpath(all_paths))

    # prepare transforms
    mel_tf = MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
    )
    db_tf       = AmplitudeToDB()
    num_samples = int(args.sample_rate * args.duration)

    # 1) Generate each spectrogram once
    print(f"Found {len(wav_set)} unique WAVs → generating specs …")
    wav_to_img = {}
    for wav_path in tqdm(wav_set, desc="Spec files"):
        # compute relative path under orig_root
        rel = wav_path.relative_to(orig_root).with_suffix(f".{args.img_format}")
        img_path = args.spec_dir / rel

        # if missing, create it
        if not img_path.exists():
            make_spec(
                wav_path, img_path,
                mel_tf, db_tf, num_samples,
                args.img_size, args.img_format, args.quality
            )
        wav_to_img[wav_path] = img_path

    # 2) Write pair CSV pointing to already-created images
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as fout:
        wtr = csv.writer(fout)
        wtr.writerow(["file1", "file2", "label"])
        for a, b, lbl in pairs:
            img1 = wav_to_img[a]
            img2 = wav_to_img[b]
            wtr.writerow([str(img1), str(img2), lbl])

    print("Done.")
    

if __name__ == "__main__":
    main()