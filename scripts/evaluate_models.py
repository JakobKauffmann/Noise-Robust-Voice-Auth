#!/usr/bin/env python3
"""
evaluate_models.py

Run end-to-end evaluation of SincNet, MobileNetV2, and Fusion models
on all specified test conditions and save metrics to JSON.
"""

import argparse
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

# --- Safe imports of your project modules ---
try:
    from models.sincnet import SincNetEmbedding
    from models.mobilenet_embedding import MobileNetV2Embedding
    from scripts.train_fusion import FusionClassifier
    from scripts.train_sincnet import PairClassifier as SincPairClassifier
    from scripts.train_mobilenetv2 import PairClassifier as MobilePairClassifier
    from datasets.preprocessed_raw_dataset import RawAudioDatasetPreprocessed
    from datasets.spectrogram_dataset import MelSpectrogramPairDataset
except ImportError as e:
    print(f"ERROR: Could not import modules: {e}")
    print("Make sure your PYTHONPATH includes your project root.")
    raise

# --- Helper Functions ---

def load_state_dict_flexible(model, state_dict, strict=True):
    """Loads a state_dict handling 'module.' and '_orig_mod.' prefixes."""
    if not state_dict or not isinstance(state_dict, dict):
        print("Warning: Received invalid state_dict; skipping.")
        return
    first_key = next(iter(state_dict))
    is_dp   = first_key.startswith("module.")
    is_comp = first_key.startswith("_orig_mod.")
    sd = state_dict
    if is_comp:
        sd = OrderedDict((k.replace("_orig_mod.", "", 1), v) for k,v in sd.items())
        first_key = next(iter(sd)); is_dp = first_key.startswith("module.")
    if is_dp:
        sd = OrderedDict((k[7:], v) for k,v in sd.items())
    try:
        model.load_state_dict(sd, strict=strict)
    except RuntimeError as e:
        print(f"Warning: strict load failed: {e}")
        if strict:
            model.load_state_dict(sd, strict=False)

def calculate_eer(y_true, y_score):
    """Compute Equal Error Rate (EER) and corresponding threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr, fill_value="extrapolate")(x), 0., 1.)
    try:
        thresh = interp1d(fpr, thresholds, fill_value="extrapolate")(eer)
    except ValueError:
        thresh = thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    return eer * 100, float(thresh)

def calculate_fmr_at_fnmr(y_true, y_score, target_fnmr):
    """Compute FMR at a target FNMR level."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnmr = 1 - tpr
    try:
        interp_fpr = interp1d(fnmr, fpr, fill_value="extrapolate")
        interp_thr = interp1d(fnmr, thresholds, fill_value="extrapolate")
        fmr_target = float(interp_fpr(target_fnmr) * 100)
        thr_target = float(interp_thr(target_fnmr))
        return fmr_target, thr_target, float(interp1d(thresholds, fnmr, fill_value="extrapolate")(thr_target) * 100)
    except Exception:
        return float("nan"), float("nan"), float("nan")

def main():
    parser = argparse.ArgumentParser(description="Evaluate biometric models on test sets")
    parser.add_argument("--workspace",  type=Path,  default=Path.cwd(),
                        help="Project root (contains data/ and outputs/)")
    parser.add_argument("--output-dir", type=Path,  default=Path("outputs"),
                        help="Where checkpoints and metrics live")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers",type=int, default=4)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WORKSPACE = args.workspace
    OUTPUT_DIR = args.output_dir
    RESULTS_FILE = OUTPUT_DIR / "metrics" / "evaluation_results_all_models.json"

    # checkpoint paths
    sinc_ckpt  = OUTPUT_DIR / "checkpoints" / "sincnet_best.pt"
    mob_ckpt   = OUTPUT_DIR / "checkpoints" / "mobilenetv2_best.pt"
    fus_ckpt   = OUTPUT_DIR / "checkpoints" / "fusion_best.pt"

    # test CSVs
    test_csvs = {
        "clean": {
            "raw":  WORKSPACE / "data/pairs/pairs_raw_clean_test_preprocessed_local.csv",
            "spec": WORKSPACE / "data/pairs/pairs_spec_clean_test_local.csv"
        },
        "noisy": {
            "raw":  WORKSPACE / "data/pairs/pairs_raw_noisy_test_preprocessed_local.csv",
            "spec": WORKSPACE / "data/pairs/pairs_spec_noisy_test_local.csv"
        },
        "filtered": {
            "raw":  WORKSPACE / "data/pairs/pairs_raw_filtered_test_preprocessed_local.csv",
            "spec": WORKSPACE / "data/pairs/pairs_spec_filtered_test_local.csv"
        }
    }

    # --- Load models ---
    print("Loading models...")
    models = {}

    if sinc_ckpt.exists():
        ck = torch.load(sinc_ckpt, map_location=DEVICE)
        m  = SincNetEmbedding().to(DEVICE)
        c  = SincPairClassifier().to(DEVICE)
        load_state_dict_flexible(m, ck.get("model", {}))
        load_state_dict_flexible(c, ck.get("clf", {}))
        models["sincnet"] = (m, c)

    if mob_ckpt.exists():
        ck = torch.load(mob_ckpt, map_location=DEVICE)
        m  = MobileNetV2Embedding(freeze=False).to(DEVICE)
        c  = MobilePairClassifier().to(DEVICE)
        load_state_dict_flexible(m, ck.get("model", {}), strict=False)
        load_state_dict_flexible(c, ck.get("clf", {}))
        models["mobilenet"] = (m, c)

    if fus_ckpt.exists() and sinc_ckpt.exists() and mob_ckpt.exists():
        fus_ck = torch.load(fus_ckpt, map_location=DEVICE)
        sinc_ck = torch.load(sinc_ckpt, map_location=DEVICE)
        mob_ck  = torch.load(mob_ckpt, map_location=DEVICE)
        m_sinc = SincNetEmbedding().to(DEVICE)
        m_mob  = MobileNetV2Embedding(freeze=False).to(DEVICE)
        c_fus  = FusionClassifier().to(DEVICE)
        load_state_dict_flexible(m_sinc, sinc_ck.get("model", {}))
        load_state_dict_flexible(m_mob,  mob_ck.get("model", {}), strict=False)
        load_state_dict_flexible(c_fus, fus_ck.get("clf", {}))
        models["fusion"] = (m_sinc, m_mob, c_fus)

    # --- Evaluate ---
    results = {}
    target_fnmrs = [0.01, 0.001]

    for cond, paths in test_csvs.items():
        print(f"Evaluating condition: {cond}")
        raw_csv, spec_csv = paths["raw"], paths["spec"]
        if not raw_csv.exists() or not spec_csv.exists():
            print(f"  Missing CSV for {cond}, skipping.")
            continue

        raw_ds  = RawAudioDatasetPreprocessed(raw_csv)
        spec_ds = MelSpectrogramPairDataset(spec_csv)
        raw_loader  = DataLoader(raw_ds,  batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
        spec_loader = DataLoader(spec_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)

        results[cond] = {}
        for name, comp in models.items():
            print(f"  Model: {name}")
            y_true, y_score = [], []

            try:
                if name == "fusion":
                    m_sinc, m_mob, clf = comp
                    m_sinc.eval(); m_mob.eval(); clf.eval()
                    raw_iter  = iter(raw_loader)
                    spec_iter = iter(spec_loader)
                    for _ in tqdm(range(min(len(raw_loader), len(spec_loader))),
                                  desc=f"    Fusion {cond}", leave=False):
                        w1, w2, y1 = next(raw_iter)
                        i1, i2, y2 = next(spec_iter)
                        y = y1.float().to(DEVICE)
                        with torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):
                            a1 = m_sinc(w1.to(DEVICE)); a2 = m_sinc(w2.to(DEVICE))
                            s1 = m_mob(i1.to(DEVICE)); s2 = m_mob(i2.to(DEVICE))
                            logits = clf(a1, a2, s1, s2)
                        y_true.append(y.cpu().numpy())
                        y_score.append(logits.cpu().numpy())
                else:
                    model, clf = comp
                    loader = raw_loader if name=="sincnet" else spec_loader
                    model.eval(); clf.eval()
                    for x1, x2, y in tqdm(loader, desc=f"    {name} {cond}", leave=False):
                        y_true.append(y.float().numpy())
                        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):
                            e1 = model(x1.to(DEVICE)); e2 = model(x2.to(DEVICE))
                            logits = clf(e1, e2)
                        y_score.append(logits.cpu().numpy())

                y_true  = np.concatenate(y_true).ravel()
                y_score = np.concatenate(y_score).ravel()

                eer, thresh = calculate_eer(y_true, y_score)
                auc_val = roc_auc_score(y_true, y_score)

                stats = {
                    "EER (%)": eer,
                    "AUC": auc_val,
                    "EER_Threshold": thresh,
                    "Num_Samples": len(y_true)
                }
                for fnmr in target_fnmrs:
                    fmr, thr, act_fnmr = calculate_fmr_at_fnmr(y_true, y_score, fnmr)
                    key = f"FMR_at_FNMR_{fnmr*100:.1f}%"
                    stats[key] = fmr
                    stats[f"{key}_Threshold"] = thr
                    stats[f"Actual_FNMR_{fnmr*100:.1f}%"] = act_fnmr

                results[cond][name] = stats
                print(f"    => EER {eer:.2f}%, AUC {auc_val:.4f}")
            except Exception as e:
                print(f"    ERROR evaluating {name}: {e}")
                results[cond][name] = {"error": str(e)}

    # --- Save and print ---
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to {RESULTS_FILE}\n")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
