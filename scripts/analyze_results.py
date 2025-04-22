# scripts/analyze_results.py
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_training_losses(stats, labels, out_dir: Path):
    """
    stats: list of lists of dicts [{"epoch":…, "train_loss":…}, …]
    labels: list of model names
    """
    plt.figure()
    for s, label in zip(stats, labels):
        epochs = [e["epoch"] for e in s]
        losses = [e["train_loss"] for e in s]
        plt.plot(epochs, losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "train_losses.png")
    plt.close()


def plot_eval_metrics(eval_stats, out_dir: Path):
    """
    eval_stats: dict {
       "clean": {...},
       "noisy": {...},
       "filtered": {...}
    }
    """
    # metrics to plot
    metrics = ["accuracy", "auc", "fmr", "fnmr", "eer"]
    splits  = list(eval_stats.keys())

    for metric in metrics:
        values = [eval_stats[split][metric] for split in splits]
        plt.figure()
        plt.bar(splits, values)
        plt.xlabel("Test Split")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} by Test Split")
        plt.savefig(out_dir / f"eval_{metric}.png")
        plt.close()

    # optionally confusion matrices printed out
    cm_dir = out_dir / "cms"
    cm_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        cm = eval_stats[split]["confusion"]
        # display as text table
        with open(cm_dir / f"cm_{split}.txt", "w") as f:
            f.write(f"Confusion matrix for {split}:\n")
            f.write(f"  TP: {cm['tp']}   FP: {cm['fp']}\n")
            f.write(f"  FN: {cm['fn']}   TN: {cm['tn']}\n")


def main():
    p = argparse.ArgumentParser(
        description="Analyze & visualize training/eval metrics"
    )
    p.add_argument(
        "--metrics_dir", type=Path, required=True,
        help="Folder containing JSON metrics files"
    )
    p.add_argument(
        "--output_dir", type=Path, required=True,
        help="Where to save figures and tables"
    )
    args = p.parse_args()

    md = args.metrics_dir
    od = args.output_dir

    # 1. load training stats
    s_stats = load_json(md / "sincnet_train_stats.json")
    m_stats = load_json(md / "mobilenetv2_train_stats.json")
    f_stats = load_json(md / "fusion_train_stats.json")

    # 2. load evaluation stats
    eval_stats = load_json(md / "fusion_eval_stats.json")

    # 3. plot losses
    plot_training_losses(
        [s_stats, m_stats, f_stats],
        ["SincNet", "MobileNetV2", "Fusion"],
        od
    )

    # 4. plot evaluation
    plot_eval_metrics(eval_stats, od)

    print(f"Saved all analysis figures and tables to {od.resolve()}")


if __name__ == "__main__":
    main()