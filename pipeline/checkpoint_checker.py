#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SincNet Checkpoint Analyzer

This utility script analyzes SincNet checkpoint files and provides information about training progress,
allowing you to make informed decisions about resuming training.

Usage:
    python checkpoint_checker.py --checkpoint_dir /path/to/checkpoints --config_file /path/to/config.json
"""

import os
import torch
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import glob
import time


def get_checkpoint_info(checkpoint_path):
    """Extract key information from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'train_loss': checkpoint.get('train_loss', 'Unknown'),
            'train_acc': checkpoint.get('train_acc', 'Unknown'),
            'val_loss': checkpoint.get('val_loss', 'Unknown'),
            'val_acc': checkpoint.get('val_acc', 'Unknown'),
            'best_acc': checkpoint.get('best_acc', 'Unknown'),
            'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
            'modified_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(checkpoint_path)))
        }
        
        # Check if history is available
        if 'history' in checkpoint:
            info['has_history'] = True
            info['history_epochs'] = len(checkpoint['history'].get('train_loss', []))
        else:
            info['has_history'] = False
            
        return info
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
        return None


def plot_checkpoint_metrics(checkpoints_info):
    """Generate plots of training metrics from checkpoints."""
    epochs = [info['epoch'] for info in checkpoints_info if info is not None]
    train_losses = [info['train_loss'] for info in checkpoints_info if info is not None and info['train_loss'] != 'Unknown']
    val_losses = [info['val_loss'] for info in checkpoints_info if info is not None and info['val_loss'] != 'Unknown']
    train_accs = [info['train_acc'] for info in checkpoints_info if info is not None and info['train_acc'] != 'Unknown']
    val_accs = [info['val_acc'] for info in checkpoints_info if info is not None and info['val_acc'] != 'Unknown']
    
    if not train_losses or not val_losses:
        print("Not enough data to plot training curves")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracies
    axes[1].plot(epochs, train_accs, 'b-', label='Train Accuracy')
    axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig


def find_best_checkpoint(checkpoints_info):
    """Find the checkpoint with the highest validation accuracy."""
    best_idx = -1
    best_acc = -1
    
    for i, info in enumerate(checkpoints_info):
        if info is None:
            continue
            
        if info['val_acc'] != 'Unknown' and info['val_acc'] > best_acc:
            best_acc = info['val_acc']
            best_idx = i
    
    if best_idx >= 0:
        return checkpoints_info[best_idx], best_idx
    else:
        return None, -1


def analyze_tuning_results(tuning_dir):
    """Analyze hyperparameter tuning results if available."""
    if not os.path.exists(tuning_dir):
        return None
    
    results_file = os.path.join(tuning_dir, "all_results.pkl")
    if not os.path.exists(results_file):
        return None
    
    try:
        import pickle
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        # Sort by validation accuracy
        results.sort(key=lambda x: x['best_val_acc'], reverse=True)
        
        # Return top configurations
        return results[:5]  # Return top 5 configurations
    except Exception as e:
        print(f"Error analyzing tuning results: {str(e)}")
        return None


def main(args):
    checkpoint_dir = args.checkpoint_dir
    config_file = args.config_file
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return
    
    # Load configuration
    config = None
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print("Loaded model configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            print()
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
    
    # Find all checkpoint files
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Analyze each checkpoint
    checkpoints_info = []
    
    print("Analyzing checkpoints...")
    for cp_file in checkpoint_files:
        info = get_checkpoint_info(cp_file)
        checkpoints_info.append(info)
    
    # Display checkpoint information
    table_data = []
    for i, (cp_file, info) in enumerate(zip(checkpoint_files, checkpoints_info)):
        if info is None:
            row = [i+1, os.path.basename(cp_file), "Error loading checkpoint", "", "", "", "", ""]
        else:
            row = [
                i+1,
                os.path.basename(cp_file),
                info['epoch'],
                f"{info['train_loss']:.4f}" if info['train_loss'] != 'Unknown' else 'Unknown',
                f"{info['train_acc']:.2f}%" if info['train_acc'] != 'Unknown' else 'Unknown',
                f"{info['val_loss']:.4f}" if info['val_loss'] != 'Unknown' else 'Unknown',
                f"{info['val_acc']:.2f}%" if info['val_acc'] != 'Unknown' else 'Unknown',
                info['modified_time']
            ]
        table_data.append(row)
    
    print("\nCheckpoint Summary:")
    print(tabulate(
        table_data, 
        headers=["No.", "Checkpoint", "Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Modified Time"],
        tablefmt="grid"
    ))
    
    # Find the best checkpoint
    best_info, best_idx = find_best_checkpoint(checkpoints_info)
    if best_info:
        print("\nBest checkpoint:")
        print(f"  File: {os.path.basename(checkpoint_files[best_idx])}")
        print(f"  Epoch: {best_info['epoch']}")
        print(f"  Validation Accuracy: {best_info['val_acc']:.2f}%")
        print(f"  Validation Loss: {best_info['val_loss']:.4f}")
        
        # Command to resume from best checkpoint
        output_dir = os.path.dirname(os.path.dirname(checkpoint_dir))
        print("\nTo resume from this checkpoint, use:")
        print(f"  python run_sincnet_updated.py --dataset_path /path/to/dataset --output_path {output_dir} --resume_training --start_epoch {best_info['epoch']}")
    
    # Plot training curves
    fig = plot_checkpoint_metrics(checkpoints_info)
    if fig:
        # Save the figure
        plot_path = os.path.join(os.path.dirname(checkpoint_dir), "checkpoint_analysis.png")
        fig.savefig(plot_path)
        print(f"\nTraining curves saved to {plot_path}")
        plt.show()
    
    # Analyze tuning results if available
    tuning_dir = os.path.join(os.path.dirname(checkpoint_dir), "tuning_results")
    top_configs = analyze_tuning_results(tuning_dir)
    
    if top_configs:
        print("\nTop Hyperparameter Configurations from Tuning:")
        for i, config in enumerate(top_configs):
            print(f"\nConfiguration #{i+1} - Val Acc: {config['best_val_acc']:.2f}%, Val Loss: {config['best_val_loss']:.4f}")
            for key, value in config['config'].items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SincNet Checkpoint Analyzer")
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing checkpoint files')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to the model configuration file (config.json)')
    
    args = parser.parse_args()
    main(args)