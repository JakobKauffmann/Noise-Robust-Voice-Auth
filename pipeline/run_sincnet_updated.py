#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run SincNet training for Voice Authentication with complete resumption capabilities
Example usage:
    # Start new training:
    python run_sincnet_updated.py --dataset_path /path/to/dataset --output_path /path/to/output
    
    # Resume training from epoch 10:
    python run_sincnet_updated.py --dataset_path /path/to/dataset --output_path /path/to/output --resume_training --start_epoch 10
    
    # Start hyperparameter tuning:
    python run_sincnet_updated.py --dataset_path /path/to/dataset --output_path /path/to/output --tune_model
    
    # Resume hyperparameter tuning:
    python run_sincnet_updated.py --dataset_path /path/to/dataset --output_path /path/to/output --tune_model --resume_tuning
    
    # Only extract embeddings using best model:
    python run_sincnet_updated.py --dataset_path /path/to/dataset --output_path /path/to/output --extract_only
    
    # Resume embeddings extraction:
    python run_sincnet_updated.py --dataset_path /path/to/dataset --output_path /path/to/output --extract_only --resume_extraction
"""

import os
import argparse
import time
from enhanced_sincnet_model import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SincNet for Voice Authentication with Complete Resumption")
    
    # Dataset and output paths
    parser.add_argument('--dataset_path', type=str, 
                        default="/content/drive/Shareddrives/VoxCeleb1/Dev_Filtered/wav",
                        help='Path to the dataset directory')
    parser.add_argument('--output_path', type=str, 
                        default="/content/drive/Shareddrives/VoxCeleb1/Features",
                        help='Path to save model and results')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20, 
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'plateau', 'none'], default='plateau',
                        help='Learning rate scheduler type')
    parser.add_argument('--num_workers', type=int, default=2, 
                        help='Number of worker threads for data loading')
    parser.add_argument('--max_samples_per_speaker', type=int, default=100, 
                        help='Maximum number of samples per speaker')
    parser.add_argument('--min_samples_per_speaker', type=int, default=2, 
                        help='Minimum number of samples per speaker')
    parser.add_argument('--segment_length', type=float, default=3.0, 
                        help='Audio segment length in seconds')
    
    # Model parameters
    parser.add_argument('--sinc_filters', type=int, default=80, 
                        help='Number of SincNet filters')
    parser.add_argument('--sinc_kernel_size', type=int, default=251, 
                        help='SincNet kernel size')
    parser.add_argument('--conv1_channels', type=int, default=64, 
                        help='Number of channels in first conv layer')
    parser.add_argument('--conv2_channels', type=int, default=128, 
                        help='Number of channels in second conv layer')
    parser.add_argument('--conv3_channels', type=int, default=128, 
                        help='Number of channels in third conv layer')
    parser.add_argument('--feature_dim', type=int, default=1024, 
                        help='Dimension of first FC layer')
    parser.add_argument('--emb_dim', type=int, default=512, 
                        help='Dimension of embedding')
    parser.add_argument('--dropout_rate', type=float, default=0.5, 
                        help='Dropout rate')
    parser.add_argument('--use_attention', action='store_true', 
                        help='Use self-attention module')
    
    # Control flow and resumption parameters
    parser.add_argument('--resume_training', action='store_true', 
                        help='Resume training from checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0, 
                        help='Epoch to start/resume from')
    parser.add_argument('--extract_only', action='store_true', 
                        help='Skip training and only extract embeddings')
    parser.add_argument('--tune_model', action='store_true', 
                        help='Perform hyperparameter tuning before training')
    parser.add_argument('--n_tune_trials', type=int, default=10, 
                        help='Number of hyperparameter configurations to try')
    parser.add_argument('--resume_tuning', action='store_true',
                        help='Resume hyperparameter tuning from previous state')
    parser.add_argument('--resume_extraction', action='store_true',
                        help='Resume embedding extraction from previous state')
    
    args = parser.parse_args()
    
    # Print execution start info
    print(f"{'='*60}")
    print(f"Starting SincNet execution at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Check for resumption flags and print appropriate message
    if args.resume_training:
        print(f"Resuming training from epoch {args.start_epoch}")
    elif args.tune_model and args.resume_tuning:
        print("Resuming hyperparameter tuning from previous state")
    elif args.extract_only and args.resume_extraction:
        print("Resuming embedding extraction from previous state")
    else:
        if args.tune_model:
            print(f"Starting new hyperparameter tuning with {args.n_tune_trials} trials")
        elif args.extract_only:
            print("Starting new embedding extraction (skipping training)")
        else:
            print(f"Starting new training for {args.num_epochs} epochs")
    
    # Run the main function
    try:
        results = main(args)
        print(f"{'='*60}")
        print("SincNet execution completed successfully!")
        print(f"Best accuracy: {results['best_acc']:.2f}%")
        print(f"Embeddings saved to: {results['embeddings_path']}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"{'='*60}")
        print(f"ERROR: SincNet execution failed with error: {str(e)}")
        print("Check the log file for details")
        print(f"{'='*60}")
        raise e