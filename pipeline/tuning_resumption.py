import os
import pickle
import glob
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

# Import from enhanced_sincnet_model.py - make sure it's in the same directory
from enhanced_sincnet_model import EnhancedSincNet, RawAudioDataset, device

def resume_tuning(dataset_path, output_path, n_trials=10, resume=True):
    """
Perform hyperparameter tuning with ability to resume interrupted tuning.

Args:
dataset_path (str): Path to dataset
output_path (str): Path to save tuning results
n_trials (int): Number of different configurations to try
resume (bool): Whether to resume from previous tuning

Returns:
dict: Best hyperparameters and model configuration
    """
    # Create tuning results directory if it doesn't exist
    tuning_dir = os.path.join(output_path, "tuning_results")
    os.makedirs(tuning_dir, exist_ok=True)

    # Create a log file for tuning progress
    log_file = os.path.join(tuning_dir, "tuning_log.txt")

    # Check if we're resuming previous tuning
    completed_trials = 0
    tried_configs = []
    results = []

    if resume:
        # Check for existing tuning log
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_lines = f.readlines()
                completed_trials = len([l for l in log_lines if l.startswith("Trial") and "finished" in l])
            print(f"Found {completed_trials} completed trials in log file")

        # Check for existing trial results
        trial_files = sorted(glob.glob(os.path.join(tuning_dir, "trial_*.pkl")))
        completed_trials = max(completed_trials, len(trial_files))

        # Load previous results
        all_results_path = os.path.join(tuning_dir, "all_results.pkl")
        if os.path.exists(all_results_path):
            with open(all_results_path, 'rb') as f:
                results = pickle.load(f)
                tried_configs = [r['config'] for r in results]
                print(f"Loaded {len(results)} previous trial results")

                # Print best result so far
                best_idx = np.argmax([r['best_val_acc'] for r in results])
                best_result = results[best_idx]
                print(f"Best trial so far: {best_idx+1}, Acc: {best_result['best_val_acc']:.2f}%, Loss: {best_result['best_val_loss']:.4f}")

    # If we've completed all trials already, just return the best
    if completed_trials >= n_trials:
        print(f"All {n_trials} trials were already completed.")
        # Find best configuration
        results.sort(key=lambda x: x['best_val_acc'], reverse=True)
        return results[0]['config']

    # Create smaller dataset for faster tuning
    tune_dataset = RawAudioDataset(
        dataset_path,
        max_samples_per_speaker=20,  # Only use 20 samples per speaker for quicker tuning
        min_samples_per_speaker=2,
        segment_length=2  # Shorter segments for faster tuning
    )

    # Split into train and validation
    train_indices, val_indices = train_test_split(
        list(range(len(tune_dataset))),
        test_size=0.2,
        stratify=tune_dataset.labels,
        random_state=42
    )

    # Hyperparameter ranges to search
    # Hyperparameter ranges to search - with safer values
    hyperparams = {
        'sinc_filters': [40, 60, 80],  # Reduced range
        'sinc_kernel_size': [101, 251],  # Avoid very large kernel sizes
        'conv1_channels': [32, 64],
        'conv2_channels': [64, 128],
        'conv3_channels': [64, 128],
        'feature_dim': [512, 1024],
        'emb_dim': [192, 256],
        'dropout_rate': [0.3, 0.5],
        'learning_rate': [3e-4, 1e-3],
        'batch_size': [32, 64],  # Avoid small batch sizes like 16
        'use_attention': [True, False]
    }

    # Log tuning start/resume
    with open(log_file, 'a') as f:
        if completed_trials == 0:
            f.write(f"=== Starting new tuning session, target: {n_trials} trials ===\n")
        else:
            f.write(f"=== Resuming tuning session from trial {completed_trials+1}, target: {n_trials} trials ===\n")

    # Start/resume trials
    for trial in range(completed_trials, n_trials):
        print(f"\n=== Trial {trial+1}/{n_trials} ===")

        # Generate a new configuration that hasn't been tried
        while True:
            config = {
                'sinc_filters': random.choice(hyperparams['sinc_filters']),
                'sinc_kernel_size': random.choice(hyperparams['sinc_kernel_size']),
                'conv1_channels': random.choice(hyperparams['conv1_channels']),
                'conv2_channels': random.choice(hyperparams['conv2_channels']),
                'conv3_channels': random.choice(hyperparams['conv3_channels']),
                'feature_dim': random.choice(hyperparams['feature_dim']),
                'emb_dim': random.choice(hyperparams['emb_dim']),
                'dropout_rate': random.choice(hyperparams['dropout_rate']),
                'learning_rate': random.choice(hyperparams['learning_rate']),
                'batch_size': random.choice(hyperparams['batch_size']),
                'use_attention': random.choice(hyperparams['use_attention'])
            }

            # Check if this configuration was already tried
            if config not in tried_configs:
                tried_configs.append(config)
                break

        # Log trial start
        with open(log_file, 'a') as f:
            f.write(f"Trial {trial+1} started with config: {config}\n")

        print("Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        try:
            # Create model with this configuration
            model = EnhancedSincNet(
                sinc_filters=config['sinc_filters'],
                sinc_kernel_size=config['sinc_kernel_size'],
                conv1_channels=config['conv1_channels'],
                conv2_channels=config['conv2_channels'],
                conv3_channels=config['conv3_channels'],
                feature_dim=config['feature_dim'],
                emb_dim=config['emb_dim'],
                dropout_rate=config['dropout_rate'],
                use_attention=config['use_attention'],
                n_classes=len(tune_dataset.speaker_ids)
            ).to(device)

            # In tuning_resumption.py, modify the following part in the training loop
            # (inside the try block where you create the model and dataloaders):

            # First, add this line after creating the model:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # This helps with small batch sizes

            # Then modify the batch_norm layers to handle single samples:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.momentum = 0.1
                    module.eps = 1e-05
                    # Add this to handle the error:
                    module.track_running_stats = False  # Disable running stats tracking

            # Or you could modify the DataLoader to ensure the last batch is dropped:
            train_loader = DataLoader(
                torch.utils.data.Subset(tune_dataset, train_indices),
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=2,
                drop_last=True  # Add this parameter
            )

            val_loader = DataLoader(
                torch.utils.data.Subset(tune_dataset, val_indices),
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=2,
                drop_last=True  # Add this parameter
            )

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=config['learning_rate'])
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

            # Train for a few epochs to see how it performs
            best_val_loss = float('inf')
            best_val_acc = 0.0

            for epoch in range(5):  # Just 5 epochs for tuning
                # Train
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for waveforms, labels in train_loader:
                    waveforms, labels = waveforms.to(device), labels.to(device)

                    optimizer.zero_grad()

                    # Forward pass
                    outputs, _ = model(waveforms)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    # Statistics
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                # Validation
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for waveforms, labels in val_loader:
                        waveforms, labels = waveforms.to(device), labels.to(device)

                        # Forward pass
                        outputs, _ = model(waveforms)
                        loss = criterion(outputs, labels)

                        # Statistics
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()

                train_loss = train_loss / len(train_loader)
                train_acc = 100. * train_correct / train_total
                val_loss = val_loss / len(val_loader)
                val_acc = 100. * val_correct / val_total

                print(f"Epoch {epoch+1}/5 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

                # Log epoch results
                with open(log_file, 'a') as f:
                    f.write(f"  Epoch {epoch+1}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%\n")

                # Update scheduler
                scheduler.step(val_loss)

                # Keep track of best validation metrics
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

            # Store results
            result = {
                'config': config,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc
            }
            results.append(result)

            # Save result to file
            with open(os.path.join(tuning_dir, f"trial_{trial+1}.pkl"), 'wb') as f:
                pickle.dump(result, f)

            # Log trial completion
            with open(log_file, 'a') as f:
                f.write(f"Trial {trial+1} finished - Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.2f}%\n")

            print(f"Trial {trial+1} finished - Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.2f}%")

        except Exception as e:
            # Log error
            with open(log_file, 'a') as f:
                f.write(f"Trial {trial+1} failed with error: {str(e)}\n")

            print(f"Error in trial {trial+1}: {str(e)}")

            # Still save the configuration to avoid repeating it
            with open(os.path.join(tuning_dir, f"trial_{trial+1}_failed.pkl"), 'wb') as f:
                pickle.dump({'config': config, 'error': str(e)}, f)

        # Save all results after each trial
        with open(os.path.join(tuning_dir, "all_results.pkl"), 'wb') as f:
            pickle.dump(results, f)

        # Also save as CSV for easier analysis
        results_df = pd.DataFrame([
            {**r['config'], 'val_loss': r['best_val_loss'], 'val_acc': r['best_val_acc']}
            for r in results
        ])
        results_df.to_csv(os.path.join(tuning_dir, "all_results.csv"), index=False)

    # Find best configuration
    if results:
        results.sort(key=lambda x: x['best_val_acc'], reverse=True)
        best_config = results[0]['config']

        print("\n=== Best Configuration ===")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
        print(f"Best Val Acc: {results[0]['best_val_acc']:.2f}%")
        print(f"Best Val Loss: {results[0]['best_val_loss']:.4f}")

        return best_config
    else:
        print("No successful trials were completed!")
        return None