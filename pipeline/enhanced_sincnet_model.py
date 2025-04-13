import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torchaudio
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd
import glob
import pickle
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#----------------------------------------------------------------------
# Enhanced SincConv Implementation
#----------------------------------------------------------------------
class SincConv(nn.Module):
    """Sinc-based convolution for raw audio processing"""

    def __init__(self, out_channels, kernel_size, sample_rate=16000,
                 min_low_hz=50, min_band_hz=50):
        super(SincConv, self).__init__()

        # Make sure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Initialize filter cutoff parameters (in Hz)
        # low_hz: linearly spaced starting from min_low_hz
        low_hz = np.linspace(min_low_hz, sample_rate/2 - min_band_hz, out_channels)
        self.low_hz_ = nn.Parameter(torch.Tensor(low_hz))

        # Each filter's bandwidth starts at a minimum value
        self.band_hz_ = nn.Parameter(torch.Tensor(np.full((out_channels,), min_band_hz)))

        # Create a time axis "n" for the filter - symmetric around 0
        n = torch.arange(0, self.kernel_size) - (self.kernel_size - 1) / 2
        self.register_buffer('n_', n)

        # Precompute the Hamming window
        self.register_buffer('window_', torch.hamming_window(self.kernel_size))

    def forward(self, x):
        # Enforce that the low cutoff is at least min_low_hz and positive via abs()
        low = self.min_low_hz + torch.abs(self.low_hz_)

        # Ensure high cutoff = low + min_band + abs(band_param) and clamp to Nyquist (sample_rate/2)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                           self.min_low_hz, self.sample_rate/2)

        n = self.n_.to(x.device)  # time axis
        window = self.window_.to(x.device)

        filters = []
        for i in range(self.out_channels):
            low_i = low[i]
            high_i = high[i]
            t = n / self.sample_rate  # time vector in seconds

            # Compute the left and right parts of the band-pass using the sinc function.
            # sinc(x) is defined as sin(x)/x (with special handling at 0)
            low_pass1 = 2 * low_i * self.sinc(2 * np.pi * low_i * t)
            low_pass2 = 2 * high_i * self.sinc(2 * np.pi * high_i * t)
            band_pass = low_pass2 - low_pass1

            # Apply window and normalize the filter
            band_pass = band_pass * window
            band_pass = band_pass / (band_pass.abs().sum() + 1e-8)
            filters.append(band_pass)

        # Stack filters to shape (out_channels, 1, kernel_size)
        filters = torch.stack(filters).view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size//2)

    def sinc(self, x):
        # Numerically stable sinc implementation
        return torch.where(x == 0, torch.ones_like(x), torch.sin(x)/x)


#----------------------------------------------------------------------
# Enhanced SincNet Architecture
#----------------------------------------------------------------------
class EnhancedSincNet(nn.Module):
    """Enhanced SincNet model for speaker verification."""

    def __init__(self,
                 input_channels=1,
                 sinc_filters=80,
                 sinc_kernel_size=251,
                 sample_rate=16000,
                 conv1_channels=64,
                 conv2_channels=128,
                 conv3_channels=128,
                 feature_dim=1024,
                 emb_dim=512,
                 n_classes=None,
                 dropout_rate=0.5,
                 use_attention=True):
        super(EnhancedSincNet, self).__init__()

        # Save configuration for later
        self.config = {
            'input_channels': input_channels,
            'sinc_filters': sinc_filters,
            'sinc_kernel_size': sinc_kernel_size,
            'sample_rate': sample_rate,
            'conv1_channels': conv1_channels,
            'conv2_channels': conv2_channels,
            'conv3_channels': conv3_channels,
            'feature_dim': feature_dim,
            'emb_dim': emb_dim,
            'dropout_rate': dropout_rate,
            'use_attention': use_attention
        }

        #----------------------------------
        # SincNet layer
        #----------------------------------
        self.sinc_conv = SincConv(
            out_channels=sinc_filters,
            kernel_size=sinc_kernel_size,
            sample_rate=sample_rate,
            min_low_hz=50,
            min_band_hz=50
        )

        # Layer normalization and activation for SincNet output
        self.bn_sinc = nn.BatchNorm1d(sinc_filters)

        #----------------------------------
        # First convolutional block
        #----------------------------------
        self.conv1 = nn.Conv1d(sinc_filters, conv1_channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(conv1_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate/2)  # Lower dropout for earlier layers

        #----------------------------------
        # Second convolutional block
        #----------------------------------
        self.conv2 = nn.Conv1d(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(conv2_channels)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate/2)

        #----------------------------------
        # Third convolutional block
        #----------------------------------
        self.conv3 = nn.Conv1d(conv2_channels, conv3_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(conv3_channels)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(dropout_rate/2)

        #----------------------------------
        # Self-attention module (optional)
        #----------------------------------
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(conv3_channels)

        #----------------------------------
        # Adaptive pooling to handle variable length
        #----------------------------------
        self.adaptive_pool = nn.AdaptiveAvgPool1d(128)

        #----------------------------------
        # Fully connected layers
        #----------------------------------
        self.fc1 = nn.Linear(conv3_channels * 128, feature_dim)
        self.bn_fc1 = nn.BatchNorm1d(feature_dim)
        self.dropout_fc1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(feature_dim, emb_dim)
        self.bn_fc2 = nn.BatchNorm1d(emb_dim)
        self.dropout_fc2 = nn.Dropout(dropout_rate)

        #----------------------------------
        # Classification layer (optional)
        #----------------------------------
        self.classifier = nn.Linear(emb_dim, n_classes) if n_classes else None

        # Non-linearity to use throughout the network
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Forward pass through the SincNet model

        Args:
            x: Raw waveform of shape (batch, 1, samples)

        Returns:
            Embeddings and class probabilities (if training)
        """
        #----------------------------------
        # SincNet layer
        #----------------------------------
        x = self.sinc_conv(x)
        x = self.bn_sinc(x)
        x = self.activation(x)

        #----------------------------------
        # Convolutional blocks
        #----------------------------------
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        #----------------------------------
        # Self-attention (optional)
        #----------------------------------
        if self.use_attention:
            x = self.attention(x)

        #----------------------------------
        # Adaptive pooling
        #----------------------------------
        x = self.adaptive_pool(x)

        #----------------------------------
        # Flatten and fully connected layers
        #----------------------------------
        x = x.view(x.size(0), -1)  # Flatten

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.activation(x)
        x = self.dropout_fc1(x)

        x = self.fc2(x)
        embeddings = self.bn_fc2(x)  # Final embeddings
        x = self.activation(embeddings)
        x = self.dropout_fc2(x)

        #----------------------------------
        # Classification (if needed)
        #----------------------------------
        if self.classifier is not None:
            logits = self.classifier(x)
            return logits, embeddings

        return embeddings


#----------------------------------------------------------------------
# Self-Attention Module
#----------------------------------------------------------------------
class SelfAttention(nn.Module):
    """Self-attention mechanism for sequence data."""

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        # Query, Key, Value projections
        self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        # Scaling factor for dot product
        self.scale = (in_channels // 8) ** -0.5

        # Final output projection
        self.out_proj = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x shape: [batch, channels, seq_len]
        batch_size, channels, seq_len = x.size()

        # Query, Key, Value projections
        # Shape: [batch, channels//8, seq_len]
        q = self.query(x)
        k = self.key(x)
        # Shape: [batch, channels, seq_len]
        v = self.value(x)

        # Reshape for matrix multiplication
        # Shape: [batch, seq_len, channels//8]
        q = q.permute(0, 2, 1)
        # Shape: [batch, channels//8, seq_len]
        k = k
        # Shape: [batch, seq_len, channels]
        v = v.permute(0, 2, 1)

        # Compute attention scores
        # Shape: [batch, seq_len, seq_len]
        scores = torch.matmul(q, k) * self.scale

        # Apply softmax to get attention weights
        # Shape: [batch, seq_len, seq_len]
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to value
        # Shape: [batch, seq_len, channels]
        output = torch.matmul(attn_weights, v)

        # Reshape back to original format
        # Shape: [batch, channels, seq_len]
        output = output.permute(0, 2, 1)

        # Final projection
        output = self.out_proj(output)

        # Residual connection
        return output + x


#----------------------------------------------------------------------
# Dataset Class for Raw Audio
#----------------------------------------------------------------------
class RawAudioDataset(Dataset):
    def __init__(self, root_dir, max_samples_per_speaker=None, min_samples_per_speaker=2, segment_length=3):
        """Initialize the VoxCeleb raw audio dataset.

        Args:
            root_dir (str): Root directory of the dataset.
            max_samples_per_speaker (int, optional): Maximum number of samples per speaker.
            min_samples_per_speaker (int): Minimum number of samples per speaker (for train/val split).
            segment_length (float): Length of audio segments in seconds.
        """
        self.root_dir = root_dir
        self.segment_length = segment_length
        self.sample_rate = 16000
        self.segment_samples = int(segment_length * self.sample_rate)

        # Find all speaker IDs
        all_speaker_dirs = sorted([d for d in os.listdir(root_dir) if d.startswith('id')])

        # Find all wav files for each speaker
        self.files = []
        self.labels = []

        print("Scanning dataset...")
        speaker_file_counts = {}

        # First, count files per speaker
        for speaker_id in tqdm(all_speaker_dirs):
            speaker_dir = os.path.join(root_dir, speaker_id)
            session_dirs = os.listdir(speaker_dir)

            speaker_files = []
            for session_id in session_dirs:
                session_dir = os.path.join(speaker_dir, session_id)
                wav_files = glob.glob(os.path.join(session_dir, "*.wav"))
                speaker_files.extend(wav_files)

            speaker_file_counts[speaker_id] = len(speaker_files)

        # Filter speakers with too few samples
        valid_speakers = [spk for spk, count in speaker_file_counts.items()
                          if count >= min_samples_per_speaker]

        print(
            f"Found {len(valid_speakers)}/{len(all_speaker_dirs)} speakers with at least {min_samples_per_speaker} samples")
        self.speaker_ids = valid_speakers

        # Create a mapping from speaker ID to integer label
        self.id_to_label = {speaker_id: i for i, speaker_id in enumerate(self.speaker_ids)}
        self.label_to_id = {i: speaker_id for i, speaker_id in enumerate(self.speaker_ids)}

        # Now collect files only from valid speakers
        for speaker_id in tqdm(self.speaker_ids):
            speaker_dir = os.path.join(root_dir, speaker_id)
            session_dirs = os.listdir(speaker_dir)

            speaker_files = []
            for session_id in session_dirs:
                session_dir = os.path.join(speaker_dir, session_id)
                wav_files = glob.glob(os.path.join(session_dir, "*.wav"))
                speaker_files.extend(wav_files)

            # Limit samples per speaker if specified
            if max_samples_per_speaker is not None and len(speaker_files) > max_samples_per_speaker:
                speaker_files = random.sample(speaker_files, max_samples_per_speaker)

            self.files.extend(speaker_files)
            self.labels.extend([self.id_to_label[speaker_id]] * len(speaker_files))

        print(f"Found {len(self.files)} files from {len(self.speaker_ids)} speakers")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]

        # Load raw audio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        # Normalize audio
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        # Handle varying lengths (crop or pad)
        if waveform.shape[1] >= self.segment_samples:
            # Randomly crop to segment_samples
            max_audio_start = waveform.shape[1] - self.segment_samples
            audio_start = random.randint(0, max_audio_start)
            waveform = waveform[:, audio_start:audio_start + self.segment_samples]
        else:
            # Pad with zeros
            padding = self.segment_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))

        return waveform, label


#----------------------------------------------------------------------
# Training Functions
#----------------------------------------------------------------------
def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for waveforms, labels in progress_bar:
        waveforms, labels = waveforms.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs, _ = model(waveforms)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            'loss': running_loss / (progress_bar.n + 1),
            'acc': 100. * correct / total
        })

    if scheduler:
        scheduler.step()

    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for waveforms, labels in progress_bar:
            waveforms, labels = waveforms.to(device), labels.to(device)

            # Forward pass
            outputs, _ = model(waveforms)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'acc': 100. * correct / total
            })

    return running_loss / len(val_loader), 100. * correct / total


#----------------------------------------------------------------------
# Plot Filters Function
#----------------------------------------------------------------------
def plot_sinc_filters(model, n_filters=20, output_path=None):
    """Plot the learned sinc filters."""
    if not hasattr(model, 'sinc_conv'):
        print("Model does not have sinc convolution layer")
        return

    # Get the filters
    low_hz = model.sinc_conv.min_low_hz + torch.abs(model.sinc_conv.low_hz_)
    band_hz = model.sinc_conv.min_band_hz + torch.abs(model.sinc_conv.band_hz_)
    high_hz = low_hz + band_hz

    low_hz = low_hz.detach().cpu().numpy()
    high_hz = high_hz.detach().cpu().numpy()

    plt.figure(figsize=(12, 8))

    # Plot filter frequencies
    for i in range(min(n_filters, len(low_hz))):
        plt.plot([low_hz[i], high_hz[i]], [i, i], 'b-')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter index')
    plt.title('SincNet learned filter bands')
    plt.grid(True)

    if output_path:
        plt.savefig(os.path.join(output_path, 'sincnet_filters.png'))
    plt.show()


#----------------------------------------------------------------------
# Main Training Script
#----------------------------------------------------------------------
def main(args):
    """Main training function with improved checkpointing and resumption."""

    # Set paths
    dataset_path = args.dataset_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    # Create a log file for tracking overall progress
    log_file = os.path.join(output_path, "training_log.txt")

    # Log start of execution
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Starting SincNet execution at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Args: {args}\n")

    # Set parameters
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    max_samples_per_speaker = args.max_samples_per_speaker
    min_samples_per_speaker = args.min_samples_per_speaker
    segment_length = args.segment_length

    # Decide whether to tune model
    if args.tune_model:
        with open(log_file, 'a') as f:
            f.write(f"Starting hyperparameter tuning with {args.n_tune_trials} trials\n")

        # Use the resume_tuning function for resumable hyperparameter tuning
        from tuning_resumption import resume_tuning

        # Set resume flag based on args.resume_tuning
        best_config = resume_tuning(
            dataset_path=dataset_path,
            output_path=output_path,
            n_trials=args.n_tune_trials,
            resume=args.resume_tuning
        )

        with open(log_file, 'a') as f:
            f.write(f"Tuning complete. Best config: {best_config}\n")
    else:
        # Use default or specified configuration
        best_config = {
            'sinc_filters': args.sinc_filters,
            'sinc_kernel_size': args.sinc_kernel_size,
            'conv1_channels': args.conv1_channels,
            'conv2_channels': args.conv2_channels,
            'conv3_channels': args.conv3_channels,
            'feature_dim': args.feature_dim,
            'emb_dim': args.emb_dim,
            'dropout_rate': args.dropout_rate,
            'use_attention': args.use_attention,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }

    # Save configuration
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        import json
        json.dump(best_config, f, indent=4)

    # Set number of workers to 2 to avoid system warnings
    num_workers = args.num_workers

    # Create datasets
    with open(log_file, 'a') as f:
        f.write(f"Loading dataset from {dataset_path}\n")

    dataset = RawAudioDataset(
        dataset_path,
        max_samples_per_speaker=max_samples_per_speaker,
        min_samples_per_speaker=min_samples_per_speaker,
        segment_length=segment_length
    )

    # Split into train and validation
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        stratify=dataset.labels,
        random_state=42
    )

    # Log dataset information
    with open(log_file, 'a') as f:
        f.write(f"Dataset loaded with {len(dataset)} samples, {len(dataset.speaker_ids)} speakers\n")
        f.write(f"Train set: {len(train_indices)} samples, Val set: {len(val_indices)} samples\n")

    # Create data loaders
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=best_config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=best_config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        drop_last=True
    )

    # Create model using best configuration
    model = EnhancedSincNet(
        sinc_filters=best_config['sinc_filters'],
        sinc_kernel_size=best_config['sinc_kernel_size'],
        conv1_channels=best_config['conv1_channels'],
        conv2_channels=best_config['conv2_channels'],
        conv3_channels=best_config['conv3_channels'],
        feature_dim=best_config['feature_dim'],
        emb_dim=best_config['emb_dim'],
        dropout_rate=best_config['dropout_rate'],
        use_attention=best_config['use_attention'],
        n_classes=len(dataset.speaker_ids)
    ).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open(log_file, 'a') as f:
        f.write(f"Model created with {total_params:,} total parameters, {trainable_params:,} trainable\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=best_config['learning_rate'])

    # Learning rate scheduler - choose between different strategies
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    else:
        scheduler = None

    # Initialize variables
    best_acc = 0.0
    start_epoch = 0
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Resume from checkpoint if specified
    if args.resume_training:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{args.start_epoch}.pth")
        if os.path.exists(checkpoint_path):
            with open(log_file, 'a') as f:
                f.write(f"Resuming from checkpoint: {checkpoint_path}\n")

            print(f"Loading checkpoint from epoch {args.start_epoch}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_acc = checkpoint.get('best_acc', 0.0)
            start_epoch = args.start_epoch

            # Load history if available
            if 'history' in checkpoint:
                history = checkpoint['history']

            print(f"Resuming from epoch {start_epoch} with best accuracy: {best_acc:.2f}%")
        else:
            with open(log_file, 'a') as f:
                f.write(f"Checkpoint not found: {checkpoint_path}, starting from scratch\n")

            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
            start_epoch = 0

    # Skip training if extract_only is True
    if args.extract_only:
        with open(log_file, 'a') as f:
            f.write("Skipping training, proceeding to embedding extraction\n")

        print("Skipping training, proceeding to embedding extraction...")
    else:
        # Training loop
        with open(log_file, 'a') as f:
            f.write(f"Starting training for {num_epochs} epochs (from epoch {start_epoch})\n")

        for epoch in range(start_epoch, num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_start_time = time.time()

            try:
                # Train
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device
                )

                # Validate
                val_loss, val_acc = validate(model, val_loader, criterion, device)

                # Update history
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                # Log epoch results
                epoch_time = time.time() - epoch_start_time

                with open(log_file, 'a') as f:
                    f.write(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Time={epoch_time:.1f}s\n")

                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                print(f"Epoch time: {epoch_time:.1f} seconds")

                # Update plateau scheduler if used
                if args.scheduler == 'plateau' and scheduler:
                    scheduler.step(val_loss)
                # Update cosine scheduler if used
                elif args.scheduler == 'cosine' and scheduler:
                    scheduler.step()

                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_acc': best_acc,
                    'history': history
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    model_save_path = os.path.join(output_path, "best_model.pth")
                    torch.save(model.state_dict(), model_save_path)

                    with open(log_file, 'a') as f:
                        f.write(f"New best model saved with acc: {best_acc:.2f}%\n")

                    print(f"New best model saved to {model_save_path}")

            except Exception as e:
                with open(log_file, 'a') as f:
                    f.write(f"Error during epoch {epoch + 1}: {str(e)}\n")

                print(f"Error during epoch {epoch + 1}: {str(e)}")
                # Save checkpoint anyway so we don't lose progress
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}_error.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_acc': best_acc,
                    'history': history
                }, checkpoint_path)

                with open(log_file, 'a') as f:
                    f.write(f"Emergency checkpoint saved to {checkpoint_path}\n")

                print(f"Emergency checkpoint saved to {checkpoint_path}")
                raise e

        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over epochs')

        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy over epochs')

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'training_history.png'))

        with open(log_file, 'a') as f:
            f.write(f"Training completed. Best accuracy: {best_acc:.2f}%\n")

    # Load best model for embedding extraction
    best_model_path = os.path.join(output_path, "best_model.pth")
    if os.path.exists(best_model_path):
        with open(log_file, 'a') as f:
            f.write(f"Loading best model for embedding extraction\n")

        print("Loading best model for embedding extraction")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        with open(log_file, 'a') as f:
            f.write("No best model found, using current model for embedding extraction\n")

        print("No best model found, using current model for embedding extraction")

    # Plot the learned sinc filters
    plot_sinc_filters(model, output_path=output_path)

    # Extract embeddings phase
    with open(log_file, 'a') as f:
        f.write("Starting embedding extraction phase\n")

    print("Starting embedding extraction phase")

    # Create a new dataset without the max_samples constraint for embedding extraction
    extract_dataset = RawAudioDataset(
        dataset_path,
        min_samples_per_speaker=min_samples_per_speaker,
        segment_length=segment_length
    )

    # If resuming extraction, check for existing embeddings
    embeddings_save_path = os.path.join(output_path, "sincnet_embeddings.pkl")
    if args.resume_extraction and os.path.exists(embeddings_save_path):
        with open(log_file, 'a') as f:
            f.write(f"Found existing embeddings at {embeddings_save_path}, resuming extraction\n")

        print(f"Found existing embeddings at {embeddings_save_path}, resuming extraction")

        with open(embeddings_save_path, 'rb') as f:
            existing_data = pickle.load(f)
            all_embeddings = existing_data['embeddings']
            all_labels = existing_data['labels']
            processed_files = existing_data['files']

            # Find remaining files
            remaining_files = [f for f in extract_dataset.files if f not in processed_files]

            # If all files processed, skip extraction
            if not remaining_files:
                with open(log_file, 'a') as f:
                    f.write("All files already processed for embeddings, skipping extraction\n")

                print("All files already processed for embeddings, skipping extraction")
                return {
                    'model': model,
                    'best_acc': best_acc,
                    'embeddings_path': embeddings_save_path
                }

            # Create a new dataset with only the remaining files
            print(f"Resuming extraction: {len(processed_files)} files already processed, {len(remaining_files)} files remaining")

            # Create indices for the remaining files
            remaining_indices = [i for i, f in enumerate(extract_dataset.files) if f in remaining_files]
            extract_subset = torch.utils.data.Subset(extract_dataset, remaining_indices)

            # Initialize with existing embeddings
            processed_samples = len(processed_files)
    else:
        # Start fresh extraction
        all_embeddings = []
        all_labels = []
        processed_samples = 0
        extract_subset = extract_dataset

    # Extract embeddings
    new_embeddings = []
    new_labels = []
    new_files = []

    extract_loader = DataLoader(
        extract_subset,
        batch_size=best_config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        drop_last=True
    )

    with torch.no_grad():
        for waveforms, labels in tqdm(extract_loader, desc="Extracting embeddings"):
            waveforms = waveforms.to(device)
            batch_size = len(labels)

            # Forward pass - handle both tuple or single output
            model_output = model(waveforms)
            if isinstance(model_output, tuple):
                _, embeddings = model_output
            else:
                embeddings = model_output

            new_embeddings.append(embeddings.cpu().numpy())
            new_labels.append(labels.numpy())
            processed_samples += batch_size

    # Process new embeddings
    if new_embeddings:
        new_embeddings = np.vstack(new_embeddings)
        new_labels = np.concatenate(new_labels)

        # If resuming, combine with existing embeddings
        if args.resume_extraction and os.path.exists(embeddings_save_path):
            all_embeddings = np.vstack([all_embeddings, new_embeddings])
            all_labels = np.concatenate([all_labels, new_labels])
            file_paths = processed_files + [extract_dataset.files[i] for i in remaining_indices[:len(new_embeddings)]]
        else:
            all_embeddings = new_embeddings
            all_labels = new_labels
            file_paths = extract_dataset.files[:len(new_embeddings)]

        # Make sure dimensions match
        if len(all_embeddings) != len(file_paths):
            with open(log_file, 'a') as f:
                f.write(f"Warning: Found {len(all_embeddings)} embeddings but {len(file_paths)} files\n")

            print(f"Warning: Found {len(all_embeddings)} embeddings but {len(file_paths)} files")
            min_len = min(len(all_embeddings), len(file_paths))
            all_embeddings = all_embeddings[:min_len]
            all_labels = all_labels[:min_len]
            file_paths = file_paths[:min_len]

        # Save embeddings (with backup to avoid corruption)
        with open(log_file, 'a') as f:
            f.write(f"Saving {len(all_embeddings)} embeddings to {embeddings_save_path}\n")

        # First save to a temporary file
        temp_embeddings_path = os.path.join(output_path, "sincnet_embeddings.temp.pkl")
        with open(temp_embeddings_path, 'wb') as f:
            pickle.dump({
                'embeddings': all_embeddings,
                'labels': all_labels,
                'files': file_paths,
                'id_to_label': extract_dataset.id_to_label,
                'label_to_id': extract_dataset.label_to_id
            }, f)

        # If successful, move to final path
        os.replace(temp_embeddings_path, embeddings_save_path)

        print(f"Embeddings saved to {embeddings_save_path}")

        # Also save in numpy format for easier loading
        np.save(os.path.join(output_path, "sincnet_embeddings.npy"), all_embeddings)
        np.save(os.path.join(output_path, "sincnet_labels.npy"), all_labels)

        # Save mapping between files and embeddings
        with open(os.path.join(output_path, "sincnet_file_mapping.pkl"), 'wb') as f:
            pickle.dump({
                'files': file_paths,
                'id_to_label': extract_dataset.id_to_label,
                'label_to_id': extract_dataset.label_to_id
            }, f)
    else:
        with open(log_file, 'a') as f:
            f.write("No new embeddings extracted\n")

        print("No new embeddings extracted")

    # Log completion
    with open(log_file, 'a') as f:
        f.write(f"SincNet execution completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n")

    return {
        'model': model,
        'best_acc': best_acc,
        'embeddings_path': embeddings_save_path
    }