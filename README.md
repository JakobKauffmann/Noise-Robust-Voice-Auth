# Noise-Robust-Voice-Auth
## Mac Setup Guide for Voice Authentication Project

This guide provides instructions for setting up the voice authentication project on a Mac, including both Intel and Apple Silicon (M1/M2/M3) Macs.

### Setting up the Environment

#### 1. Install Miniconda

If you haven't already, install Miniconda:
```bash
# Download Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh  # For Apple Silicon
# OR
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh  # For Intel Macs

# Install
bash Miniconda3-latest-*.sh
```

#### 2. Create the Environment

Use the Mac-specific environment file:
```bash
conda env create -f environment-mac.yml
```

#### 3. Activate the Environment

```bash
conda activate voice-auth-mac
```

### GPU Acceleration on Apple Silicon Macs

For Apple Silicon (M1/M2/M3) Macs, you can leverage the built-in GPU using PyTorch's MPS (Metal Performance Shaders) backend.

Add this code at the beginning of your Python scripts:

```python
import torch

# Check if MPS is available (Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")
```

### Code Modifications for Mac Compatibility

#### 1. Update Model Initialization

Wherever models are initialized, make sure to move them to the correct device:

```python
model = VoiceAuthentication(
    sincnet=sincnet,
    spectrogram_cnn=spectrogram_cnn,
    fusion=fusion,
    noise_suppression=noise_suppression,
    sample_rate=args.sample_rate,
    embedding_dim=args.feature_dim
)
model = model.to(device)  # Move to appropriate device
```

#### 2. Handling MPS Limitations

Some operations might not be supported on MPS yet. Add fallbacks:

```python
def process_tensor(tensor):
    try:
        # Try to process on current device
        result = some_operation(tensor)
    except RuntimeError as e:
        if "is not implemented for" in str(e) and device.type == "mps":
            # Fallback to CPU for unsupported operations
            result = some_operation(tensor.cpu()).to(device)
        else:
            raise
    return result
```

### Performance Optimization

For Mac users, especially when using CPU-only:

1. **Reduce batch size** to avoid memory issues
2. **Use smaller models** during development
3. **Pre-calculate features** when possible to avoid repeated computation

### Remote Development for Training

For full model training, consider using remote resources:

1. **Google Colab**: Free GPU access
2. **Kaggle Kernels**: 30 hours/week of GPU access
3. **University Computing Resources**: Check if your university offers HPC access

### Testing Your Setup

Run a quick test to verify your environment:

```python
import torch
import torchaudio
import librosa
import numpy as np

# Print versions
print(f"PyTorch: {torch.__version__}")
print(f"Torchaudio: {torchaudio.__version__}")
print(f"Librosa: {librosa.__version__}")

# Check device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Apple Silicon GPU) is available!")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Create a random tensor and move to device
x = torch.randn(1, 16000).to(device)
print(f"Tensor device: {x.device}")
```
