# models/sincnet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv1d(nn.Module):
    """Sinc-based 1D convolution as in SincNet."""

    @staticmethod
    def _hz2mel(hz):
        # Static method to convert Hz to Mel scale
        if isinstance(hz, torch.Tensor):
            return 2595.0 * torch.log10(1.0 + hz / 700.0)
        else:
            return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel2hz(mel):
        # Static method to convert Mel scale to Hz
        if isinstance(mel, torch.Tensor):
            return 700.0 * (10 ** (mel / 2595.0) - 1.0)
        else:
            return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate,
        in_channels=1,
        stride=1, # Added stride (usually 1 for SincNet first layer)
        padding=0, # Added padding (usually calculated or 'same')
        dilation=1, # Added dilation
        bias=False, # SincNet typically doesn't use bias here
        groups=1, # Added groups
        min_low_hz=50,
        min_band_hz=50,
    ):
        super().__init__()

        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
            # print(f"Warning: kernel_size must be odd, changed to {kernel_size}") # Optional warning

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.in_channels = in_channels # Store in_channels
        self.groups = groups # Store groups
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Initialize filterbank parameters (cutoff frequencies) on the Mel scale
        low_hz = 30.0
        high_hz = sample_rate / 2.0 - (self.min_low_hz + self.min_band_hz)

        mel = torch.linspace(
            self._hz2mel(low_hz),
            self._hz2mel(high_hz),
            self.out_channels + 1,
        )
        hz = self._mel2hz(mel)

        # Store lower and band frequencies as learnable parameters
        self.low_hz_ = nn.Parameter(hz[:-1].view(-1, 1))
        self.band_hz_ = nn.Parameter((hz[1:] - hz[:-1]).view(-1, 1))

        # Pre-compute Hamming window and time axis for filter generation
        # Note: SincNet paper uses Hamming window, but some implementations use Hann or others.
        # Using Hamming window here.
        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )
        self.register_buffer("window_", torch.hamming_window(self.kernel_size))
        self.register_buffer("n_", 2 * math.pi * n_lin / self.sample_rate) # Pre-calculate 2*pi*t/fs

    def forward(self, waveforms):
        """
        Calculates the SincNet filter responses and performs convolution.

        Args:
            waveforms (torch.Tensor): Input tensor of shape [batch, 1, time].

        Returns:
            torch.Tensor: Output tensor of shape [batch, out_channels, time_out].
        """
        # Calculate filter cutoff frequencies, ensuring they are within valid range
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        # Calculate band frequencies (f_high - f_low)
        band = (high - low)

        # --- Filter Generation ---
        # Calculate the right half of the sinc filters (t > 0)
        # Avoid division by zero at t=0 by using a small epsilon or handling it separately
        n_ = self.n_.to(waveforms.device) # Move precomputed values to correct device
        low = low.to(waveforms.device)
        high = high.to(waveforms.device)

        # Compute sinc components for high and low cutoffs
        sinc_high = torch.sin(high * n_) / (n_ + 1e-9) # Add epsilon for stability at n=0
        sinc_low = torch.sin(low * n_) / (n_ + 1e-9)

        # Create bandpass filters by subtracting low-pass sinc from high-pass sinc
        # The filter is (sin(2*pi*f_high*t) - sin(2*pi*f_low*t)) / (2*pi*t)
        # Our n_ already includes 2*pi/fs, so this simplifies
        filters_right = (sinc_high - sinc_low) / (2. * math.pi) # Divide by 2*pi

        # Handle the center tap (t=0) separately: limit of sinc(x)/x is 1, so limit is (2*pi*f_high - 2*pi*f_low)/(2*pi) = f_high - f_low = band
        # Or simpler: sinc(0) = 1, so (2*f_high - 2*f_low) -> normalized gives band
        # Center tap value should be proportional to the bandwidth
        # filters_center = band * 2 / self.sample_rate # Normalize band value
        filters_center = band # Direct band value often works

        # Combine right half, center tap, and flipped left half
        filters = torch.cat(
            [torch.flip(filters_right, dims=[1]), filters_center, filters_right], dim=1
        ) # [out_channels, kernel_size]

        # Apply the window function (e.g., Hamming)
        window_ = self.window_.to(waveforms.device)
        filters = filters * window_

        # Normalize filters to have sum 1 (or unit energy, common practice)
        # Normalization helps stabilize training
        filters = filters / torch.sum(filters, dim=1, keepdim=True) # Normalize sum to 1

        # Reshape filters for convolution: [out_channels, 1, kernel_size]
        # Assuming in_channels=1 as per original SincNet
        filters = filters.view(self.out_channels, 1, self.kernel_size)

        # --- Convolution ---
        # Perform 1D convolution
        # Note: Adjust padding calculation if needed. 'same' padding is common.
        # If padding=0, output length will be different.
        # SincNet paper implies padding to keep length, so use calculated padding or 'same'.
        padding_val = (self.kernel_size - 1) // 2 # Common 'same' padding for odd kernel
        if self.padding == 'same':
             padding_to_use = padding_val
        else:
             padding_to_use = self.padding # Use specified padding

        output = F.conv1d(
            waveforms,
            filters,
            stride=self.stride,
            padding=padding_to_use,
            dilation=self.dilation,
            bias=None, # No bias in SincConv
            groups=self.groups,
        )

        return output


class SincNetEmbedding(nn.Module):
    """
    SincNet model for speaker embedding extraction.
    Input: (B, 1, L) waveform (L = samples, e.g., 16000 * 3 = 48000)
    Output: (B, emb_dim) speaker embedding (e.g., B, 256)
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        N_filt: int = 80,      # Number of Sinc filters (out_channels)
        filt_dim: int = 251,   # Sinc filter kernel size (must be odd)
        pool_len: int = 3,     # MaxPool kernel size
        stride: int = 1,       # Stride for standard conv layers
        emb_dim: int = 256,    # Output embedding dimension
        input_length_samples: int = 48000 # Expected input length (e.g., 3 sec * 16kHz)
    ):
        super().__init__()

        # --- Layer 1: SincConv ---
        # Padding='same' ensures output length is input_length / stride (if stride=1, length is same)
        self.sinc_conv = SincConv1d(
            out_channels=N_filt,
            kernel_size=filt_dim,
            sample_rate=sample_rate,
            padding='same' # Use 'same' padding
        )
        # Calculate output length after SincConv (with stride=1, padding='same')
        len1 = input_length_samples
        self.ln1 = nn.LayerNorm(len1) # LayerNorm after SincConv
        self.pool1 = nn.MaxPool1d(pool_len)
        # Calculate output length after MaxPool1d
        len2 = len1 // pool_len

        # --- Layer 2: Standard Conv1d ---
        self.conv2 = nn.Conv1d(N_filt, 60, kernel_size=5, stride=stride, padding='same')
        self.bn2 = nn.BatchNorm1d(60) # BatchNorm often used here
        # Calculate output length after Conv1d (stride=1, padding='same')
        len3 = len2
        self.pool2 = nn.MaxPool1d(pool_len)
        # Calculate output length after MaxPool1d
        len4 = len3 // pool_len

        # --- Layer 3: Standard Conv1d ---
        self.conv3 = nn.Conv1d(60, 60, kernel_size=5, stride=stride, padding='same')
        self.bn3 = nn.BatchNorm1d(60)
        # Calculate output length after Conv1d (stride=1, padding='same')
        len5 = len4
        self.pool3 = nn.MaxPool1d(pool_len)
        # Calculate output length after MaxPool1d
        len6 = len5 // pool_len

        # --- Flatten and Fully Connected Layers ---
        # Calculate the flattened dimension after the last pooling layer
        flat_dim = 60 * len6

        # Check if flat_dim is valid
        if flat_dim <= 0:
             raise ValueError(f"Calculated flattened dimension ({flat_dim}) is not positive. "
                              f"Check input length ({input_length_samples}), filter sizes, "
                              f"and pooling parameters.")

        self.fc1 = nn.Linear(flat_dim, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, emb_dim)
        # No final activation/normalization on the embedding itself usually

    def forward(self, x):
        # Input shape: (B, 1, L)
        # Ensure input has channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Layer 1: SincConv -> LayerNorm -> ReLU -> MaxPool
        x = self.sinc_conv(x)
        x = F.leaky_relu(self.ln1(x)) # Using LeakyReLU as in some SincNet versions
        x = self.pool1(x)

        # Layer 2: Conv1d -> BatchNorm -> ReLU -> MaxPool
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Layer 3: Conv1d -> BatchNorm -> ReLU -> MaxPool
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Flatten the output for the fully connected layers
        x = x.flatten(1) # Shape: (B, 60 * len6)

        # Fully Connected Layers
        x = F.leaky_relu(self.bn4(self.fc1(x)))
        emb = self.fc2(x) # Shape: (B, emb_dim)

        return emb

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class SincConv1d(nn.Module):
#     """Sinc-based 1D convolution as in SincNet."""

#     def __init__(
#         self,
#         out_channels,
#         kernel_size,
#         sample_rate,
#         in_channels=1,
#         min_low_hz=50,
#         min_band_hz=50,
#     ):
#         super().__init__()
#         if in_channels != 1:
#             raise ValueError("SincConv1d only supports in_channels=1")
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.sample_rate = sample_rate
#         self.min_low_hz = min_low_hz
#         self.min_band_hz = min_band_hz

#         # Initialize filterbank using mel scale boundaries
#         low_hz = 30.0
#         high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
#         mel_bins = torch.linspace(
#             self._hz2mel(low_hz), self._hz2mel(high_hz), out_channels + 1
#         )
#         hz_bins = self._mel2hz(mel_bins)
#         low = hz_bins[:-1]
#         high = hz_bins[1:]
#         self.low_hz_ = nn.Parameter(low.view(-1, 1))
#         self.band_hz_ = nn.Parameter((high - low).view(-1, 1))

#         # Pre-compute window and time axis
#         n_lin = torch.linspace(0, (kernel_size / 2) - 1, int(kernel_size / 2))
#         self.register_buffer('n_', n_lin)
#         window = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(kernel_size) / (kernel_size - 1))
#         self.register_buffer('window_', window)

#     def forward(self, x):
#         low = self.min_low_hz + torch.abs(self.low_hz_)
#         high = torch.clamp(
#             low + self.min_band_hz + torch.abs(self.band_hz_),
#             self.min_low_hz,
#             self.sample_rate / 2,
#         )
#         band = (high - low)[:, 0]

#         filters = []
#         t_right = self.n_ / self.sample_rate
#         half = self.kernel_size // 2
#         for i in range(self.out_channels):
#             f1 = low[i, 0].item()
#             f2 = high[i, 0].item()
#             # band-pass via difference of two sinc functions
#             h_high = torch.sin(2 * math.pi * f2 * t_right) / (math.pi * t_right)
#             h_low = torch.sin(2 * math.pi * f1 * t_right) / (math.pi * t_right)
#             # handle t=0 singularity: lim sinc = 2*f
#             h_high[0] = 2 * f2
#             h_low[0] = 2 * f1
#             # use half-window for positive times
#             h_band = (h_high - h_low) * self.window_[half:half + self.n_.shape[0]]
#             h_band = h_band / (2 * band[i])
#             # central sample ensures odd-length symmetry
#             central = torch.tensor([1.0], device=h_band.device, dtype=h_band.dtype)
#             h = torch.cat([h_band.flip(0), central, h_band])
#             filters.append(h)
#         filt = torch.stack(filters).view(self.out_channels, 1, self.kernel_size)
#         return F.conv1d(x, filt, stride=1, padding=self.kernel_size // 2)

#     @staticmethod
#     def _hz2mel(hz):
#         if isinstance(hz, torch.Tensor):
#             return 2595.0 * torch.log10(1.0 + hz / 700.0)
#         else:
#             return 2595.0 * math.log10(1.0 + hz / 700.0)

#     @staticmethod
#     def _mel2hz(mel):
#         if isinstance(mel, torch.Tensor):
#             return 700.0 * (10 ** (mel / 2595.0) - 1.0)
#         else:
#             return 700.0 * (10 ** (mel / 2595.0) - 1.0)


# class SincNetEmbedding(nn.Module):
#     """
#     Input: (B, 1, L) waveform
#     Output: (B, 256) speaker embedding
#     """

#     def __init__(
#         self,
#         sample_rate: int = 16000,
#         N_filt: int = 80,
#         filt_dim: int = 251,
#         emb_dim: int = 256,
#     ):
#         super().__init__()
#         self.sinc_conv = SincConv1d(N_filt, filt_dim, sample_rate)
#         self.bn1 = nn.BatchNorm1d(N_filt)
#         self.pool1 = nn.MaxPool1d(3)

#         self.conv2 = nn.Conv1d(N_filt, 60, kernel_size=5, padding=2)
#         self.bn2 = nn.BatchNorm1d(60)
#         self.pool2 = nn.MaxPool1d(3)

#         self.conv3 = nn.Conv1d(60, 60, kernel_size=5, padding=2)
#         self.bn3 = nn.BatchNorm1d(60)
#         self.pool3 = nn.MaxPool1d(3)

#         seq_len = sample_rate * 3
#         seq_len = seq_len // 3 // 3 // 3
#         flat_dim = 60 * seq_len

#         self.fc1 = nn.Linear(flat_dim, 512)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.fc2 = nn.Linear(512, emb_dim)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.sinc_conv(x)))
#         x = self.pool1(x)
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool2(x)
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.pool3(x)
#         x = x.flatten(1)
#         x = F.relu(self.bn4(self.fc1(x)))
#         emb = self.fc2(x)
#         return emb

# # import math
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F


# # class SincConv1d(nn.Module):
# #     """Sinc-based 1D convolution as in SincNet."""

# #     def __init__(
# #         self,
# #         out_channels,
# #         kernel_size,
# #         sample_rate,
# #         in_channels=1,
# #         min_low_hz=50,
# #         min_band_hz=50,
# #     ):
# #         super().__init__()
# #         if in_channels != 1:
# #             raise ValueError("SincConv1d only supports in_channels=1")
# #         self.out_channels = out_channels
# #         self.kernel_size = kernel_size
# #         self.sample_rate = sample_rate
# #         self.min_low_hz = min_low_hz
# #         self.min_band_hz = min_band_hz

# #         # Initialize filterbank using mel scale boundaries
# #         low_hz = 30.0
# #         high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
# #         mel_bins = torch.linspace(
# #             self._hz2mel(low_hz), self._hz2mel(high_hz), out_channels + 1
# #         )
# #         hz_bins = self._mel2hz(mel_bins)
# #         low = hz_bins[:-1]
# #         high = hz_bins[1:]
# #         self.low_hz_ = nn.Parameter(low.view(-1, 1))
# #         self.band_hz_ = nn.Parameter((high - low).view(-1, 1))

# #         # Pre-compute window and time axis
# #         n_lin = torch.linspace(0, (kernel_size / 2) - 1, int(kernel_size / 2))
# #         self.register_buffer('n_', n_lin)
# #         window = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(kernel_size) / (kernel_size - 1))
# #         self.register_buffer('window_', window)

# #     def forward(self, x):
# #         low = self.min_low_hz + torch.abs(self.low_hz_)
# #         high = torch.clamp(
# #             low + self.min_band_hz + torch.abs(self.band_hz_),
# #             self.min_low_hz,
# #             self.sample_rate / 2,
# #         )
# #         band = (high - low)[:, 0]

# #         filters = []
# #         t_right = self.n_ / self.sample_rate
# #         half = self.kernel_size // 2
# #         for i in range(self.out_channels):
# #             f1 = low[i, 0].item()
# #             f2 = high[i, 0].item()
# #             # band-pass via difference of two sinc functions
# #             h_high = torch.sin(2 * math.pi * f2 * t_right) / (math.pi * t_right)
# #             h_low = torch.sin(2 * math.pi * f1 * t_right) / (math.pi * t_right)
# #             # use half-window for positive times
# #             h_band = (h_high - h_low) * self.window_[half:half + self.n_.shape[0]]
# #             h_band = h_band / (2 * band[i])
# #             # central sample ensures odd-length symmetry
# #             central = torch.tensor([1.0], device=h_band.device, dtype=h_band.dtype)
# #             h = torch.cat([h_band.flip(0), central, h_band])
# #             filters.append(h)
# #         filt = torch.stack(filters).view(self.out_channels, 1, self.kernel_size)
# #         return F.conv1d(x, filt, stride=1, padding=self.kernel_size // 2)

# #     @staticmethod
# #     def _hz2mel(hz):
# #         if isinstance(hz, torch.Tensor):
# #             return 2595.0 * torch.log10(1.0 + hz / 700.0)
# #         else:
# #             return 2595.0 * math.log10(1.0 + hz / 700.0)

# #     @staticmethod
# #     def _mel2hz(mel):
# #         if isinstance(mel, torch.Tensor):
# #             return 700.0 * (10 ** (mel / 2595.0) - 1.0)
# #         else:
# #             return 700.0 * (10 ** (mel / 2595.0) - 1.0)


# # class SincNetEmbedding(nn.Module):
# #     """
# #     Input: (B, 1, L) waveform
# #     Output: (B, 256) speaker embedding
# #     """

# #     def __init__(
# #         self,
# #         sample_rate: int = 16000,
# #         N_filt: int = 80,
# #         filt_dim: int = 251,
# #         emb_dim: int = 256,
# #     ):
# #         super().__init__()
# #         self.sinc_conv = SincConv1d(N_filt, filt_dim, sample_rate)
# #         self.bn1 = nn.BatchNorm1d(N_filt)
# #         self.pool1 = nn.MaxPool1d(3)

# #         self.conv2 = nn.Conv1d(N_filt, 60, kernel_size=5, padding=2)
# #         self.bn2 = nn.BatchNorm1d(60)
# #         self.pool2 = nn.MaxPool1d(3)

# #         self.conv3 = nn.Conv1d(60, 60, kernel_size=5, padding=2)
# #         self.bn3 = nn.BatchNorm1d(60)
# #         self.pool3 = nn.MaxPool1d(3)

# #         seq_len = sample_rate * 3
# #         seq_len = seq_len // 3 // 3 // 3
# #         flat_dim = 60 * seq_len

# #         self.fc1 = nn.Linear(flat_dim, 512)
# #         self.bn4 = nn.BatchNorm1d(512)
# #         self.fc2 = nn.Linear(512, emb_dim)

# #     def forward(self, x):
# #         x = F.relu(self.bn1(self.sinc_conv(x)))
# #         x = self.pool1(x)
# #         x = F.relu(self.bn2(self.conv2(x)))
# #         x = self.pool2(x)
# #         x = F.relu(self.bn3(self.conv3(x)))
# #         x = self.pool3(x)
# #         x = x.flatten(1)
# #         x = F.relu(self.bn4(self.fc1(x)))
# #         emb = self.fc2(x)
# #         return emb

# # # import math
# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F


# # # class SincConv1d(nn.Module):
# # #     """Sinc-based 1D convolution as in SincNet."""

# # #     def __init__(
# # #         self,
# # #         out_channels,
# # #         kernel_size,
# # #         sample_rate,
# # #         in_channels=1,
# # #         min_low_hz=50,
# # #         min_band_hz=50,
# # #     ):
# # #         super().__init__()
# # #         if in_channels != 1:
# # #             raise ValueError("SincConv1d only supports in_channels=1")
# # #         self.out_channels = out_channels
# # #         self.kernel_size = kernel_size
# # #         self.sample_rate = sample_rate
# # #         self.min_low_hz = min_low_hz
# # #         self.min_band_hz = min_band_hz

# # #         # Initialize filterbank using mel scale boundaries
# # #         low_hz = 30.0
# # #         high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
# # #         mel_bins = torch.linspace(
# # #             self._hz2mel(low_hz), self._hz2mel(high_hz), out_channels + 1
# # #         )
# # #         hz_bins = self._mel2hz(mel_bins)
# # #         low = hz_bins[:-1]
# # #         high = hz_bins[1:]
# # #         self.low_hz_ = nn.Parameter(low.view(-1, 1))
# # #         self.band_hz_ = nn.Parameter((high - low).view(-1, 1))

# # #         # Pre-compute window and time axis
# # #         n_lin = torch.linspace(0, (kernel_size / 2) - 1, int(kernel_size / 2))
# # #         self.register_buffer('n_', n_lin)
# # #         window = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(kernel_size) / (kernel_size - 1))
# # #         self.register_buffer('window_', window)

# # #     def forward(self, x):
# # #         low = self.min_low_hz + torch.abs(self.low_hz_)
# # #         high = torch.clamp(
# # #             low + self.min_band_hz + torch.abs(self.band_hz_),
# # #             self.min_low_hz,
# # #             self.sample_rate / 2,
# # #         )
# # #         band = (high - low)[:, 0]

# # #         filters = []
# # #         t_right = self.n_ / self.sample_rate
# # #         half = self.kernel_size // 2
# # #         for i in range(self.out_channels):
# # #             f1 = low[i, 0].item()
# # #             f2 = high[i, 0].item()
# # #             h_high = torch.sin(2 * math.pi * f2 * t_right) / (math.pi * t_right)
# # #             h_low = torch.sin(2 * math.pi * f1 * t_right) / (math.pi * t_right)
# # #             # match window slice to t_right length
# # #             h_band = (h_high - h_low) * self.window_[half:half + self.n_.shape[0]]
# # #             h_band = h_band / (2 * band[i])
# # #             h = torch.cat([h_band.flip(0), h_band])
# # #             filters.append(h)
# # #         filt = torch.stack(filters).view(self.out_channels, 1, self.kernel_size)
# # #         return F.conv1d(x, filt, stride=1, padding=self.kernel_size // 2)

# # #     @staticmethod
# # #     def _hz2mel(hz):
# # #         if isinstance(hz, torch.Tensor):
# # #             return 2595.0 * torch.log10(1.0 + hz / 700.0)
# # #         else:
# # #             return 2595.0 * math.log10(1.0 + hz / 700.0)

# # #     @staticmethod
# # #     def _mel2hz(mel):
# # #         if isinstance(mel, torch.Tensor):
# # #             return 700.0 * (10 ** (mel / 2595.0) - 1.0)
# # #         else:
# # #             return 700.0 * (10 ** (mel / 2595.0) - 1.0)


# # # class SincNetEmbedding(nn.Module):
# # #     """
# # #     Input: (B, 1, L) waveform
# # #     Output: (B, 256) speaker embedding
# # #     """

# # #     def __init__(
# # #         self,
# # #         sample_rate: int = 16000,
# # #         N_filt: int = 80,
# # #         filt_dim: int = 251,
# # #         emb_dim: int = 256,
# # #     ):
# # #         super().__init__()
# # #         self.sinc_conv = SincConv1d(N_filt, filt_dim, sample_rate)
# # #         self.bn1 = nn.BatchNorm1d(N_filt)
# # #         self.pool1 = nn.MaxPool1d(3)

# # #         self.conv2 = nn.Conv1d(N_filt, 60, kernel_size=5, padding=2)
# # #         self.bn2 = nn.BatchNorm1d(60)
# # #         self.pool2 = nn.MaxPool1d(3)

# # #         self.conv3 = nn.Conv1d(60, 60, kernel_size=5, padding=2)
# # #         self.bn3 = nn.BatchNorm1d(60)
# # #         self.pool3 = nn.MaxPool1d(3)

# # #         seq_len = sample_rate * 3
# # #         seq_len = seq_len // 3 // 3 // 3
# # #         flat_dim = 60 * seq_len

# # #         self.fc1 = nn.Linear(flat_dim, 512)
# # #         self.bn4 = nn.BatchNorm1d(512)
# # #         self.fc2 = nn.Linear(512, emb_dim)

# # #     def forward(self, x):
# # #         x = F.relu(self.bn1(self.sinc_conv(x)))
# # #         x = self.pool1(x)
# # #         x = F.relu(self.bn2(self.conv2(x)))
# # #         x = self.pool2(x)
# # #         x = F.relu(self.bn3(self.conv3(x)))
# # #         x = self.pool3(x)
# # #         x = x.flatten(1)
# # #         x = F.relu(self.bn4(self.fc1(x)))
# # #         emb = self.fc2(x)
# # #         return emb

# # # # import math
# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.nn.functional as F


# # # # class SincConv1d(nn.Module):
# # # #     """Sinc-based 1D convolution as in SincNet."""
# # # #     def __init__(
# # # #         self,
# # # #         out_channels,
# # # #         kernel_size,
# # # #         sample_rate,
# # # #         in_channels=1,
# # # #         min_low_hz=50,
# # # #         min_band_hz=50,
# # # #     ):
# # # #         super().__init__()
# # # #         if in_channels != 1:
# # # #             raise ValueError("SincConv1d only supports in_channels=1")
# # # #         self.out_channels = out_channels
# # # #         self.kernel_size = kernel_size
# # # #         self.sample_rate = sample_rate
# # # #         self.min_low_hz = min_low_hz
# # # #         self.min_band_hz = min_band_hz

# # # #         # Initialize filterbank using mel scale boundaries
# # # #         low_hz = 30.0
# # # #         high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
# # # #         # Generate mel-spaced frequencies
# # # #         mel_bins = torch.linspace(
# # # #             self._hz2mel(low_hz), self._hz2mel(high_hz), out_channels + 1
# # # #         )
# # # #         hz_bins = self._mel2hz(mel_bins)
# # # #         low = hz_bins[:-1]
# # # #         high = hz_bins[1:]
# # # #         self.low_hz_ = nn.Parameter(low.view(-1, 1))
# # # #         self.band_hz_ = nn.Parameter((high - low).view(-1, 1))

# # # #         # Pre-compute window and time axis
# # # #         n_lin = torch.linspace(0, (kernel_size / 2) - 1, int(kernel_size / 2))
# # # #         self.register_buffer('n_', n_lin)
# # # #         window = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(kernel_size) / (kernel_size - 1))
# # # #         self.register_buffer('window_', window)

# # # #     def forward(self, x):
# # # #         # Construct filters on-the-fly
# # # #         low = self.min_low_hz + torch.abs(self.low_hz_)
# # # #         high = torch.clamp(
# # # #             low + self.min_band_hz + torch.abs(self.band_hz_),
# # # #             self.min_low_hz,
# # # #             self.sample_rate / 2,
# # # #         )
# # # #         band = (high - low)[:, 0]

# # # #         filters = []
# # # #         t_right = self.n_ / self.sample_rate
# # # #         for i in range(self.out_channels):
# # # #             f1 = low[i, 0].item()
# # # #             f2 = high[i, 0].item()
# # # #             # band-pass via difference of two sinc functions
# # # #             h_high = torch.sin(2 * math.pi * f2 * t_right) / (math.pi * t_right)
# # # #             h_low = torch.sin(2 * math.pi * f1 * t_right) / (math.pi * t_right)
# # # #             h_band = (h_high - h_low) * self.window_[self.kernel_size // 2:]
# # # #             h_band = h_band / (2 * band[i])
# # # #             h = torch.cat([h_band.flip(0), h_band])
# # # #             filters.append(h)
# # # #         filt = torch.stack(filters).view(self.out_channels, 1, self.kernel_size)
# # # #         return F.conv1d(x, filt, stride=1, padding=self.kernel_size // 2)

# # # #     @staticmethod
# # # #     def _hz2mel(hz):
# # # #         # hz can be float or tensor
# # # #         if isinstance(hz, torch.Tensor):
# # # #             return 2595.0 * torch.log10(1.0 + hz / 700.0)
# # # #         else:
# # # #             return 2595.0 * math.log10(1.0 + hz / 700.0)

# # # #     @staticmethod
# # # #     def _mel2hz(mel):
# # # #         # mel can be tensor or float
# # # #         if isinstance(mel, torch.Tensor):
# # # #             return 700.0 * (10 ** (mel / 2595.0) - 1.0)
# # # #         else:
# # # #             return 700.0 * (10 ** (mel / 2595.0) - 1.0)


# # # # class SincNetEmbedding(nn.Module):
# # # #     """
# # # #     Input: (B, 1, L) waveform
# # # #     Output: (B, 256) speaker embedding
# # # #     """
# # # #     def __init__(
# # # #         self,
# # # #         sample_rate: int = 16000,
# # # #         N_filt: int = 80,
# # # #         filt_dim: int = 251,
# # # #         emb_dim: int = 256,
# # # #     ):
# # # #         super().__init__()
# # # #         self.sinc_conv = SincConv1d(N_filt, filt_dim, sample_rate)
# # # #         self.bn1 = nn.BatchNorm1d(N_filt)
# # # #         self.pool1 = nn.MaxPool1d(3)

# # # #         self.conv2 = nn.Conv1d(N_filt, 60, kernel_size=5, padding=2)
# # # #         self.bn2 = nn.BatchNorm1d(60)
# # # #         self.pool2 = nn.MaxPool1d(3)

# # # #         self.conv3 = nn.Conv1d(60, 60, kernel_size=5, padding=2)
# # # #         self.bn3 = nn.BatchNorm1d(60)
# # # #         self.pool3 = nn.MaxPool1d(3)

# # # #         # Calculate flattened dimension after pooling for 3-second input
# # # #         seq_len = sample_rate * 3
# # # #         seq_len = seq_len // 3 // 3 // 3
# # # #         flat_dim = 60 * seq_len

# # # #         self.fc1 = nn.Linear(flat_dim, 512)
# # # #         self.bn4 = nn.BatchNorm1d(512)
# # # #         self.fc2 = nn.Linear(512, emb_dim)

# # # #     def forward(self, x):
# # # #         x = F.relu(self.bn1(self.sinc_conv(x)))
# # # #         x = self.pool1(x)
# # # #         x = F.relu(self.bn2(self.conv2(x)))
# # # #         x = self.pool2(x)
# # # #         x = F.relu(self.bn3(self.conv3(x)))
# # # #         x = self.pool3(x)
# # # #         x = x.flatten(1)
# # # #         x = F.relu(self.bn4(self.fc1(x)))
# # # #         emb = self.fc2(x)
# # # #         return emb

# # # # # # models/sincnet.py
# # # # # import math
# # # # # import torch
# # # # # import torch.nn as nn
# # # # # import torch.nn.functional as F

# # # # # class SincConv1d(nn.Module):
# # # # #     """Sinc-based 1D convolution as in SincNet."""
# # # # #     def __init__(self, out_channels, kernel_size, sample_rate,
# # # # #                  in_channels=1, min_low_hz=50, min_band_hz=50):
# # # # #         super().__init__()
# # # # #         if in_channels != 1:
# # # # #             raise ValueError("SincConv1d only supports in_channels=1")
# # # # #         self.out_channels = out_channels
# # # # #         self.kernel_size = kernel_size
# # # # #         self.sample_rate = sample_rate
# # # # #         self.min_low_hz = min_low_hz
# # # # #         self.min_band_hz = min_band_hz

# # # # #         low_hz = 30.0
# # # # #         high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
# # # # #         mel = torch.linspace(self._hz2mel(low_hz),
# # # # #                              self._hz2mel(high_hz),
# # # # #                              out_channels + 1)
# # # # #         hz = self._mel2hz(mel)
# # # # #         low = hz[:-1]
# # # # #         high = hz[1:]
# # # # #         self.low_hz_  = nn.Parameter(low.view(-1,1))
# # # # #         self.band_hz_ = nn.Parameter((high - low).view(-1,1))

# # # # #         n_lin = torch.linspace(0, (kernel_size / 2) - 1, int(kernel_size/2))
# # # # #         self.register_buffer('n_', n_lin)
# # # # #         window = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(kernel_size) / (kernel_size-1))
# # # # #         self.register_buffer('window_', window)

# # # # #     def forward(self, x):
# # # # #         low  = self.min_low_hz  + torch.abs(self.low_hz_)
# # # # #         high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
# # # # #                            self.min_low_hz, self.sample_rate/2)
# # # # #         band = (high - low)[:,0]

# # # # #         filters = []
# # # # #         t_right = self.n_ / self.sample_rate
# # # # #         for i in range(self.out_channels):
# # # # #             f1 = low[i,0]; f2 = high[i,0]
# # # # #             h_high = torch.sin(2*math.pi*f2*t_right) / (math.pi*t_right)
# # # # #             h_low  = torch.sin(2*math.pi*f1*t_right) / (math.pi*t_right)
# # # # #             h_band = (h_high - h_low) * self.window_[self.kernel_size//2:]
# # # # #             h_band = h_band / (2 * band[i])
# # # # #             h = torch.cat([h_band.flip(0), h_band])
# # # # #             filters.append(h)
# # # # #         filt = torch.stack(filters).view(self.out_channels, 1, self.kernel_size)
# # # # #         return F.conv1d(x, filt, stride=1, padding=self.kernel_size//2)

# # # # #     @staticmethod
# # # # #     def _hz2mel(hz):
# # # # #         return 2595 * torch.log10(1 + hz / 700)
# # # # #     @staticmethod
# # # # #     def _mel2hz(mel):
# # # # #         return 700 * (10**(mel / 2595) - 1)

# # # # # class SincNetEmbedding(nn.Module):
# # # # #     """
# # # # #     (B,1,L) -> (B,256)
# # # # #     """
# # # # #     def __init__(self,
# # # # #                  sample_rate: int = 16000,
# # # # #                  N_filt: int = 80,
# # # # #                  filt_dim: int = 251,
# # # # #                  emb_dim: int = 256):
# # # # #         super().__init__()
# # # # #         self.sinc_conv = SincConv1d(N_filt, filt_dim, sample_rate)
# # # # #         self.bn1  = nn.BatchNorm1d(N_filt)
# # # # #         self.pool1 = nn.MaxPool1d(3)
# # # # #         self.conv2 = nn.Conv1d(N_filt,  60, 5, padding=2)
# # # # #         self.bn2   = nn.BatchNorm1d(60)
# # # # #         self.pool2 = nn.MaxPool1d(3)
# # # # #         self.conv3 = nn.Conv1d(60, 60, 5, padding=2)
# # # # #         self.bn3   = nn.BatchNorm1d(60)
# # # # #         self.pool3 = nn.MaxPool1d(3)

# # # # #         seq_len = sample_rate * 3 // 3 // 3 // 3
# # # # #         flat_dim = 60 * seq_len
# # # # #         self.fc1 = nn.Linear(flat_dim, 512)
# # # # #         self.bn4 = nn.BatchNorm1d(512)
# # # # #         self.fc2 = nn.Linear(512, emb_dim)

# # # # #     def forward(self, x):
# # # # #         x = F.relu(self.bn1(self.sinc_conv(x)))
# # # # #         x = self.pool1(x)
# # # # #         x = F.relu(self.bn2(self.conv2(x)))
# # # # #         x = self.pool2(x)
# # # # #         x = F.relu(self.bn3(self.conv3(x)))
# # # # #         x = self.pool3(x)
# # # # #         x = x.flatten(1)
# # # # #         x = F.relu(self.bn4(self.fc1(x)))
# # # # #         emb = self.fc2(x)
# # # # #         return emb
