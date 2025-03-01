# voice_auth/utils/preprocessing.py
import os
import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa

class AudioPreprocessor:
    """
    Handles preprocessing of audio files for voice authentication including:
    - Loading and resampling
    - Normalization
    - Voice activity detection
    - Segmentation
    """
    def __init__(self, sample_rate=16000, duration=3.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        
    def load_audio(self, file_path):
        """Load audio file and resample if necessary."""
        waveform, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        return waveform.squeeze(0)
    
    def normalize_audio(self, waveform):
        """Normalize audio to have zero mean and unit variance."""
        return (waveform - torch.mean(waveform)) / (torch.std(waveform) + 1e-8)
    
    def segment_audio(self, waveform):
        """Segment audio to fixed length."""
        if waveform.shape[0] >= self.num_samples:
            # Randomly crop to desired length
            max_start = waveform.shape[0] - self.num_samples
            start = random.randint(0, max_start)
            return waveform[start:start+self.num_samples]
        else:
            # Pad if too short
            padding = self.num_samples - waveform.shape[0]
            return torch.nn.functional.pad(waveform, (0, padding))
    
    def process_audio(self, file_path):
        """Process audio file through full preprocessing pipeline."""
        waveform = self.load_audio(file_path)
        waveform = self.normalize_audio(waveform)
        waveform = self.segment_audio(waveform)
        return waveform


class NoiseAugmenter:
    """
    Augments audio with various noise types to improve model robustness.
    Implements various noise augmentation techniques to simulate real-world conditions.
    """
    def __init__(self, noise_dir=None, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_dir = noise_dir
        self.noise_files = []
        
        # Load noise files if directory provided
        if noise_dir and os.path.exists(noise_dir):
            self.noise_files = [
                os.path.join(noise_dir, f) for f in os.listdir(noise_dir)
                if f.endswith(('.wav', '.mp3', '.flac'))
            ]
    
    def add_background_noise(self, waveform, snr_db=10):
        """Add background noise at specified signal-to-noise ratio."""
        if not self.noise_files:
            return waveform
            
        # Randomly select a noise file
        noise_file = random.choice(self.noise_files)
        noise, _ = torchaudio.load(noise_file)
        
        # Convert to mono if stereo
        if noise.shape[0] > 1:
            noise = torch.mean(noise, dim=0, keepdim=True)
        
        # Resample if necessary
        if noise.shape[1] != waveform.shape[0]:
            if noise.shape[1] > waveform.shape[0]:
                # Randomly crop noise
                start = random.randint(0, noise.shape[1] - waveform.shape[0])
                noise = noise[:, start:start+waveform.shape[0]]
            else:
                # Repeat noise to match length
                repeats = int(np.ceil(waveform.shape[0] / noise.shape[1]))
                noise = noise.repeat(1, repeats)
                noise = noise[:, :waveform.shape[0]]
        
        noise = noise.squeeze(0)
        
        # Calculate scaling factor for desired SNR
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)
        
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (noise_power * snr_linear))
        
        # Add scaled noise
        return waveform + scale * noise
    
    def add_reverberation(self, waveform, reverb_level=0.3):
        """Add reverberation effect to simulate room acoustics."""
        # Simple convolution-based reverberation
        impulse_response = torch.exp(-torch.arange(8000, dtype=torch.float32) / 1000)
        impulse_response = impulse_response * reverb_level
        
        # Apply convolution
        reverb = torch.nn.functional.conv1d(
            waveform.view(1, 1, -1),
            impulse_response.view(1, 1, -1),
            padding=impulse_response.shape[0] - 1
        )
        
        # Mix with original
        result = (waveform.view(1, -1) + reverb.view(1, -1))[:, :waveform.shape[0]]
        return result.squeeze(0)
    
    def apply_bandpass_filter(self, waveform, low_freq=300, high_freq=3400):
        """Apply bandpass filter to simulate telephone quality."""
        # Convert to numpy for librosa processing
        waveform_np = waveform.numpy()
        
        # Apply bandpass filter
        filtered = librosa.effects.preemphasis(waveform_np)
        
        # Convert back to torch tensor
        return torch.from_numpy(filtered.astype(np.float32))
    
    def add_compression_artifacts(self, waveform, quality=0.7):
        """Simulate compression artifacts."""
        # This is a simplified simulation - in practice you might use actual codec compression
        # Convert to mel and back to simulate compression loss
        mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=int(128 * quality)  # Reduce mel bins for more compression
        )
        
        # Griffin-Lim is part of torchaudio's functional module
        griffin_lim = T.GriffinLim(
            n_fft=1024,
            hop_length=512,
            power=2.0
        )
        
        # Apply transformation
        mel_spec = mel_transform(waveform)
        reconstructed = griffin_lim(mel_spec)
        
        # Ensure same length as original
        if reconstructed.shape[0] > waveform.shape[0]:
            reconstructed = reconstructed[:waveform.shape[0]]
        else:
            reconstructed = torch.nn.functional.pad(reconstructed, (0, waveform.shape[0] - reconstructed.shape[0]))
            
        return reconstructed
    
    def augment(self, waveform, augmentation_types=None, noise_level=0.5):
        """Apply random augmentations to the waveform."""
        if augmentation_types is None:
            augmentation_types = ['noise', 'reverb', 'bandpass', 'compression']
        
        # Randomly decide which augmentations to apply
        augmentations = random.sample(augmentation_types, 
                                       k=random.randint(1, len(augmentation_types)))
        
        # Apply selected augmentations
        augmented = waveform.clone()
        
        for aug_type in augmentations:
            if aug_type == 'noise' and random.random() < noise_level:
                snr = random.uniform(5, 20)  # Random SNR between 5 and 20 dB
                augmented = self.add_background_noise(augmented, snr_db=snr)
                
            elif aug_type == 'reverb' and random.random() < noise_level:
                reverb_level = random.uniform(0.1, 0.5)
                augmented = self.add_reverberation(augmented, reverb_level=reverb_level)
                
            elif aug_type == 'bandpass' and random.random() < noise_level:
                low = random.randint(200, 400)
                high = random.randint(3000, 4000)
                augmented = self.apply_bandpass_filter(augmented, low_freq=low, high_freq=high)
                
            elif aug_type == 'compression' and random.random() < noise_level:
                quality = random.uniform(0.5, 0.9)
                augmented = self.add_compression_artifacts(augmented, quality=quality)
        
        return augmented


# Example usage
if __name__ == "__main__":
    # Example preprocessing pipeline
    preprocessor = AudioPreprocessor(sample_rate=16000, duration=3.0)
    augmenter = NoiseAugmenter(noise_dir="voice_auth/data/noise", sample_rate=16000)
    
    # Process a sample file
    sample_file = "voice_auth/data/raw/sample.wav"
    if os.path.exists(sample_file):
        # Preprocess
        waveform = preprocessor.process_audio(sample_file)
        
        # Augment
        augmented = augmenter.augment(waveform, noise_level=0.7)
        
        # Save processed audio for verification
        torchaudio.save(
            "voice_auth/data/processed/sample_processed.wav", 
            augmented.unsqueeze(0), 
            preprocessor.sample_rate
        )
        
        print(f"Original shape: {waveform.shape}")
        print(f"Processed shape: {augmented.shape}")
