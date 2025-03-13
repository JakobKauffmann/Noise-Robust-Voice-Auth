#!/usr/bin/env python
"""
Preprocessing Module:
Loads an audio file, adds Gaussian noise, and saves the result.
Usage:
    python preprocessing.py input_audio.wav output_audio.wav [noise_factor]
"""

import librosa
import numpy as np
import soundfile as sf
import sys

def add_noise(audio, noise_factor=0.005):
    # Generate Gaussian noise and add it to the audio signal.
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    # Normalize to avoid clipping.
    augmented_audio = augmented_audio / np.max(np.abs(augmented_audio))
    return augmented_audio

def preprocess_audio(input_path, output_path, noise_factor=0.005):
    # Load the audio file.
    audio, sr = librosa.load(input_path, sr=None)
    print(f"Loaded audio: {input_path} (sample rate={sr}, length={len(audio)})")
    # Add noise.
    noisy_audio = add_noise(audio, noise_factor)
    # Save the processed audio.
    sf.write(output_path, noisy_audio, sr)
    print(f"Saved preprocessed audio with noise to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocessing.py input_audio.wav output_audio.wav [noise_factor]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    noise_factor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
    preprocess_audio(input_path, output_path, noise_factor)
