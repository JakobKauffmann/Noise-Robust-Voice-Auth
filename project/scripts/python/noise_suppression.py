#!/usr/bin/env python
"""
Noise Suppression Module:
Loads a noisy audio file, applies noise reduction using the noisereduce package,
and writes out the noise-suppressed audio.
Usage:
    python noise_suppression.py input_noisy.wav output_suppressed.wav
"""

import librosa
import noisereduce as nr
import soundfile as sf
import sys

def suppress_noise(input_path, output_path):
    # Load the noisy audio.
    audio, sr = librosa.load(input_path, sr=None)
    print(f"Loaded audio: {input_path} (sample rate={sr}, length={len(audio)})")
    # Use the first 0.5 seconds as a noise sample.
    noise_sample = audio[:int(0.5 * sr)]
    reduced_audio = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample, prop_decrease=1.0)
    # Save the noise-suppressed audio.
    sf.write(output_path, reduced_audio, sr)
    print(f"Saved noise-suppressed audio to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python noise_suppression.py input_noisy.wav output_suppressed.wav")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    suppress_noise(input_path, output_path)
