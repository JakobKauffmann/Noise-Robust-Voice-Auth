#!/usr/bin/env julia
# Mel Spectrogram Generation Module:
# Reads an audio file, computes its mel spectrogram, and saves the result as a CSV file.
# Usage:
#   julia mel_spectrogram.jl input_audio.wav output_mel.csv

using WAV
using DSP
using DelimitedFiles

# Convert Hz to Mel scale.
function hz_to_mel(hz)
    return 2595 * log10(1 + hz / 700)
end

# Convert Mel scale to Hz.
function mel_to_hz(mel)
    return 700 * (10^(mel / 2595) - 1)
end

# Create a mel filterbank matrix.
function mel_filterbank(n_filters, n_fft, sr, fmin=0, fmax=8000)
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mels = range(mel_min, mel_max, length=n_filters+2)
    freqs = mel_to_hz.(mels)
    bins = floor.((n_fft+1) * freqs ./ sr)
    
    filterbank = zeros(n_filters, div(n_fft, 2)+1)
    for m in 1:n_filters
        f_m_minus = Int(bins[m])
        f_m = Int(bins[m+1])
        f_m_plus = Int(bins[m+2])
        for k in f_m_minus+1:f_m
            filterbank[m, k+1] = (k - bins[m]) / (bins[m+1] - bins[m])
        end
        for k in f_m+1:f_m_plus
            filterbank[m, k+1] = (bins[m+2] - k) / (bins[m+2] - bins[m+1])
        end
    end
    return filterbank
end

function mel_spectrogram(audio_path::String, output_csv::String; n_fft=1024, hop_length=512, n_mels=40)
    # Read audio file.
    audio, sr = wavread(audio_path)
    audio = audio[:,1]  # use first channel if stereo

    # Compute the Short-Time Fourier Transform.
    stft_matrix = stft(audio, n_fft=n_fft, hop=hop_length, window=hann(n_fft))
    power_spec = abs.(stft_matrix).^2

    # Build the mel filterbank.
    filterbank = mel_filterbank(n_mels, n_fft, sr)
    mel_spec = filterbank * power_spec

    # Convert to decibel scale.
    mel_spec_db = 10 .* log10.(mel_spec .+ 1e-10)

    # Save the mel spectrogram as CSV.
    writedlm(output_csv, mel_spec_db, ',')
    println("Saved mel spectrogram to ", output_csv)
end

# Main execution.
if length(ARGS) < 2
    println("Usage: julia mel_spectrogram.jl input_audio.wav output_mel.csv")
    exit(1)
end

input_audio = ARGS[1]
output_csv = ARGS[2]
mel_spectrogram(input_audio, output_csv)