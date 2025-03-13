#!/usr/bin/env julia
# CNN Feature Extraction Module using Flux:
# Loads a mel spectrogram from a CSV file, builds a CNN model, and extracts features.
# Usage:
#   julia cnn_feature_extraction.jl mel_spectrogram.csv

using Flux
using DelimitedFiles

function load_mel_spectrogram(csv_file::String)
    mel_spec = readdlm(csv_file, ',')
    return Array(mel_spec)
end

# Build a simple CNN for feature extraction.
function build_cnn(input_shape)
    return Chain(
        x -> reshape(x, input_shape...),  # reshape to (height, width, channels)
        Conv((3,3), 1=>8, relu),
        MaxPool((2,2)),
        Conv((3,3), 8=>16, relu),
        MaxPool((2,2)),
        flatten,
        Dense(16 * div(input_shape[1]-4,4) * div(input_shape[2]-4,4), 64, relu)
    )
end

function extract_features(csv_file::String)
    mel_spec = load_mel_spectrogram(csv_file)
    # Normalize spectrogram.
    mel_spec = (mel_spec .- minimum(mel_spec)) ./ (maximum(mel_spec) - minimum(mel_spec) + 1e-6)
    n_mels, time_frames = size(mel_spec)
    # Add a channel dimension (here, channel = 1).
    input_shape = (n_mels, time_frames, 1)
    
    cnn_model = build_cnn(input_shape)
    # Add a batch dimension.
    input_tensor = reshape(mel_spec, n_mels, time_frames, 1, 1)
    features = cnn_model(input_tensor)
    println("Extracted features: ", features)
    return features
end

if length(ARGS) < 1
    println("Usage: julia cnn_feature_extraction.jl mel_spectrogram.csv")
    exit(1)
end

csv_file = ARGS[1]
extract_features(csv_file)