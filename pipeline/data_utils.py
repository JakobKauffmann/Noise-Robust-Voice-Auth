# data_utils.py
import os
import glob
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class VoxCelebDatasetWithPath(Dataset):
    """
    Dataset for VoxCeleb1 that converts raw audio into Mel spectrograms.
    Returns a tuple of (features, label, file_path).
    """
    def __init__(self, root_dir, transform=None, subset='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        # List speaker directories sorted alphabetically.
        speaker_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        speaker_dirs.sort()
        
        # Simple split: 70% train, 15% val, 15% test.
        num_speakers = len(speaker_dirs)
        train_split = int(0.7 * num_speakers)
        val_split = int(0.85 * num_speakers)
        if subset == 'train':
            selected_speakers = speaker_dirs[:train_split]
        elif subset == 'val':
            selected_speakers = speaker_dirs[train_split:val_split]
        elif subset == 'test':
            selected_speakers = speaker_dirs[val_split:]
        else:
            raise ValueError("Invalid subset")
            
        self.speaker_to_label = {spk: idx for idx, spk in enumerate(selected_speakers)}
        for spk in selected_speakers:
            spk_dir = os.path.join(root_dir, spk)
            wav_files = glob.glob(os.path.join(spk_dir, '*.wav'))
            for wav_path in wav_files:
                # We return file path twice for convenience.
                self.samples.append((wav_path, self.speaker_to_label[spk], wav_path))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        wav_path, label, file_path = self.samples[idx]
        waveform, sr = torchaudio.load(wav_path)
        # Use only one channel if multichannel.
        if waveform.shape[0] > 1:
            waveform = waveform[0].unsqueeze(0)
        if self.transform is not None:
            features = self.transform(waveform)
        else:
            features = waveform
        return features, label, file_path

def get_data_loaders(root_dir, batch_size=32, num_workers=4):
    """
    Creates data loaders for training, validation, and testing.
    Uses a MelSpectrogram transform (with conversion to dB).
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=40
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()
    transform = torch.nn.Sequential(mel_transform, db_transform)
    
    train_dataset = VoxCelebDatasetWithPath(root_dir, transform=transform, subset='train')
    val_dataset = VoxCelebDatasetWithPath(root_dir, transform=transform, subset='val')
    test_dataset = VoxCelebDatasetWithPath(root_dir, transform=transform, subset='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    DATA_ROOT = '/path/to/voxceleb1'  # Change this path accordingly.
    train_loader, val_loader, test_loader = get_data_loaders(DATA_ROOT)
    for features, label, file_path in train_loader:
        print(features.shape, label.shape, file_path[0])
        break
