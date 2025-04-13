# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from torch.utils.data import DataLoader
from data_utils import get_data_loaders, VoxCelebDatasetWithPath
from models import MelCNN, FeatureFusion, DecisionModule
from train_eval import train_model, evaluate_model, evaluate_authentication, compute_eer

# Set device and optimize for the A100 GPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# Paths & hyperparameters -- change these as needed.
DATA_ROOT = '/path/to/voxceleb1'  # Update to your VoxCeleb1 dataset directory.
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 30
LEARNING_RATE = 0.0003
CHECKPOINT_PATH_MEL = 'melcnn_checkpoint.pth'
NUM_CLASSES = 1000  # Update to the number of speakers in the training set.

# ---- Phase 1: Train MelCNN to extract embeddings ----
print("Loading data loaders for MelCNN training...")
train_loader, val_loader, test_loader = get_data_loaders(DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
print("Training MelCNN...")
mel_model = MelCNN(emb_dim=256, feature_dim=1024, dropout_rate=0.5).to(device)
criterion = nn.CrossEntropyLoss()  # Dummy loss – replace with your own if needed.
optimizer = optim.Adam(mel_model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
mel_model = train_model(mel_model, train_loader, val_loader, optimizer, criterion, scheduler, device, NUM_EPOCHS, CHECKPOINT_PATH_MEL)

# Optionally, load the best checkpoint.
if os.path.exists(CHECKPOINT_PATH_MEL):
    checkpoint = torch.load(CHECKPOINT_PATH_MEL, map_location=device)
    mel_model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded MelCNN checkpoint.")

# ---- Phase 2: Load precomputed SincNet embeddings and mapping ----
print("Loading precomputed SincNet embeddings and file mapping...")
sincnet_emb_path = '/content/drive/Shareddrives/VoxCeleb1/SincNet/Features/sincnet_embeddings.npy'
sincnet_map_path = '/content/drive/Shareddrives/VoxCeleb1/SincNet/Features/sincnet_file_mapping.pkl'
with open(sincnet_map_path, 'rb') as f:
    file_mapping = pickle.load(f)
sincnet_embeddings = np.load(sincnet_emb_path)
sincnet_embeddings = torch.tensor(sincnet_embeddings, dtype=torch.float).to(device)

# ---- Phase 3: Train Fusion & Decision Module ----
print("Loading training data for fusion (with file paths)...")
# Use the dataset that returns (features, label, file_path)
from data_utils import VoxCelebDatasetWithPath
# We use the same transform as in get_data_loaders.
import torchaudio
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=40
)
db_transform = torchaudio.transforms.AmplitudeToDB()
transform = torch.nn.Sequential(mel_transform, db_transform)
fusion_dataset = VoxCelebDatasetWithPath(DATA_ROOT, transform=transform, subset='train')
fusion_loader = DataLoader(fusion_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

fusion_model = FeatureFusion(emb_dim=256).to(device)
decision_model = DecisionModule(emb_dim=256, num_classes=NUM_CLASSES, dropout_rate=0.5).to(device)
optimizer_fusion = optim.Adam(list(fusion_model.parameters()) + list(decision_model.parameters()), lr=LEARNING_RATE)
criterion_fusion = nn.CrossEntropyLoss()

def train_fused_model(mel_model, fusion_model, decision_model, data_loader, optimizer, criterion, device):
    mel_model.eval()  # Freeze MelCNN (assumed pre-trained)
    fusion_model.train()
    decision_model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for features, labels, file_paths in data_loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Get MelCNN embedding.
        mel_emb = mel_model(features)  # [B, 256]
        # For each sample, look up SincNet embedding.
        batch_sinc_emb = []
        for fp in file_paths:
            # Assume file_mapping is a dict: {full_wav_path: index}
            # If paths do not exactly match, adjust (e.g., based on basename)
            key = fp  # or os.path.basename(fp) if needed.
            idx = file_mapping.get(key, None)
            if idx is None:
                raise ValueError(f"File path {key} not found in SincNet mapping!")
            batch_sinc_emb.append(sincnet_embeddings[idx])
        sinc_emb = torch.stack(batch_sinc_emb)  # [B, 256]
        # Fuse using attention.
        fused_emb = fusion_model(mel_emb, sinc_emb)
        logits = decision_model(fused_emb)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

NUM_EPOCHS_FUSION = 10
print("Training fused decision model...")
for epoch in range(NUM_EPOCHS_FUSION):
    loss_fused, acc_fused = train_fused_model(mel_model, fusion_model, decision_model, fusion_loader, optimizer_fusion, criterion_fusion, device)
    print(f"Fusion Epoch {epoch+1}/{NUM_EPOCHS_FUSION}: Loss {loss_fused:.4f}, Acc {acc_fused:.4f}")
    # Optionally, add checkpoint saving here similar to MelCNN.

# ---- Phase 4: Testing ----
print("Testing identification performance...")
# Test MelCNN only
test_mel_acc = evaluate_model(mel_model, fusion_loader, device)
print(f"MelCNN Identification Accuracy: {test_mel_acc:.4f}")

# For SincNet only, assume we can compute nearest-neighbor accuracy from precomputed embeddings.
# (You would build a separate evaluation routine based on your SincNet extraction.)
print("SincNet identification evaluation is assumed to be done offline.")

# Test fused model
fusion_model.eval()
decision_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels, file_paths in fusion_loader:
        features = features.to(device)
        labels = labels.to(device)
        mel_emb = mel_model(features)
        batch_sinc_emb = []
        for fp in file_paths:
            key = fp
            idx = file_mapping.get(key, None)
            if idx is None:
                raise ValueError(f"File path {key} not found in SincNet mapping!")
            batch_sinc_emb.append(sincnet_embeddings[idx])
        sinc_emb = torch.stack(batch_sinc_emb)
        fused_emb = fusion_model(mel_emb, sinc_emb)
        logits = decision_model(fused_emb)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
fused_acc = correct / total
print(f"Fused Model Identification Accuracy: {fused_acc:.4f}")

print("Evaluating authentication (EER) for MelCNN...")
eer_mel = evaluate_authentication(mel_model, fusion_loader, device)
print(f"MelCNN Authentication EER: {eer_mel:.4f}")

# For fused model authentication evaluation, we extract fused embeddings and compute EER.
def get_fused_embeddings(mel_model, fusion_model, decision_model, data_loader, device):
    mel_model.eval()
    fusion_model.eval()
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for features, labels, file_paths in data_loader:
            features = features.to(device)
            labels_list.append(labels)
            mel_emb = mel_model(features)
            batch_sinc_emb = []
            for fp in file_paths:
                key = fp
                idx = file_mapping.get(key, None)
                if idx is None:
                    raise ValueError(f"File path {key} not found in SincNet mapping!")
                batch_sinc_emb.append(sincnet_embeddings[idx])
            sinc_emb = torch.stack(batch_sinc_emb)
            fused_emb = fusion_model(mel_emb, sinc_emb)
            embeddings_list.append(fused_emb.cpu())
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return embeddings, labels

fused_embs, fused_labels = get_fused_embeddings(mel_model, fusion_model, decision_model, fusion_loader, device)
fused_embs = nn.functional.normalize(fused_embs, p=2, dim=1)
sim_matrix = torch.matmul(fused_embs, fused_embs.t()).numpy()
labels_np = fused_labels.numpy()
genuine = []
imposter = []
N = fused_embs.shape[0]
for i in range(N):
    for j in range(i+1, N):
        if labels_np[i] == labels_np[j]:
            genuine.append(sim_matrix[i,j])
        else:
            imposter.append(sim_matrix[i,j])
scores = genuine + imposter
targets = [1]*len(genuine) + [0]*len(imposter)
eer_fused = compute_eer(scores, targets)
print(f"Fused Model Authentication EER: {eer_fused:.4f}")
