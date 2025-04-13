# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import get_data_loaders
from models import MelCNN, FeatureFusion, DecisionModule, load_sincnet, DummySincNet
from train_eval import train_model, evaluate_model, evaluate_authentication, compute_eer

def main():
    # Optimize for the A100 GPU.
    torch.backends.cudnn.benchmark = True

    # Configuration parameters
    DATA_ROOT = '/path/to/voxceleb1'  # CHANGE this to your VoxCeleb1 dataset path.
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.0003
    CHECKPOINT_PATH_MEL = 'melcnn_checkpoint.pth'
    NUM_CLASSES = 1000  # Adjust based on number of speakers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load the data.
    train_loader, val_loader, test_loader = get_data_loaders(DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # 1. Train the MelCNN to obtain speaker embeddings.
    mel_model = MelCNN(emb_dim=256, feature_dim=1024, dropout_rate=0.5).to(device)
    # For illustration, we attach a dummy classification loss on the embeddings.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mel_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
    print("Starting training of MelCNN...")
    mel_model = train_model(mel_model, train_loader, val_loader, optimizer, criterion, scheduler, device, NUM_EPOCHS, CHECKPOINT_PATH_MEL)

    # 2. Load pre-trained SincNet.
    SINCNET_CHECKPOINT = 'sincnet_checkpoint.pth'
    if os.path.exists(SINCNET_CHECKPOINT):
        sinc_model = load_sincnet(SINCNET_CHECKPOINT, device)
    else:
        print("SincNet checkpoint not found. Using DummySincNet.")
        sinc_model = DummySincNet(emb_dim=256).to(device)
    sinc_model.eval()

    # 3. Feature-level fusion and decision module training.
    fusion_model = FeatureFusion(emb_dim=256).to(device)
    decision_model = DecisionModule(emb_dim=256, num_classes=NUM_CLASSES, dropout_rate=0.5).to(device)
    optimizer_decision = optim.Adam(list(fusion_model.parameters()) + list(decision_model.parameters()), lr=LEARNING_RATE)
    criterion_decision = nn.CrossEntropyLoss()

    def train_fused_model(mel_model, sinc_model, fusion_model, decision_model, data_loader, optimizer, criterion, device):
        mel_model.eval()
        sinc_model.eval()
        fusion_model.train()
        decision_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                mel_emb = mel_model(inputs)
                # Here, simulate SincNet input (replace with your actual feature computation if available).
                dummy_input = torch.randn(inputs.size(0), 100).to(device)
                sinc_emb = sinc_model(dummy_input)
            fused_emb = fusion_model(mel_emb, sinc_emb)
            logits = decision_model(fused_emb)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    NUM_EPOCHS_FUSION = 10
    print("Training feature fusion and decision module...")
    for epoch in range(NUM_EPOCHS_FUSION):
        loss_fusion, acc_fusion = train_fused_model(mel_model, sinc_model, fusion_model, decision_model, train_loader, optimizer_decision, criterion_decision, device)
        print(f"Fusion Epoch {epoch}: Loss {loss_fusion:.4f}, Acc {acc_fusion:.4f}")

    # 4. Testing: Evaluate identification (n-class classification).
    print("Evaluating models on Test Set (Identification)...")
    mel_acc = evaluate_model(mel_model, test_loader, device)
    print(f"MelCNN Test Identification Accuracy: {mel_acc:.4f}")
    sinc_acc = evaluate_model(sinc_model, test_loader, device)
    print(f"SincNet Test Identification Accuracy: {sinc_acc:.4f}")

    fusion_model.eval()
    decision_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            mel_emb = mel_model(inputs)
            dummy_input = torch.randn(inputs.size(0), 100).to(device)
            sinc_emb = sinc_model(dummy_input)
            fused_emb = fusion_model(mel_emb, sinc_emb)
            logits = decision_model(fused_emb)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    fused_acc = correct / total
    print(f"Fused Model Test Identification Accuracy: {fused_acc:.4f}")

    # 5. Authentication evaluation: compute Equal Error Rate (EER).
    print("Evaluating Authentication (EER) for MelCNN...")
    eer_mel = evaluate_authentication(mel_model, test_loader, device)
    print(f"MelCNN Authentication EER: {eer_mel:.4f}")

    def fused_forward(inputs):
        mel_emb = mel_model(inputs)
        dummy_input = torch.randn(inputs.size(0), 100).to(device)
        sinc_emb = sinc_model(dummy_input)
        fused_emb = fusion_model(mel_emb, sinc_emb)
        return fused_emb

    def get_fused_embeddings(model_func, data_loader, device):
        embeddings_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                emb = model_func(inputs)
                embeddings_list.append(emb.cpu())
                labels_list.append(labels)
        return torch.cat(embeddings_list, dim=0), torch.cat(labels_list, dim=0)

    fused_embs, fused_labels = get_fused_embeddings(fused_forward, test_loader, device)
    genuine_scores = []
    imposter_scores = []
    fused_embs_norm = nn.functional.normalize(fused_embs, p=2, dim=1)
    sim_matrix = torch.matmul(fused_embs_norm, fused_embs_norm.t()).numpy()
    fused_labels_np = fused_labels.numpy()
    num_samples = fused_embs.shape[0]
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            if fused_labels_np[i] == fused_labels_np[j]:
                genuine_scores.append(sim_matrix[i, j])
            else:
                imposter_scores.append(sim_matrix[i, j])
    scores = genuine_scores + imposter_scores
    targets = [1] * len(genuine_scores) + [0] * len(imposter_scores)
    eer_fused = compute_eer(scores, targets)
    print(f"Fused Model Authentication EER: {eer_fused:.4f}")

if __name__ == '__main__':
    main()
    