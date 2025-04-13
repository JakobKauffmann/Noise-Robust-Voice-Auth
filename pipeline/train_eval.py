# train_eval.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_curve
import torch.nn.functional as F

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs, checkpoint_path):
    best_val_acc = 0.0
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            embeddings = model(inputs)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
        epoch_loss = running_loss / total
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch}/{num_epochs - 1}: Loss {epoch_loss:.4f}, Val Acc {val_acc:.4f}")
        if scheduler is not None:
            scheduler.step(epoch_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Improved validation accuracy. Saving checkpoint to {checkpoint_path}")
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_acc': best_val_acc}, checkpoint_path)
    return model

def evaluate_model(model, data_loader, device, classifier=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            embeddings = model(inputs)
            if classifier is not None:
                logits = classifier(embeddings)
                preds = logits.argmax(dim=1)
            else:
                # Placeholder: in your training you would attach a classification head.
                preds = torch.zeros_like(labels)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx = np.argmin(abs_diffs)
    eer = np.mean((fpr[idx], fnr[idx]))
    return eer

def evaluate_authentication(model, data_loader, device):
    model.eval()
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            emb = model(inputs)
            embeddings_list.append(emb.cpu())
            labels_list.append(labels)
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.t()).numpy()
    labels = labels.numpy()
    genuine_scores = []
    imposter_scores = []
    num_samples = embeddings.shape[0]
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            if labels[i] == labels[j]:
                genuine_scores.append(sim_matrix[i, j])
            else:
                imposter_scores.append(sim_matrix[i, j])
    scores = genuine_scores + imposter_scores
    target = [1] * len(genuine_scores) + [0] * len(imposter_scores)
    eer = compute_eer(scores, target)
    return eer

if __name__ == '__main__':
    print("train_eval.py module is ready for import and use.")
    