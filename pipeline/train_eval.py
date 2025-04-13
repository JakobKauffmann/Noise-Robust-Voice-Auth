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
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for features, labels, _ in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            embeddings = model(features)
            # For training the Mel CNN, we attach a simple classifier head later.
            # Here, we use a dummy target by treating the embeddings as logits.
            # Replace this loss with your preferred training loss if desired.
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
            # Dummy accuracy; in practice use a classifier head.
            total += labels.size(0)
        epoch_loss = running_loss / total
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss {epoch_loss:.4f}, Val Acc {val_acc:.4f}")
        if scheduler is not None:
            scheduler.step(epoch_loss)
        # Checkpoint saving
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': max(best_val_acc, val_acc)
        }, checkpoint_path)
    return model

def evaluate_model(model, data_loader, device, classifier=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels, _ in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            embeddings = model(features)
            # If classifier provided, use it; otherwise, dummy evaluation.
            if classifier is not None:
                logits = classifier(embeddings)
                preds = logits.argmax(dim=1)
            else:
                preds = torch.zeros_like(labels)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

def compute_eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer = np.mean((fpr[np.nanargmin(np.abs(fpr - fnr))], fnr[np.nanargmin(np.abs(fpr - fnr))]))
    return eer

def evaluate_authentication(model, data_loader, device):
    model.eval()
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for features, labels, _ in data_loader:
            features = features.to(device)
            emb = model(features)
            embeddings_list.append(emb.cpu())
            labels_list.append(labels)
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.t()).numpy()
    labels = labels.numpy()
    genuine = []
    imposter = []
    N = embeddings.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            if labels[i] == labels[j]:
                genuine.append(sim_matrix[i,j])
            else:
                imposter.append(sim_matrix[i,j])
    scores = genuine + imposter
    targets = [1]*len(genuine) + [0]*len(imposter)
    eer = compute_eer(scores, targets)
    return eer

if __name__ == '__main__':
    print("train_eval.py loaded.")
