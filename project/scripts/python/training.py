#!/usr/bin/env python
"""
Training Module:
Trains a CNN for voice authentication on spectrogram inputs.
This script uses dummy data for demonstration.
It also saves the model checkpoint and training history for visualization.
Usage:
    python training.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Define a simple CNN model (assumes input spectrogram images of size 64x64).
class VoiceAuthCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(VoiceAuthCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def generate_dummy_data(num_samples=200, img_size=64, num_classes=10):
    # Create dummy spectrogram images and random labels.
    X = np.random.rand(num_samples, 1, img_size, img_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y

def train_model(epochs=20, batch_size=10, learning_rate=0.001):
    X, y = generate_dummy_data(num_samples=200)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = VoiceAuthCNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        history.append({"epoch": epoch+1, "loss": epoch_loss, "accuracy": epoch_acc})
    
    # Save model checkpoint.
    torch.save(model.state_dict(), "voice_auth_cnn.pth")
    print("Model saved to voice_auth_cnn.pth")
    
    # Save training history for visualization.
    history_df = pd.DataFrame(history)
    history_df.to_csv("training_history.csv", index=False)
    print("Training history saved to training_history.csv")

if __name__ == "__main__":
    train_model(epochs=20, batch_size=10, learning_rate=0.001)
