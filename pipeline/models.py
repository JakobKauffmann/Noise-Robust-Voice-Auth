# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

###########################################
# Mel Spectrogram CNN Model for Speaker Embedding
###########################################
class MelCNN(nn.Module):
    def __init__(self, emb_dim=256, feature_dim=1024, dropout_rate=0.5):
        super(MelCNN, self).__init__()
        # Input shape: (batch, 1, n_mels, time)
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Use AdaptiveAvgPool2d to obtain a fixed-size feature map.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, emb_dim)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        # Normalize the embedding vector.
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

###########################################
# Attention-based Feature Fusion Module
###########################################
class FeatureFusion(nn.Module):
    def __init__(self, emb_dim=256):
        super(FeatureFusion, self).__init__()
        self.attention_linear = nn.Linear(emb_dim, 1)

    def forward(self, mel_feat, sinc_feat):
        # Stack the two embeddings along a new dimension.
        feats = torch.stack([mel_feat, sinc_feat], dim=1)  # Shape: [batch, 2, emb_dim]
        attn_scores = self.attention_linear(feats)          # Shape: [batch, 2, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        fused_feat = torch.sum(feats * attn_weights, dim=1)   # Shape: [batch, emb_dim]
        return fused_feat

###########################################
# Decision Module: Fully Connected Classifier
###########################################
class DecisionModule(nn.Module):
    def __init__(self, emb_dim=256, num_classes=1000, dropout_rate=0.5):
        super(DecisionModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x):
        logits = self.fc(x)
        return logits

###########################################
# SincNet Helpers
###########################################
def load_sincnet(checkpoint_path, device):
    # Load your pre-trained SincNet model from the checkpoint.
    model = torch.load(checkpoint_path, map_location=device)
    model.eval()
    return model

# Dummy SincNet that mimics the interface.
class DummySincNet(nn.Module):
    def __init__(self, emb_dim=256):
        super(DummySincNet, self).__init__()
        self.fc = nn.Linear(100, emb_dim)  # Dummy input dimension.

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

if __name__ == '__main__':
    # Quick tests to verify model outputs.
    import torch
    mel_cnn = MelCNN(emb_dim=256).eval()
    dummy_input = torch.randn(8, 1, 40, 100)  # Example input: [batch, 1, n_mels, time]
    emb = mel_cnn(dummy_input)
    print("MelCNN embedding shape:", emb.shape)

    fusion = FeatureFusion(emb_dim=256)
    fused_emb = fusion(emb, emb)
    print("Fused embedding shape:", fused_emb.shape)

    decision = DecisionModule(emb_dim=256, num_classes=1000)
    logits = decision(fused_emb)
    print("Decision logits shape:", logits.shape)
    