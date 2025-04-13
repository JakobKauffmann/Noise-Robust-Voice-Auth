# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

###########################################
# Mel Spectrogram CNN Model for Speaker Embedding
###########################################
class MelCNN(nn.Module):
    def __init__(self, emb_dim=256, feature_dim=1024, dropout_rate=0.5):
        """
        A simple CNN that accepts Mel spectrograms (shape: [B, 1, n_mels, time])
        and outputs a 256-D embedding.
        """
        super(MelCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Use adaptive pooling so that the output is fixed.
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
        # Stack along a new dimension: shape becomes [B, 2, emb_dim]
        feats = torch.stack([mel_feat, sinc_feat], dim=1)
        attn_scores = self.attention_linear(feats)  # [B, 2, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        fused_feat = torch.sum(feats * attn_weights, dim=1)
        return fused_feat

###########################################
# Decision Module (Classifier)
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
# (For compatibility, dummy SincNet loader remains here.)
###########################################
def load_dummy_sincnet(device):
    # In case no checkpoint is found, use a dummy.
    class DummySincNet(nn.Module):
        def __init__(self, emb_dim=256):
            super(DummySincNet, self).__init__()
            self.fc = nn.Linear(100, emb_dim)
        def forward(self, x):
            x = self.fc(x)
            return F.normalize(x, p=2, dim=1)
    model = DummySincNet(emb_dim=256)
    model.to(device)
    model.eval()
    return model

if __name__ == '__main__':
    # Quick test
    mel = MelCNN(emb_dim=256)
    dummy_input = torch.randn(8, 1, 40, 100)
    emb = mel(dummy_input)
    print("MelCNN embedding shape:", emb.shape)
    
    fusion = FeatureFusion(emb_dim=256)
    fused_emb = fusion(emb, emb)
    print("Fused embedding shape:", fused_emb.shape)
    
    decision = DecisionModule(emb_dim=256, num_classes=1000)
    logits = decision(fused_emb)
    print("Decision logits shape:", logits.shape)
    
