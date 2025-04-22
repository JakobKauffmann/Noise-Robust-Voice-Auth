# models/mobilenet_embedding.py
import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV2Embedding(nn.Module):
    """
    Extracts embeddings using a pre-trained MobileNetV2 model.
    Input: (B, 3, H, W) image tensor (e.g., B, 3, 224, 224)
    Output: (B, emb_dim) embedding tensor (e.g., B, 256)
    """
    def __init__(self, emb_dim: int = 256, pretrained: bool = True, freeze: bool = True):
        """
        Initializes the MobileNetV2 embedding extractor.

        Args:
            emb_dim (int): The desired output embedding dimension.
            pretrained (bool): Whether to load weights pre-trained on ImageNet.
            freeze (bool): Whether to freeze the weights of the convolutional layers (features).
                           Set to False to fine-tune the feature extractor.
        """
        super().__init__()
        # Load the MobileNetV2 model from torchvision
        # Use weights=MobileNet_V2_Weights.IMAGENET1K_V1 for current PyTorch versions
        try:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.mobilenet_v2(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            print("Warning: Using legacy pretrained=True for MobileNetV2. Consider updating torchvision.")
            backbone = models.mobilenet_v2(pretrained=pretrained)


        # Freeze the feature extraction layers if requested
        if freeze:
            print("Freezing MobileNetV2 feature layers.")
            for param in backbone.features.parameters():
                param.requires_grad = False
        else:
            print("MobileNetV2 feature layers will be fine-tuned.")

        # Use the feature extractor part of MobileNetV2
        self.features = backbone.features

        # Use Adaptive Average Pooling to get a fixed-size output regardless of input image size variations
        # Output size (1, 1) collapses spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Get the number of output features from the backbone's classifier input
        # (MobileNetV2's classifier input is the number of features after pooling)
        num_features = backbone.classifier[1].in_features # Typically 1280 for MobileNetV2

        # Define the projection layer to map pooled features to the desired embedding dimension
        self.proj = nn.Linear(num_features, emb_dim)

        # Optional: Initialize the projection layer weights (e.g., Xavier initialization)
        # nn.init.xavier_uniform_(self.proj.weight)
        # nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedding extractor.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Output embedding tensor of shape (B, emb_dim).
        """
        # 1. Extract features using the convolutional backbone
        x = self.features(x) # Shape: (B, num_features_before_pool, H', W')

        # 2. Apply adaptive average pooling
        x = self.pool(x) # Shape: (B, num_features_before_pool, 1, 1)

        # 3. Flatten the pooled features
        x = torch.flatten(x, 1) # Shape: (B, num_features_before_pool)

        # 4. Project to the final embedding dimension
        emb = self.proj(x) # Shape: (B, emb_dim)

        return emb

# # models/mobilenet_embedding.py
# import torch.nn as nn
# import torchvision.models as models

# class MobileNetV2Embedding(nn.Module):
#     """
#     (B,3,224,224) -> (B,256)
#     """
#     def __init__(self, emb_dim: int = 256, pretrained: bool = True, freeze: bool = True):
#         super().__init__()
#         backbone = models.mobilenet_v2(pretrained=pretrained)
#         if freeze:
#             for p in backbone.features.parameters():
#                 p.requires_grad = False
#         self.features = backbone.features
#         self.pool     = nn.AdaptiveAvgPool2d(1)
#         self.proj     = nn.Linear(1280, emb_dim)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         emb = self.proj(x)
#         return emb
