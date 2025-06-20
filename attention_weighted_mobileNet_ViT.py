import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torchvision.models as models
import timm

# Define MobileNet Feature Extractor
class MobileNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.feature_extractor = mobilenet.features  # Extract convolutional layers

        # Reduce 960 feature maps to 512 feature maps with learnable weights
        self.conv_out = nn.Conv2d(960, 512, kernel_size=1)  
        self.relu = nn.ReLU()

        # Global Average Pooling to preserve key spatial features
        self.global_pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.feature_extractor(x)  # Shape: (Batch, 960, 7, 7)
        x = self.conv_out(x)  # Reduce to (Batch, 512, 7, 7)
        x = self.relu(x)  
        x = self.global_pool(x)  # Keeps spatial feature representations intact
        return x

# Define ViT Attention Extractor
class ViTAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=4)
        self.vit.head = nn.Identity()  # Remove final classification layer

    def forward(self, x):
        x = self.vit.forward_features(x)  # Get attention-refined token embeddings
        return x  # Shape: (Batch, 197, 768) → Represents global dependencies

# Fusion Model
class HybridModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.cnn = MobileNetFeatureExtractor()
        self.vit = ViTAttention()
        
        # Fusion Layer
        self.fc_fusion = nn.Linear(512 + 768, 1024)  # Combine CNN and ViT embeddings
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)  # Final classification layer
        )

    def forward(self, x):
        cnn_features = self.cnn(x)  # Extract CNN feature maps
        vit_features = self.vit(x)  # Extract ViT attention embeddings
        
        # Flatten CNN feature maps using Global Average Pooling
        cnn_features = torch.mean(cnn_features, dim=(2, 3))  # Shape: (Batch, 512)
        
        # Fusion
        x_fused = torch.cat([cnn_features, vit_features[:, 0, :]], dim=1)  # Merge CNN & ViT features
        x_out = self.fc_fusion(x_fused)
        return self.classifier(x_out)

# Instantiate and Move to GPU
# print(model)
def HybridmodelReturn():
    model = HybridModel(num_classes=7).to("cuda")
    return model
