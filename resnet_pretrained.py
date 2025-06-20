import torch
import torch.nn as nn
import torchvision.models as models

# Load Pretrained ResNet-50 Model
def resnetReturn():
    # Load ResNet-50 with default weights
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Modify Fully Connected Layer for 7-Class Classification
    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(num_features, 256),  # Reduce to 256 features
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, 7)  # **Updated: 7-class classification**
    )  # No Softmax for CrossEntropyLoss (PyTorch applies it internally)

    # Freeze all layers except the last three
    for param in list(resnet50.parameters())[:-6]:  
        param.requires_grad = False

    return resnet50