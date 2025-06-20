import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models

# Load Pretrained Hybrid Model
from attention_weighted_mobileNet_ViT import HybridmodelReturn  # Assuming your model is saved in hybrid_model.py

# Load Pretrained Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridmodelReturn().to(device)
model.load_state_dict(torch.load("weights/attention_weighted_hybrid_mobileNet_ViT.pth", map_location=device))
model.eval()

# **Freeze ViT Features** (Skip its execution)
for param in model.vit.parameters():
    param.requires_grad = False

# **Define Grad-CAM Function**
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None

        # Hook for gradients
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Hook for activation maps
        def forward_hook(module, input, output):
            self.activation = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx):
        input_tensor = input_tensor.to(device)  # Ensure tensor is on CUDA
        print(f"Input tensor shape: {input_tensor.shape}")  # Debugging line
        output = self.model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()  # Get the predicted class

        self.model.zero_grad()
        output[:, predicted_class].backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activation, dim=1)

        cam = F.relu(cam)
        cam = cam[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))

        cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize
        return cam, predicted_class

# **Load Images**
image_dir = "OCTDL_split/test"
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure input matches ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
])


dataset = datasets.ImageFolder(image_dir, transform=transform)
class_names = dataset.classes  # Get class names from dataset
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# **Apply Grad-CAM & Save Outputs**
output_dir = "new_saved_visuals/hybrid_attention"
target_layer = model.cnn.feature_extractor[-1]  # Get last CNN layer for Grad-CAM
grad_cam = GradCAM(model, target_layer)

for i, (img, label) in enumerate(data_loader):
    img = img.to(device)  # Move to GPU
    print("Image Shape", img.shape)  # Debugging line
    # Generate Grad-CAM Heatmap
    heatmap, predicted_class_idx = grad_cam.generate_heatmap(img, label.item())
    predicted_class_name = class_names[predicted_class_idx]  # Get class name
    
    # Convert Tensor to NumPy Image
    img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()

    # Overlay Heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(np.uint8(255 * img_np), cv2.COLOR_RGB2BGR), 0.6, heatmap_colored, 0.4, 0)

    # **Plot and Save**
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title(f"Transformed Image ({predicted_class_name})")
    
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"Grad-CAM Heatmap ({predicted_class_name})")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay ({predicted_class_name})")

    # Add Color Bar
    plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    # Save image with predicted class name in filename
    plt.savefig(f"{output_dir}/visualization_{i}_{predicted_class_name}.png")
    plt.close()

print("Grad-CAM visualizations saved successfully!")