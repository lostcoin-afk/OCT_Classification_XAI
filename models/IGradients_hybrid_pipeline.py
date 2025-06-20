import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from torchvision.transforms.functional import to_pil_image
import matplotlib.cm as cm

# Assuming this is defined earlier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function to convert tensor to numpy image
def tensor_to_np(img_tensor):
    img_tensor = img_tensor.squeeze(0).cpu().detach()
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())  # Normalize
    return img_tensor.numpy()

# # Visualization with colorbar
# def visualize_with_colorbar(orig_img, attribution, idx, label, save_path):
#     orig_np = tensor_to_np(orig_img)
#     attr_np = attribution.squeeze().detach().cpu().numpy()

#     fig, axs = plt.subplots(1, 3, figsize=(14, 5))
#     axs[0].imshow(orig_np.transpose(1, 2, 0))
#     axs[0].set_title("Original Image")
#     axs[0].axis("off")

#     im = axs[1].imshow(attr_np, cmap='hot')
#     axs[1].set_title("Attribution Map")
#     axs[1].axis("off")
#     plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

#     axs[2].imshow(orig_np.transpose(1, 2, 0))
#     axs[2].imshow(attr_np, cmap='hot', alpha=0.5)
#     axs[2].set_title("Overlay")
#     axs[2].axis("off")

#     os.makedirs(save_path, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, f"hybrid_img_{idx}_class_{label}.png"))
#     plt.close()

def visualize_with_colorbar(orig_img, attribution, idx, label, save_path):
    orig_np = tensor_to_np(orig_img)
    attr_np = attribution.squeeze().detach().cpu().numpy()
    
    # Enhanced normalization
    # attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min())  # Normalization
    # attr_np = (attr_np - np.percentile(attr_np, 5)) / (np.percentile(attr_np, 95) - np.percentile(attr_np, 5))
    attr_np = np.clip(attr_np, 0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    axs[0].imshow(orig_np.transpose(1, 2, 0))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    im = axs[1].imshow(attr_np, cmap='coolwarm')  # Change colormap
    axs[1].set_title("Attribution Map")
    axs[1].axis("off")
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04, extend='both')

    axs[2].imshow(orig_np.transpose(1, 2, 0))
    axs[2].imshow(attr_np, cmap='coolwarm', alpha=0.7)  # Increase alpha
    axs[2].set_title("Overlay")
    axs[2].axis("off")

    os.makedirs(save_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"hybrid_img_{idx}_class_{label}.png"))
    plt.close()


def get_attributions(model, input_img, label_idx):
    model.eval()
    input_img = input_img.to(device).unsqueeze(0).requires_grad_()

    # Use Integrated Gradients
    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(input_img, target=label_idx, return_convergence_delta=True)
    
    # Sum over channels for visualization
    attributions = attributions.sum(dim=1, keepdim=True)  # Shape: (1, 1, H, W)
    return attributions

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from Hybrid_mobileNet_ViT import hybrid_model_return  # Import the hybrid model
# Define test transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure input matches ViT
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
])

test_dataset = datasets.ImageFolder(root="OCTDL_split/test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load model
model = hybrid_model_return()
model.load_state_dict(torch.load("weights/hybrid_mobileNet_ViT.pth", map_location=device))
model.to(device)

# Visualization loop
save_path = "new_saved_visuals/hybrid_pipeline/"

for idx, (image, label) in enumerate(test_loader):
    image = image.to(device)
    label = label.item()

    attribution = get_attributions(model, image.squeeze(0), label)
    visualize_with_colorbar(image.squeeze(0), attribution, idx, label, save_path)

print("✅ All HybridViT visualizations saved successfully!")
