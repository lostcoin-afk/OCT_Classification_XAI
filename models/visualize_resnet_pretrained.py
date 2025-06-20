import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import IntegratedGradients
from PIL import Image
import os

# Load your custom ResNet model
from resnet_pretrained import resnetReturn  # Replace with actual model import

# Load model and set to eval
model = resnetReturn()
model.load_state_dict(torch.load("weights/resnet50_best.pth"))  # Load your trained model weights
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader
test_dataset = datasets.ImageFolder(root="OCTDL_split/test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Integrated Gradients instance
ig = IntegratedGradients(model)

# Visualization function
def visualize_and_save_attributions(input_tensor, label, idx, orig_img_pil, save_path="savedVisualization/resnet50"):
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad = True

    baseline = torch.zeros_like(input_tensor).to(device)

    # Forward pass
    output = model(input_tensor)
    pred_class = torch.argmax(output, dim=1).item()

    # Get attributions
    attributions, _ = ig.attribute(input_tensor, baseline, target=pred_class, return_convergence_delta=True)
    attribution = attributions.squeeze().detach().cpu().numpy()
    attribution = np.mean(np.abs(attribution), axis=0)

    # Normalize and convert to image
    attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
    attribution_img = Image.fromarray(np.uint8(255 * attribution)).resize(orig_img_pil.size, resample=Image.BILINEAR)
    attribution_np = np.array(attribution_img)

    import matplotlib.cm as cm

    # Plot and save
    os.makedirs(save_path, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))

    # Original Image
    axs[0].imshow(orig_img_pil)
    axs[0].axis('off')
    axs[0].set_title("Original Image")

    # Attribution Map with Colorbar
    im = axs[1].imshow(attribution_np, cmap='hot')
    axs[1].axis('off')
    axs[1].set_title("Attribution Map")

    # Add colorbar next to the attribution map
    cbar = plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label('Attribution Intensity', rotation=270, labelpad=15)

    # Overlay Attribution on Original Image
    axs[2].imshow(orig_img_pil)
    axs[2].imshow(attribution_np, cmap='hot', alpha=0.6)
    axs[2].axis('off')
    axs[2].set_title("Overlay")

    # Save Figure
    save_file = os.path.join(save_path, f"img_{idx}_class_{label}.png")
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()



# Loop through test set and save attributions
for idx, (input_tensor, label) in enumerate(test_loader):
    # Get corresponding original image (before transform)
    img_path, _ = test_dataset.samples[idx]
    orig_img_pil = Image.open(img_path).convert('RGB')

    visualize_and_save_attributions(input_tensor, label.item(), idx, orig_img_pil)

print("✅ All Integrated Gradients visualizations saved successfully!")
