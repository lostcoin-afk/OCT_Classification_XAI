import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

import torch.nn as nn
import torchvision.models as models
import timm
from resnet_pretrained import resnetReturn  # Import the hybrid model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

### **Step 1: Data Loading & Transformations** ###
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure input matches ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
])

train_dataset = datasets.ImageFolder(root='OCTDL_split/train', transform=transform)
test_dataset = datasets.ImageFolder(root='OCTDL_split/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ### **Step 2: Initialize Model, Loss, Optimizer** ###
model_name = "resnet_pretrained"
model_save_path = f"weights/{model_name}.pth"
visualization_dir = "saved_plots/resnet_pretrained"
os.makedirs(visualization_dir, exist_ok=True)  # Ensure visualization directory exists

model = resnetReturn().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)

# **Save the best model dynamically**
best_loss = float("inf")

## **Step 3: Training Loop with Loss Visualization** # ##
num_epochs = 50
train_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()  # Ensure full backpropagation through MobileNet & ViT
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    # **Save model only if training loss improves**
    if avg_train_loss < best_loss:
        best_loss = avg_train_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ Saved best model (Train Loss: {best_loss:.4f})")

print(f"Best model saved at: {model_save_path}")

### **Step 4: Plot & Save Training Loss** ###
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Loss Over Epochs")
plt.grid()
plt.savefig(f"{visualization_dir}/training_loss_plot.png")
plt.close()

### **Step 5: Load Best Model & Evaluate** ###
model.load_state_dict(torch.load(f"{model_save_path}", map_location=device))
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Compute Evaluation Metrics
accuracy = accuracy_score(y_true, y_pred)
classification_metrics = classification_report(y_true, y_pred, output_dict=True)

# Save Metrics to JSON File
metrics_save_path = f"{visualization_dir}/evaluation_metrics.json"
with open(metrics_save_path, "w") as f:
    json.dump({"Accuracy": accuracy, "Classification_Report": classification_metrics}, f, indent=4)

print(f"✅ Evaluation metrics saved at: {metrics_save_path}")

# Display Metrics
print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_true, y_pred))

### **Step 6: Plot & Save Evaluation Metrics (Precision, Recall, F1-score)**
class_names = list(classification_metrics.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'

precision_vals = [classification_metrics[class_name]["precision"] for class_name in class_names]
recall_vals = [classification_metrics[class_name]["recall"] for class_name in class_names]
f1_vals = [classification_metrics[class_name]["f1-score"] for class_name in class_names]

plt.figure(figsize=(10, 5))
plt.bar(class_names, precision_vals, color='skyblue', label="Precision")
plt.bar(class_names, recall_vals, color='orange', label="Recall")
plt.bar(class_names, f1_vals, color='red', label="F1-score")
plt.xlabel("Classes")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1-score per Class")
plt.legend()
plt.savefig(f"{visualization_dir}/precision_recall_f1_plot.png")
plt.close()
print(f"✅ Precision-Recall-F1 plot saved at: {visualization_dir}/precision_recall_f1_plot.png")

### **Step 7: Plot & Save Confusion Matrix**
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig(f"{visualization_dir}/confusion_matrix.png")
plt.close()
print(f"✅ Confusion matrix saved at: {visualization_dir}/confusion_matrix.png")