import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import cv2
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torchvision.models as models

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Pretrained GoogleNet Model (Remove Fully Connected Layer)
googlenet = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
googlenet.fc = torch.nn.Identity()  # Remove FC layer for feature extraction
googlenet = googlenet.to(device)
googlenet.eval()

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # GoogleNet requires 299x299 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Dataset
train_dataset = datasets.ImageFolder(root="OCTDL_split/train", transform=transform)
test_dataset = datasets.ImageFolder(root="OCTDL_split/test", transform=transform)

# Create Data Loaders
# import torch.utils.data.DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Extract Features for Random Forest Training
def extract_features(model, dataloader):
    features, labels = [], []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            features.extend(outputs.cpu().numpy())
            labels.extend(targets.numpy())
    return np.array(features), np.array(labels)

# Get features from training and test sets
train_features, train_labels = extract_features(googlenet, train_loader)
test_features, test_labels = extract_features(googlenet, test_loader)

# Normalize features before feeding into Random Forest
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_features, train_labels)

# Evaluate on Test Set
predictions = rf.predict(test_features)
accuracy = accuracy_score(test_labels, predictions)
classification_metrics = classification_report(test_labels, predictions, output_dict=True)

# Set Visualization Folder
model_name = "googlenet_rf"
visualization_dir = f"saved_plots/{model_name}"
os.makedirs(visualization_dir, exist_ok=True)  # Ensure visualization directory exists

# Save Metrics to JSON File
metrics_save_path = f"{visualization_dir}/evaluation_metrics.json"
with open(metrics_save_path, "w") as f:
    json.dump({"Accuracy": accuracy, "Classification_Report": classification_metrics}, f, indent=4)

print(f"✅ Evaluation metrics saved at: {metrics_save_path}")

# Display Metrics
print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(test_labels, predictions))

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
plt.xticks(rotation=45)
plt.savefig(f"{visualization_dir}/precision_recall_f1_plot.png")
plt.close()
print(f"✅ Precision-Recall-F1 plot saved at: {visualization_dir}/precision_recall_f1_plot.png")

### **Step 7: Plot & Save Confusion Matrix**
conf_matrix = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig(f"{visualization_dir}/confusion_matrix.png")
plt.close()
print(f"✅ Confusion matrix saved at: {visualization_dir}/confusion_matrix.png")