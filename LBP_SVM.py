import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import cv2
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load 4-Class Dataset
train_dataset = datasets.ImageFolder(root='OCTDL_split/train', transform=transform)
test_dataset = datasets.ImageFolder(root='OCTDL_split/test', transform=transform)

# Create Data Loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Function to Extract LBP Features
def extract_lbp_features(image):
    image = image.to(device)  # Move image to GPU
    image = image.cpu().numpy().transpose((1, 2, 0))  # Convert tensor to numpy array
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    
    # Compute LBP features
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')  # P=8 neighbors, R=1 radius
    
    # Convert LBP to histogram
    hist, _ = np.histogram(lbp, bins=256, range=(0, 256))  
    return hist / np.sum(hist)  # Normalize histogram

# Convert Dataset to Feature Vectors
def dataset_to_features(dataset):
    features, labels = [], []
    for image, label in dataset:
        image = image.to(device)  # Move image to GPU
        features.append(extract_lbp_features(image))
        labels.append(label)
    return np.array(features), np.array(labels)

# Extract Features Using LBP
X_train, y_train = dataset_to_features(train_dataset)
X_test, y_test = dataset_to_features(test_dataset)

print(f"Feature Shape: {X_train.shape}, Labels Shape: {y_train.shape}")

# Initialize SVM Classifier for 4-Class Problem
svm_model = SVC(kernel='rbf', C=1.0)

# Train the Model
svm_model.fit(X_train, y_train)

# Test the Model
y_test_pred = svm_model.predict(X_test)

# Compute Evaluation Metrics
accuracy = accuracy_score(y_test, y_test_pred)
classification_metrics = classification_report(y_test, y_test_pred, output_dict=True)

# Set Visualization Folder
model_name = "lbp_svm"
visualization_dir = f"saved_plots/{model_name}"
os.makedirs(visualization_dir, exist_ok=True)  # Ensure visualization directory exists

# Save Metrics to JSON File
metrics_save_path = f"{visualization_dir}/evaluation_metrics.json"
with open(metrics_save_path, "w") as f:
    json.dump({"Accuracy": accuracy, "Classification_Report": classification_metrics}, f, indent=4)

print(f"✅ Evaluation metrics saved at: {metrics_save_path}")

# Display Metrics
print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_test_pred))

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
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig(f"{visualization_dir}/confusion_matrix.png")
plt.close()
print(f"✅ Confusion matrix saved at: {visualization_dir}/confusion_matrix.png")