import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Path configuration
DATA_DIR = "D:/malware_dataset"
TRAIN_LABELS = os.path.normpath(os.path.join(DATA_DIR, "trainLabels.csv"))
TRAIN_HEX_DIR = os.path.normpath(os.path.join(DATA_DIR, "train/train"))
TEST_HEX_DIR = os.path.normpath(os.path.join(DATA_DIR, "test/test"))

# Load labels
labels = pd.read_csv(TRAIN_LABELS)
print("Labels loaded:", labels.head())

# Feature extraction
def extract_features(file_path):
    """Extract features from hexadecimal files."""
    with open(file_path, 'r', errors='ignore') as f:
        hex_data = f.read()
    file_size = len(hex_data)
    unique_chars = len(set(hex_data))
    return [file_size, unique_chars]

# Build feature matrix
features = []
for file_id in tqdm(labels['Id'], desc="Extracting features"):
    file_path = os.path.normpath(os.path.join(TRAIN_HEX_DIR, f"{file_id}.bytes"))
    features.append(extract_features(file_path))

features = np.array(features)
print("Feature matrix shape:", features.shape)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels['Class'])
# Convert numeric class labels to strings
target_names = [str(cls) for cls in label_encoder.classes_]

# Split data
X_train, X_val, y_train, y_val = train_test_split(features, y, test_size=0.2, random_state=42)
print("Training and validation sets prepared.")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define classifiers
classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}


# Train and evaluate models
results = {}
for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    results[name] = {
        "accuracy": acc,
        "classification_report": classification_report(y_val, y_pred, target_names=target_names, zero_division=0),
        "confusion_matrix": confusion_matrix(y_val, y_pred)
    }


# Display results
for name, result in results.items():
    print(f"\nModel: {name}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Classification Report:")
    print(result["classification_report"])

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(result["confusion_matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Feature importance (for tree-based models)
rf_importances = classifiers["RandomForest"].feature_importances_
features = ["FileSize", "UniqueChars"]

plt.figure(figsize=(10, 8))
sns.barplot(x=rf_importances, y=features)
plt.title("Feature Importance - RandomForest")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Class distribution
sns.countplot(data=labels, x="Class", order=labels['Class'].value_counts().index)
plt.title("Malware Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# T-SNE Visualization
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_train_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y_train, palette="tab10", legend="full")
plt.title("T-SNE Visualization of Malware Features")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

# Save model comparison results
final_results = []
for name, result in results.items():
    final_results.append({
        "Model": name,
        "Accuracy": result["accuracy"]
    })

final_df = pd.DataFrame(final_results)
final_df.to_csv("model_comparison_results.csv", index=False)
print("Model comparison results saved!")


# Additional Visual Insights
# Class Distribution in Train and Validation Sets
train_classes = pd.Series(y_train).map(dict(enumerate(label_encoder.classes_)))
val_classes = pd.Series(y_val).map(dict(enumerate(label_encoder.classes_)))

plt.figure(figsize=(10, 5))
sns.countplot(x=train_classes, order=label_encoder.classes_, palette="Set1", alpha=0.7)
sns.countplot(x=val_classes, order=label_encoder.classes_, palette="Set2", alpha=0.5)
plt.title("Class Distribution: Train vs Validation")
plt.xlabel("Class")
plt.ylabel("Count")
plt.legend(["Train", "Validation"])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Test and submission
test_files = [f for f in os.listdir(TEST_HEX_DIR) if f.endswith(".bytes")]
test_features = [extract_features(os.path.normpath(os.path.join(TEST_HEX_DIR, file_id))) for file_id in test_files]
test_features = np.array(test_features)
test_features_scaled = scaler.transform(test_features)

best_model = classifiers["RandomForest"]
test_predictions = best_model.predict(test_features_scaled)
test_classes = label_encoder.inverse_transform(test_predictions)

submission = pd.DataFrame({'Id': [os.path.splitext(f)[0] for f in test_files], 'Class': test_classes})
submission.to_csv("submission_from_comparison.csv", index=False)
print("Submission file saved!")
