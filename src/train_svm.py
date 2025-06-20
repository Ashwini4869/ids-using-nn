import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import time
from tqdm import tqdm

from dataloader import train_loader, test_loader


def convert_dataloader_to_numpy(dataloader):
    """Convert PyTorch DataLoader to numpy arrays."""
    X_list, y_list = [], []
    print("Converting DataLoader to numpy arrays...")
    for batch_X, batch_y in tqdm(dataloader):
        X_list.append(batch_X.numpy())
        y_list.append(batch_y.numpy())
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y


# Convert data to numpy format
print("Preparing training data...")
X_train, y_train = convert_dataloader_to_numpy(train_loader)
print("Preparing test data...")
X_test, y_test = convert_dataloader_to_numpy(test_loader)

# Initialize and train SVM
print("\nTraining SVM...")
svm = SVC(
    kernel='rbf',              # RBF kernel for non-linear decision boundary
    C=1.0,                    # Regularization parameter
    class_weight='balanced',  # Handle class imbalance
    random_state=42,         # For reproducibility
    # verbose=True
)

svm.fit(X_train, y_train)

# Save the model
model_path = f"models/svm_model_{time.strftime('%Y-%m-%d-%H-%M')}.joblib"
joblib.dump(svm, model_path)
print(f"\nModel saved to {model_path}")

# Evaluate on test set
print("\nEvaluating on test set...")
y_pred = svm.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nTest Results:")
print("-" * 50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
