import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import time
from tqdm import tqdm
import pandas as pd

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

# Initialize and train Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,         # Number of trees in the forest
    max_depth=None,           # Maximum depth of the trees (None for full depth)
    min_samples_split=2,      # Minimum samples required to split a node
    min_samples_leaf=1,       # Minimum samples required at each leaf node
    max_features='sqrt',      # Number of features to consider for best split
    bootstrap=True,           # Use bootstrap samples
    class_weight='balanced',  # Handle class imbalance
    n_jobs=-1,               # Use all available cores
    random_state=42          # For reproducibility
)

rf.fit(X_train, y_train)

# Save the model
model_path = f"models/rf_model_{time.strftime('%Y-%m-%d-%H-%M')}.joblib"
joblib.dump(rf, model_path)
print(f"\nModel saved to {model_path}")

# Evaluate on test set
print("\nEvaluating on test set...")
y_pred = rf.predict(X_test)

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

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': range(X_train.shape[1]),
    'importance': rf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print("-" * 50)
print(feature_importance.head(10).to_string(index=False))
