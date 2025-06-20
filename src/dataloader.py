from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import pandas as pd
import numpy as np
import torch
from preprocess import preprocess_data
from constants import DATASET_URL, SPLITS, BATCH_SIZE, VALIDATION_SPLIT


print("Reading CSV Files..")
# Load and get preprocessed data
train_df = pd.read_csv(DATASET_URL + SPLITS["train"])
test_df = pd.read_csv(DATASET_URL + SPLITS["test"])

print("Applying preprocessing...")
X_train_processed, X_test_processed, y_train_encoded, y_test_encoded, _, _ = (
    preprocess_data(train_df, test_df)
)

# Calculate class weights for balanced training
train_counts = np.bincount(y_train_encoded)
total_samples = len(y_train_encoded)
class_weights = torch.FloatTensor(total_samples / (len(train_counts) * train_counts))
print("Class weights:", class_weights.numpy())

# Calculate sample weights for WeightedRandomSampler
sample_weights = torch.FloatTensor([class_weights[t] for t in y_train_encoded])


# Define the torch dataset
class NSLKDDDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X.iloc[idx].values, self.y[idx]


# Create datasets
print("Loading dataset...")
full_train_dataset = NSLKDDDataset(X_train_processed, y_train_encoded)
test_dataset = NSLKDDDataset(X_test_processed, y_test_encoded)

# Split training data into train and validation
train_size = int((1 - VALIDATION_SPLIT) * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create sampler for training data
train_sampler = WeightedRandomSampler(
    weights=sample_weights[train_dataset.indices],
    num_samples=len(train_dataset),
    replacement=True,
)

# Creating dataloaders
print("Creating data loaders...")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
