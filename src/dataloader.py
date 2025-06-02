from torch.utils.data import Dataset, DataLoader
import pandas as pd
from preprocess import preprocess_data
from constants import DATASET_URL, SPLITS, BATCH_SIZE


print("Reading CSV Files..")
# Load and get preprocessed data
train_df = pd.read_csv(DATASET_URL + SPLITS["train"])
test_df = pd.read_csv(DATASET_URL + SPLITS["test"])

print("Applying preprocessing...")
X_train_processed, X_test_processed, y_train_encoded, y_test_encoded, _, _ = (
    preprocess_data(train_df, test_df)
)


# Define the torch dataset
class NSLKDDDataset(Dataset):
    def __init__(self, split: str):
        if split == "train":
            self.X = X_train_processed
            self.y = y_train_encoded
        elif split == "val":
            self.X = X_test_processed
            self.y = y_test_encoded

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X.iloc[idx].values, self.y[idx]


# Load the datasets
print("Loading dataset...")
train_dataset = NSLKDDDataset(split="train")
val_dataset = NSLKDDDataset(split="val")

# Creating dataloaders
print("Creating data loaders...")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
