import os
import torch
from torch.nn import BCELoss
from torch.optim import Adam
from tqdm import tqdm
import time

from dataloader import train_loader, val_loader
from model import Model
from constants import NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
from utils import setup_log_file, log_metrics

log_file = setup_log_file()
# Select CUDA for training if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device CUDA")
else:
    device = torch.device("cpu")
    print("Using device CPU")

# Initialize model, loss function and optimizer
model = Model().to(device)
criterion = BCELoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Training/Validation Pipeline
print("Starting Training...")
for epoch in tqdm(range(NUM_EPOCHS)):
    # Training
    train_loss = 0
    model.train()
    for x, y in train_loader:
        x = x.float()
        x, y = x.to(device), y.to(device)
        pred = model.forward(x)
        pred = pred.squeeze()
        y = y.float()
        loss = criterion(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    print(f"Epoch {epoch}/ {NUM_EPOCHS}  Training Loss: {train_loss}")

    # Validation
    valid_loss = 0
    model.eval()
    for x, y in val_loader:
        x = x.float()
        x, y = x.to(device), y.to(device)
        pred = model.forward(x)
        pred = pred.squeeze()
        y = y.float()
        loss = criterion(pred, y)
        valid_loss += loss.item()
    print(f"Validation Loss: {valid_loss}")

    log_metrics(log_file, epoch, train_loss, valid_loss)

# Save the model
trained_model_path = f"models/model_{time.strftime('%Y-%m-%d-%H-%M')}.pt"
torch.save(model.state_dict(), trained_model_path)
print(f"Model saved to {trained_model_path}")
