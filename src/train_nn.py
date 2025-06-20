import os
import torch
from torch.nn import BCELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time

from dataloader import train_loader, val_loader, class_weights
from model import Model
from constants import (
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRADIENT_CLIP,
    LR_PATIENCE,
    LR_FACTOR,
    LR_MIN,
    EARLY_STOP_PATIENCE,
)
from utils import setup_log_file, log_metrics

log_file = setup_log_file()

# Select CUDA for training if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model, loss function and optimizer
model = Model().to(device)
criterion = BCELoss(reduction="none")  # Use none reduction for weighted loss
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    min_lr=LR_MIN,
)

# Move class weights to device
class_weights = class_weights.to(device)

# Early stopping parameters
best_valid_loss = float("inf")
patience_counter = 0
best_model_path = None


def compute_weighted_loss(pred, target, weights):
    losses = criterion(pred, target)
    weighted_losses = losses * weights[target.long()]
    return weighted_losses.mean()


# Training/Validation Pipeline
print("Starting Training...")
for epoch in tqdm(range(NUM_EPOCHS)):
    # Training
    model.train()
    train_loss = 0
    for x, y in train_loader:
        # Move batch to device
        x = x.float().to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x).squeeze()
        y = y.float()

        # Compute loss with L2 regularization
        loss = compute_weighted_loss(pred, y, class_weights)
        loss = (
            loss + 0.01 * model.get_l2_regularization()
        )  # L2 reg will be on same device as model

        # Backpropagation with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            # Move batch to device
            x = x.float().to(device)
            y = y.to(device)
            pred = model(x).squeeze()
            y = y.float()
            loss = compute_weighted_loss(pred, y, class_weights)
            valid_loss += loss.item()
    valid_loss /= len(val_loader)

    # Learning rate scheduling
    old_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(valid_loss)
    new_lr = optimizer.param_groups[0]["lr"]
    if new_lr != old_lr:
        print(f"\nLearning rate decreased from {old_lr:.6f} to {new_lr:.6f}")

    # Logging
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    print(f"Training Loss: {train_loss:.6f}")
    print(f"Validation Loss: {valid_loss:.6f}")
    print(f"Learning Rate: {new_lr:.6f}")
    log_metrics(log_file, epoch, train_loss, valid_loss)

    # Save best model and early stopping
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        patience_counter = 0
        # Remove previous best model if it exists
        if best_model_path and os.path.exists(best_model_path):
            os.remove(best_model_path)
        # Save new best model
        best_model_path = f"models/model_{time.strftime('%Y-%m-%d-%H-%M')}_best.pt"
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with validation loss: {valid_loss:.6f}")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
        elif patience_counter >= EARLY_STOP_PATIENCE // 2:
            print(f"Warning: No improvement for {patience_counter} epochs")

print(f"\nBest validation loss: {best_valid_loss:.6f}")
print(f"Best model saved at: {best_model_path}")
