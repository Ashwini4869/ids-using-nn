import os
import torch
from torch.nn import BCELoss
from torch.optim import AdamW
from tqdm import tqdm
import time

from dataloader import train_loader, val_loader, class_weights
from model import Model
from constants import NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, GRADIENT_CLIP
from utils import setup_log_file, log_metrics

log_file = setup_log_file()
# Select CUDA for training if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device CUDA")
else:
    device = torch.device("cpu")
    print("Using device CPU")

# Move class weights to device
class_weights = class_weights.to(device)

# Initialize model, loss function and optimizer
model = Model().to(device)
criterion = BCELoss(reduction='none')  # Use none reduction for weighted loss
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Early stopping parameters
patience = 10  # Increased patience
best_valid_loss = float('inf')
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
    train_loss = 0
    model.train()
    for x, y in train_loader:
        x = x.float()
        x, y = x.to(device), y.to(device)
        pred = model.forward(x)
        pred = pred.squeeze()
        y = y.float()
        
        # Compute weighted loss
        loss = compute_weighted_loss(pred, y, class_weights)
        
        # Backpropagation with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f"\nEpoch {epoch}/ {NUM_EPOCHS}  Training Loss: {train_loss:.6f}")

    # Validation
    valid_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.float()
            x, y = x.to(device), y.to(device)
            pred = model.forward(x)
            pred = pred.squeeze()
            y = y.float()
            loss = compute_weighted_loss(pred, y, class_weights)
            valid_loss += loss.item()
    valid_loss /= len(val_loader)
    print(f"Validation Loss: {valid_loss:.6f}")

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
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

print(f"\nBest validation loss: {best_valid_loss:.6f}")
print(f"Best model saved at: {best_model_path}")
