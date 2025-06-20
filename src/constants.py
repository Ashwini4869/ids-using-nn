# Dataset
DATASET_URL = "hf://datasets/Mireu-Lab/NSL-KDD/"
SPLITS = {"train": "train.csv", "test": "test.csv"}
VALIDATION_SPLIT = 0.2  # Added validation split

# Hyperparameters
NUM_EPOCHS = 100  # Reduced epochs since we have better sampling
LEARNING_RATE = 0.001  # Keep initial LR
WEIGHT_DECAY = 0.001  # Reduced weight decay due to parallel paths
BATCH_SIZE = 128  # Increased batch size for stable batch norm
DROPOUT_RATE = 0.3  # Reduced dropout due to parallel paths
GRADIENT_CLIP = 0.5  # Keep gradient clip

# Learning Rate Scheduler
LR_PATIENCE = 3  # Reduced patience for faster adaptation
LR_FACTOR = 0.2  # More aggressive LR reduction
LR_MIN = 1e-6  # Minimum learning rate

# Early Stopping
EARLY_STOP_PATIENCE = 10  # Reduced patience due to better convergence

# Model Architecture
HIDDEN_SIZE = 18  # Increased for parallel paths

# Paths
TEST_MODEL_PATH = "models/model_2025-06-20-18-02_best.pt"
