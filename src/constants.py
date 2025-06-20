# Dataset
DATASET_URL = "hf://datasets/Mireu-Lab/NSL-KDD/"
SPLITS = {"train": "train.csv", "test": "test.csv"}
VALIDATION_SPLIT = 0.2  # Added validation split

# Hyperparameters
NUM_EPOCHS = 50
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.02
BATCH_SIZE = 64      # Increased for better stability
DROPOUT_RATE = 0.5
GRADIENT_CLIP = 0.5   # Reduced for more stable training

# Model Architecture
HIDDEN_SIZE = 32      # Single hidden layer size

# Paths
TEST_MODEL_PATH = "models/model_2025-06-19-17-39_best.pt"
