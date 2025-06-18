from torch import nn
from constants import DROPOUT_RATE


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # First layer: larger to capture more features
            nn.Linear(118, 128),
            nn.BatchNorm1d(128),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            # Second layer: gradual reduction
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            # Third layer: further reduction
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            # Output layer
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)
