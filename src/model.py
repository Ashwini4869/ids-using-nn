from torch import nn
from constants import DROPOUT_RATE, HIDDEN_SIZE


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Input regularization
            nn.Dropout(DROPOUT_RATE/2),  # Light dropout on input
            # Single hidden layer with strong regularization
            nn.Linear(118, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            # Output layer
            nn.Linear(HIDDEN_SIZE, 1),
            nn.Sigmoid(),
        )
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)
