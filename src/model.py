from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(118, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)
