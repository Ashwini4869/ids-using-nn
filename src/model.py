import torch
from torch import nn
import torch.nn.functional as F
from constants import DROPOUT_RATE, HIDDEN_SIZE


class FeatureExtractor(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.bn1(x)
        return F.relu(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extraction paths
        self.path1 = FeatureExtractor(118, HIDDEN_SIZE)
        self.path2 = FeatureExtractor(118, HIDDEN_SIZE)

        # Combine features
        self.combine = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE),
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE / 2),
        )

        # Final classification
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.BatchNorm1d(HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE / 2),
            nn.Linear(HIDDEN_SIZE // 2, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with careful initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_l2_regularization(self):
        """Calculate L2 regularization on the same device as model"""
        l2_reg = 0.0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                l2_reg = l2_reg + (m.weight**2).sum()
        return l2_reg

    def forward(self, x):
        # Extract features through parallel paths
        f1 = self.path1(x)
        f2 = self.path2(x)

        # Combine features
        combined = torch.cat([f1, f2], dim=1)
        features = self.combine(combined)

        # Classification
        logits = self.classifier(features)
        return self.sigmoid(logits)
