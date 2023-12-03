import torch.nn as nn
import torch

class MLPSharpeLoss(nn.Module):
    def __init__(self, input_dim=10, hidden_size=256, dropout=0.2, num_classes=1, **kwargs):
        super(MLPSharpeLoss, self).__init__()
        self.mlp = nn.Sequential(
            # nn.Dropout(p=dropout),
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)

class MLPRetLoss(nn.Module):
    def __init__(self, input_dim=10, hidden_size=256, dropout=0.2, num_classes=1, **kwargs):
        super(MLPRetLoss, self).__init__()
        self.mlp = nn.Sequential(
            # nn.Dropout(p=dropout),
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)

class MLPRegressionLoss(nn.Module):
    def __init__(self, input_dim=10, hidden_size=256, dropout=0.2, num_classes=1, **kwargs):
        super(MLPRegressionLoss, self).__init__()
        self.mlp = nn.Sequential(
            # nn.Dropout(p=dropout),
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.mlp(x)

class MLPBinaryClassificationLoss(nn.Module):
    def __init__(self, input_dim=10, hidden_size=256, dropout=0.2, num_classes=1, **kwargs):
        super(MLPBinaryClassificationLoss, self).__init__()
        self.mlp = nn.Sequential(
            # nn.Dropout(p=dropout),
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)