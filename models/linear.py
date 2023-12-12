import torch.nn as nn
import torch

class LinearSharpeLoss(nn.Module):
    def __init__(self, input_dim=60, **kwargs):
        super(LinearSharpeLoss, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)
class LinearRetLoss(nn.Module):
    def __init__(self, input_dim=60, **kwargs):
        super(LinearRetLoss, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)

class LinearRegressionLoss(nn.Module):
    def __init__(self, input_dim=60, **kwargs):
        super(LinearRegressionLoss, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1),
        )

    def forward(self, x):
        return self.mlp(x)

class LinearBinaryClassificationLoss(nn.Module):
    def __init__(self, input_dim=60, **kwargs):
        super(LinearBinaryClassificationLoss, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)

class LinearSharpeLossCustom(nn.Module):
    def __init__(self, input_dim=60, **kwargs):
        super(LinearSharpeLossCustom, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)