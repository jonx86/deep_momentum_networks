import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_size=256, num_classes=1):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        out = self.mlp(x)
        return out