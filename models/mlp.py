import torch.nn as nn
import torch

# class SimpleMLP(nn.Module):
#     def __init__(self, input_dim=10, hidden_size=256, dropout=0.2, num_classes=1, **kwargs):
#         super(SimpleMLP, self).__init__()
#         self.mlp = nn.Sequential(
#             # nn.Flatten(),
#             nn.Linear(input_dim, hidden_size),
#             nn.Tanh(),
#             nn.Dropout(p=dropout),
#             nn.Linear(hidden_size, num_classes),
#             nn.Tanh(),
#             nn.Dropout(p=dropout),
#         )
#
#     def forward(self, x):
#         return self.mlp(x)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_size=256, dropout=0.2, num_classes=1, **kwargs):
        super(SimpleMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_classes = num_classes

        self.linear1 = nn.Linear(self.input_dim, self.hidden_size)
        self.dropout1 = nn.Dropout(p=self.dropout)

        self.linear2 = nn.Linear(self.hidden_size, self.num_classes)
        self.dropout2 = nn.Dropout(p=self.dropout)

    def forward(self, input):
        output = self.dropout1(torch.tanh(self.linear1(input)))
        output = self.dropout2(torch.tanh(self.linear2(output)))
        return output

class MLP(nn.Module):
    def __init__(self, input_dim=10, hidden_size=256, dropout=0.2, num_classes=1, **kwarg):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_size),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

class MLPSharpeLoss(nn.Module):
    def __init__(self, input_dim=10, hidden_size=256, dropout=0.2, num_classes=1, **kwargs):
        super(MLPSharpeLoss, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_classes = num_classes

        self.linear1 = nn.Linear(self.input_dim, self.hidden_size)
        self.dropout1 = nn.Dropout(p=self.dropout)

        self.linear2 = nn.Linear(self.hidden_size, self.num_classes)
        self.dropout2 = nn.Dropout(p=self.dropout)

    def forward(self, input):
        output = self.dropout1(torch.tanh(self.linear1(input)))
        output = self.dropout2(torch.tanh(self.linear2(output)))
        return output

class MLPRetLoss(nn.Module):
    def __init__(self, input_dim=10, hidden_size=256, dropout=0.2, num_classes=1, **kwargs):
        super(MLPRetLoss, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_classes = num_classes

        self.linear1 = nn.Linear(self.input_dim, self.hidden_size)
        self.dropout1 = nn.Dropout(p=self.dropout)

        self.linear2 = nn.Linear(self.hidden_size, self.num_classes)
        self.dropout2 = nn.Dropout(p=self.dropout)

    def forward(self, input):
        output = self.dropout1(torch.tanh(self.linear1(input)))
        output = self.dropout2(torch.tanh(self.linear2(output)))
        return output

class MLPRegressionLoss(nn.Module):
    def __init__(self, input_dim=10, hidden_size=256, dropout=0.2, num_classes=1, **kwargs):
        super(MLPRegressionLoss, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_classes = num_classes

        self.linear1 = nn.Linear(self.input_dim, self.hidden_size)
        self.dropout1 = nn.Dropout(p=self.dropout)

        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout2 = nn.Dropout(p=self.dropout)

        self.linear3 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, input):
        output = self.dropout1(torch.tanh(self.linear1(input)))
        output = self.dropout2(torch.tanh(self.linear2(output)))
        output = self.linear3(output)
        return output

class MLPBinaryClassificationLoss(nn.Module):
    def __init__(self, input_dim=10, hidden_size=256, dropout=0.2, num_classes=1, **kwargs):
        super(MLPBinaryClassificationLoss, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_classes = num_classes

        self.linear1 = nn.Linear(self.input_dim, self.hidden_size)
        self.dropout1 = nn.Dropout(p=self.dropout)

        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout2 = nn.Dropout(p=self.dropout)

    def forward(self, input):
        output = self.dropout1(torch.tanh(self.linear1(input)))
        output = self.dropout2(torch.tanh(self.linear2(output)))
        output = torch.sigmoid(output)
        return output