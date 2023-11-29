import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, lookback_size, input_size, hidden_size, dropout=0.0, device=None):
        super(MLP, self).__init__()
        self.lookback_size = lookback_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device

        self.linear1 = nn.Linear((self.lookback_size + 1) * input_size, hidden_size, device=self.device)
        self.dropout1 = nn.Dropout(p=self.dropout)

        self.linear2 = nn.Linear(hidden_size, 1, device=self.device)
        self.dropout2 = nn.Dropout(p=self.dropout)

    def forward(self, input):
        output = self.dropout1(torch.tanh(self.linear1(input)))
        output = self.dropout2(torch.tanh(self.linear2(output)))
        return output
