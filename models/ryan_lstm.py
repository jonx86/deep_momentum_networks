import torch
import torch.nn as nn


class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, dropout=0.0, device=None):
    super(LSTM, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.dropout = dropout
    self.device = device

    self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, dropout=self.dropout, device=self.device)
    self.linear = nn.Linear(self.hidden_size, 1, device=self.device)

  def forward(self, input):
    output, _ = self.lstm(input)
    output = self.linear(output)
    output = torch.tanh(output)
    return output
