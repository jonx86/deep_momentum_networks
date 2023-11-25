import pandas as pd
import numpy as np
from sklearn.ensemble import  RandomForestRegressor

from utils.utils import get_cv_splits, load_features, train_val_split, process_jobs, get_returns_breakout

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
  def __init__(self, hidden_size, , dropout=0.0, device=None):
    super(PositionalEncoding, self).__init__()
    self.hidden_size = hidden_size

    self.dropout = dropout
    self.device = device





class Transformer(nn.Module):
  def __init__(self, input_size, hidden_size, maximum_length, dropout=0.0, device=None):
    super(Transformer, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.maximum_length = maximum_length
    self.dropout = dropout
    self.device = device

    self.embedding = nn.Embedding(self.input_size, self.hidden_size, device=self.device)

    encoding = torch.zeros(self.maximum_length, self.hidden_size, device=self.device)
    position = torch.arange(self.maximum_length, device=self.device)[:, None]
    feature = torch.arange(0, self.hidden_size / 2, device=self.device)[None, :]
    denominator = torch.exp(2 * feature / hidden_size * torch.log(10000))
    encoding[:, 0::2] = torch.sin(position / denominator)
    encoding[:, 1::2] = torch.cos(position / denominator)
    self.register_buffer("encoding", encoding[None, :, :])




    self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, dropout=self.dropout, device=self.device)
    self.linear = nn.Linear(self.hidden_size, 1, device=self.device)

  def forward(self, input):
    output = self.lstm(input)
    output = self.linear(output)
    output = torch.tanh(output)

    return output


# load the data
feats = load_features()