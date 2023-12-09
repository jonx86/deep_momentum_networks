import torch.nn as nn
import torch
import random
import numpy as np

class simpleLSTMSharpeLoss(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(simpleLSTMSharpeLoss, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            batch_first=True)

        # you may or may not need to do this, LSTM does not work on GPU for me
        # and can't fix the bug as of now, params needs to be same memory block
        # https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506"
        self.lstm.flatten_parameters()

        self.linear = nn.Linear(self.hidden_dim, 1)
        self.dropOut1 = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        outputs = torch.tanh(self.linear(outputs))
        # we want to return the prediction at the last time-step , this is the position size at t+1
        return outputs


class FullLSTMSharpeLoss(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(FullLSTMSharpeLoss, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            batch_first=True)

        # you may or may not need to do this, LSTM does not work on GPU for me
        # and can't fix the bug as of now, params needs to be same memory block
        # https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506"
        self.lstm.flatten_parameters()

        # fully connecte4d block, 2-layer MLP
        self.fcBlock = nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_rate),
            nn.LazyLinear(out_features=1),
            nn.Tanh())
        
    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        outputs = self.fcBlock(outputs)
        return outputs

class simpleLSTMSharpeLossCustom(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=.30):
        super(simpleLSTMSharpeLossCustom, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            batch_first=True)

        # you may or may not need to do this, LSTM does not work on GPU for me
        # and can't fix the bug as of now, params needs to be same memory block
        # https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506"
        self.lstm.flatten_parameters()

        self.linear = nn.Linear(self.hidden_dim, 1)
        self.dropOut1 = nn.Dropout()
        
    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        outputs = torch.tanh(self.linear(outputs))
        # we want to return the prediction at the last time-step , this is the position size at t+1
        return outputs


class FullLSTMSharpeLossCustom(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=.30):
        super(FullLSTMSharpeLossCustom, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            batch_first=True)

        # you may or may not need to do this, LSTM does not work on GPU for me
        # and can't fix the bug as of now, params needs to be same memory block
        # https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506"
        self.lstm.flatten_parameters()

        # fully connecte4d block, 2-layer MLP
        self.fcBlock = nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_rate),
            nn.LazyLinear(out_features=1),
            nn.Tanh())

    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        outputs = self.fcBlock(outputs)
        return outputs

class simpleLSTMRetLoss(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(simpleLSTMRetLoss, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            batch_first=True)

        # you may or may not need to do this, LSTM does not work on GPU for me
        # and can't fix the bug as of now, params needs to be same memory block
        # https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506"
        self.lstm.flatten_parameters()

        self.linear = nn.Linear(self.hidden_dim, 1)
        self.dropOut1 = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        outputs = torch.tanh(self.linear(outputs))
        # we want to return the prediction at the last time-step , this is the position size at t+1
        return outputs

class simpleLSTMRegressionLoss(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(simpleLSTMRegressionLoss, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            batch_first=True)

        # you may or may not need to do this, LSTM does not work on GPU for me
        # and can't fix the bug as of now, params needs to be same memory block
        # https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506"
        self.lstm.flatten_parameters()

        self.linear = nn.Linear(self.hidden_dim, 1)
        self.dropOut1 = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        outputs = self.linear(outputs)
        # we want to return the prediction at the last time-step , this is the position size at t+1
        return outputs

class simpleLSTMBinaryClassificationLoss(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(simpleLSTMBinaryClassificationLoss, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            batch_first=True)

        # you may or may not need to do this, LSTM does not work on GPU for me
        # and can't fix the bug as of now, params needs to be same memory block
        # https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506"
        self.lstm.flatten_parameters()

        self.linear = nn.Linear(self.hidden_dim, 1)
        self.dropOut1 = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        outputs = torch.sigmoid(self.linear(outputs))
        # we want to return the prediction at the last time-step , this is the position size at t+1
        return outputs

class simpleLSTM2LSharpeLoss(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=2, dropout_rate=0.30):
        super(simpleLSTM2LSharpeLoss, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = layers

        if self.num_layers > 1:
            self.dropout_rate = dropout_rate
        else:
            self.dropout = None
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            batch_first=True)

        # you may or may not need to do this, LSTM does not work on GPU for me
        # and can't fix the bug as of now, params needs to be same memory block
        # https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506"
        self.lstm.flatten_parameters()

        self.linear = nn.Linear(self.hidden_dim, 1)
        # self.dropOut1 = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        outputs = torch.tanh(self.linear(outputs))
        # we want to return the prediction at the last time-step , this is the position size at t+1
        return outputs


class FullLSTM2LSharpeLoss(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=2, dropout_rate=0.30):
        super(FullLSTM2LSharpeLoss, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = layers

        if self.num_layers > 1:
            self.dropout_rate = dropout_rate
        else:
            self.dropout = None
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            batch_first=True)

        # you may or may not need to do this, LSTM does not work on GPU for me
        # and can't fix the bug as of now, params needs to be same memory block
        # https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506"
        self.lstm.flatten_parameters()

        # fully connecte4d block, 2-layer MLP
        self.fcBlock = nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_rate),
            nn.LazyLinear(out_features=1),
            nn.Tanh())

    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        outputs = self.fcBlock(outputs)
        return outputs