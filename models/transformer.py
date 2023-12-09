import torch.nn as nn
import torch
import random
import numpy as np

class TransformerEncoderSharpeLoss(nn.Module):
    def __init__(self, input_size, hidden_size, maximum_length, number_of_heads=4, number_of_encoder_layers=1, feedforward_size=100, dropout=0.0, device=None):
        super(TransformerEncoderSharpeLoss, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.maximum_length = maximum_length
        self.number_of_heads = number_of_heads
        self.number_of_encoder_layers = number_of_encoder_layers
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Linear(self.input_size, self.hidden_size, device=self.device)

        encoding = torch.zeros(self.maximum_length, self.hidden_size, device=self.device)
        position = torch.arange(self.maximum_length, device=self.device)[:, None]
        feature = torch.arange(0, self.hidden_size / 2, device=self.device)[None, :]
        denominator = torch.exp(2 * feature / hidden_size * torch.log(torch.tensor(10000)))
        encoding[:, 0::2] = torch.sin(position / denominator)
        encoding[:, 1::2] = torch.cos(position / denominator)
        self.register_buffer("encoding", encoding[None, :, :])

        transformer_encoder_layer = nn.TransformerEncoderLayer(self.hidden_size, self.number_of_heads, dim_feedforward=self.feedforward_size, dropout=dropout, batch_first=True, device=device)
        self.transformer = torch.nn.TransformerEncoder(transformer_encoder_layer, self.number_of_encoder_layers, mask_check=True)

        self.linear = nn.Linear(self.hidden_size, 1, device=self.device)

    def forward(self, input):
        sequence_length = input.shape[1]

        output = self.encoding[:, 0:sequence_length, :] + self.embedding(input)

        mask = nn.Transformer.generate_square_subsequent_mask(sequence_length, device=self.device)
        mask = mask != 0

        output = self.transformer(output, mask=mask, is_causal=True)
        output = self.linear(output)
        output = torch.tanh(output)

        return output
    
class TransformerEncoderRetLoss(nn.Module):
    def __init__(self, input_size, hidden_size, maximum_length, number_of_heads=4, number_of_encoder_layers=1, feedforward_size=100, dropout=0.0, device=None):
        super(TransformerEncoderRetLoss, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.maximum_length = maximum_length
        self.number_of_heads = number_of_heads
        self.number_of_encoder_layers = number_of_encoder_layers
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Linear(self.input_size, self.hidden_size, device=self.device)

        encoding = torch.zeros(self.maximum_length, self.hidden_size, device=self.device)
        position = torch.arange(self.maximum_length, device=self.device)[:, None]
        feature = torch.arange(0, self.hidden_size / 2, device=self.device)[None, :]
        denominator = torch.exp(2 * feature / hidden_size * torch.log(torch.tensor(10000)))
        encoding[:, 0::2] = torch.sin(position / denominator)
        encoding[:, 1::2] = torch.cos(position / denominator)
        self.register_buffer("encoding", encoding[None, :, :])

        transformer_encoder_layer = nn.TransformerEncoderLayer(self.hidden_size, self.number_of_heads, dim_feedforward=self.feedforward_size, dropout=dropout, batch_first=True, device=device)
        self.transformer = torch.nn.TransformerEncoder(transformer_encoder_layer, self.number_of_encoder_layers, mask_check=True)

        self.linear = nn.Linear(self.hidden_size, 1, device=self.device)

    def forward(self, input):
        sequence_length = input.shape[1]

        output = self.encoding[:, 0:sequence_length, :] + self.embedding(input)

        mask = nn.Transformer.generate_square_subsequent_mask(sequence_length, device=self.device)
        mask = mask != 0

        output = self.transformer(output, mask=mask, is_causal=True)
        output = self.linear(output)
        output = torch.tanh(output)

        return output

class TransformerEncoderRegressionLoss(nn.Module):
    def __init__(self, input_size, hidden_size, maximum_length, number_of_heads=4, number_of_encoder_layers=1, feedforward_size=100, dropout=0.0, device=None):
        super(TransformerEncoderRegressionLoss, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.maximum_length = maximum_length
        self.number_of_heads = number_of_heads
        self.number_of_encoder_layers = number_of_encoder_layers
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Linear(self.input_size, self.hidden_size, device=self.device)

        encoding = torch.zeros(self.maximum_length, self.hidden_size, device=self.device)
        position = torch.arange(self.maximum_length, device=self.device)[:, None]
        feature = torch.arange(0, self.hidden_size / 2, device=self.device)[None, :]
        denominator = torch.exp(2 * feature / hidden_size * torch.log(torch.tensor(10000)))
        encoding[:, 0::2] = torch.sin(position / denominator)
        encoding[:, 1::2] = torch.cos(position / denominator)
        self.register_buffer("encoding", encoding[None, :, :])

        transformer_encoder_layer = nn.TransformerEncoderLayer(self.hidden_size, self.number_of_heads, dim_feedforward=self.feedforward_size, dropout=dropout, batch_first=True, device=device)
        self.transformer = torch.nn.TransformerEncoder(transformer_encoder_layer, self.number_of_encoder_layers, mask_check=True)

        self.linear = nn.Linear(self.hidden_size, 1, device=self.device)

    def forward(self, input):
        sequence_length = input.shape[1]

        output = self.encoding[:, 0:sequence_length, :] + self.embedding(input)

        mask = nn.Transformer.generate_square_subsequent_mask(sequence_length, device=self.device)
        mask = mask != 0

        output = self.transformer(output, mask=mask, is_causal=True)
        output = self.linear(output)

        return output
    
class TransformerEncoderBinaryClassificationLoss(nn.Module):
    def __init__(self, input_size, hidden_size, maximum_length, number_of_heads=4, number_of_encoder_layers=1, feedforward_size=100, dropout=0.0, device=None):
        super(TransformerEncoderBinaryClassificationLoss, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.maximum_length = maximum_length
        self.number_of_heads = number_of_heads
        self.number_of_encoder_layers = number_of_encoder_layers
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Linear(self.input_size, self.hidden_size, device=self.device)

        encoding = torch.zeros(self.maximum_length, self.hidden_size, device=self.device)
        position = torch.arange(self.maximum_length, device=self.device)[:, None]
        feature = torch.arange(0, self.hidden_size / 2, device=self.device)[None, :]
        denominator = torch.exp(2 * feature / hidden_size * torch.log(torch.tensor(10000)))
        encoding[:, 0::2] = torch.sin(position / denominator)
        encoding[:, 1::2] = torch.cos(position / denominator)
        self.register_buffer("encoding", encoding[None, :, :])

        transformer_encoder_layer = nn.TransformerEncoderLayer(self.hidden_size, self.number_of_heads, dim_feedforward=self.feedforward_size, dropout=dropout, batch_first=True, device=device)
        self.transformer = torch.nn.TransformerEncoder(transformer_encoder_layer, self.number_of_encoder_layers, mask_check=True)

        self.linear = nn.Linear(self.hidden_size, 1, device=self.device)

    def forward(self, input):
        sequence_length = input.shape[1]

        output = self.encoding[:, 0:sequence_length, :] + self.embedding(input)

        mask = nn.Transformer.generate_square_subsequent_mask(sequence_length, device=self.device)
        mask = mask != 0

        output = self.transformer(output, mask=mask, is_causal=True)
        output = self.linear(output)
        output = torch.sigmoid(output)

        return output

class TransformerEncoderSharpeLossCustom(nn.Module):
    def __init__(self, input_size, hidden_size, maximum_length, number_of_heads=4, number_of_encoder_layers=1, feedforward_size=100, dropout=0.0, device=None):
        super(TransformerEncoderSharpeLossCustom, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.maximum_length = maximum_length
        self.number_of_heads = number_of_heads
        self.number_of_encoder_layers = number_of_encoder_layers
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Linear(self.input_size, self.hidden_size, device=self.device)

        encoding = torch.zeros(self.maximum_length, self.hidden_size, device=self.device)
        position = torch.arange(self.maximum_length, device=self.device)[:, None]
        feature = torch.arange(0, self.hidden_size / 2, device=self.device)[None, :]
        denominator = torch.exp(2 * feature / hidden_size * torch.log(torch.tensor(10000)))
        encoding[:, 0::2] = torch.sin(position / denominator)
        encoding[:, 1::2] = torch.cos(position / denominator)
        self.register_buffer("encoding", encoding[None, :, :])

        transformer_encoder_layer = nn.TransformerEncoderLayer(self.hidden_size, self.number_of_heads, dim_feedforward=self.feedforward_size, dropout=dropout, batch_first=True, device=device)
        self.transformer = torch.nn.TransformerEncoder(transformer_encoder_layer, self.number_of_encoder_layers, mask_check=True)

        self.linear = nn.Linear(self.hidden_size, 1, device=self.device)

    def forward(self, input):
        sequence_length = input.shape[1]

        output = self.encoding[:, 0:sequence_length, :] + self.embedding(input)

        mask = nn.Transformer.generate_square_subsequent_mask(sequence_length, device=self.device)
        mask = mask != 0

        output = self.transformer(output, mask=mask, is_causal=True)
        output = self.linear(output)
        output = torch.tanh(output)

        return output