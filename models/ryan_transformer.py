import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, maximum_length, number_of_heads=2, number_of_encoder_layers=1, number_of_decoder_layers=1, feedforward_size=None, dropout=0.0, device=None):
        super(Transformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.maximum_length = maximum_length
        self.number_of_heads = number_of_heads
        self.number_of_encoder_layers = number_of_encoder_layers
        self.number_of_decoder_layers = number_of_decoder_layers
        self.feedforward_size = feedforward_size
        if self.feedforward_size is None:
            self.feedforward_size = self.hidden_size
        self.dropout = dropout
        self.device = device

        self.src_embedding = nn.Linear(self.input_size, self.hidden_size, device=self.device)
        self.tgt_embedding = nn.Linear(self.input_size, self.hidden_size, device=self.device)

        encoding = torch.zeros(self.maximum_length, self.hidden_size, device=self.device)
        position = torch.arange(self.maximum_length, device=self.device)[:, None]
        feature = torch.arange(0, self.hidden_size / 2, device=self.device)[None, :]
        denominator = torch.exp(2 * feature / hidden_size * torch.log(torch.tensor(10000)))
        encoding[:, 0::2] = torch.sin(position / denominator)
        encoding[:, 1::2] = torch.cos(position / denominator)
        self.register_buffer("encoding", encoding[None, :, :])

        self.transformer = nn.Transformer(d_model=self.hidden_size, nhead=self.number_of_heads, num_encoder_layers=self.number_of_encoder_layers, num_decoder_layers=self.number_of_decoder_layers, dim_feedforward=self.feedforward_size, dropout=dropout, batch_first=True, device=self.device)

        self.linear = nn.Linear(self.hidden_size, 1, device=self.device)

    def forward(self, input):
        sequence_length = input.shape[1]

        src = self.encoding[:, 0:sequence_length-1, :] + self.src_embedding(input[:, 0:sequence_length-1, :])
        tgt = self.encoding[:, 0:1, :] + self.tgt_embedding(input[:, sequence_length-1:, :])

        output = self.transformer(src, tgt)
        output = self.linear(output)
        output = torch.tanh(output)

        output = torch.squeeze(output, 1)

        return output
