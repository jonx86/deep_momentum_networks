import torch.nn as nn
import torch


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, device):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, padding=self.padding,
                              dilation=dilation, device=device)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs[:, :, :-self.padding]


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, device):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.DilatedConv = CausalConv1d(in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        kernel_size=self.kernel_size,
                                        dilation=self.dilation,
                                        device=device)

        self.residConnection = nn.Conv1d(in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         kernel_size=1, dilation=1,
                                         device=device)

        self.skipConnection = nn.Conv1d(in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        kernel_size=1, dilation=1,
                                        device=device)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # outputs
        outputs = self.DilatedConv(inputs)
        tanh, sig = self.tanh(outputs), self.sigmoid(outputs)
        outputs = tanh * sig

        # resid connection
        resid_connection = self.residConnection(outputs) + inputs

        # skip connection
        skip_connection = self.skipConnection(outputs)
        return resid_connection, skip_connection


class ResStack(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, device):
        super(ResStack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.device = device

        # the paper mentions dialiation rates of 5, 10, 15, 21, 42 as layers of the residual block
        self.cnnFiveDilation = ResBlock(in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        kernel_size=2, dilation=5,
                                        device=self.device)

        self.cnn10Dilation = ResBlock(in_channels=self.in_channels,
                                      out_channels=self.out_channels,
                                      kernel_size=2, dilation=10,
                                      device=self.device)

        self.cnn15Dilation = ResBlock(in_channels=self.in_channels,
                                      out_channels=self.out_channels,
                                      kernel_size=2, dilation=15,
                                      device=self.device)

        self.cnn21Dilation = ResBlock(in_channels=self.in_channels,
                                      out_channels=self.out_channels,
                                      kernel_size=2, dilation=21,
                                      device=self.device)

        self.cnn42Dilation = ResBlock(in_channels=self.in_channels,
                                      out_channels=self.out_channels,
                                      kernel_size=2, dilation=42,
                                      device=self.device)

    def forward(self, inputs):
        # iterate through the residual stack
        residual_connection, skip_connection1 = self.cnnFiveDilation(inputs)
        residual_connection, skip_connection2 = self.cnn10Dilation(residual_connection)
        residual_connection, skip_connection3 = self.cnn15Dilation(residual_connection)
        residual_connection, skip_connection4 = self.cnn21Dilation(residual_connection)
        residual_connection, skip_connection5 = self.cnn42Dilation(residual_connection)

        skips = torch.stack((skip_connection1,
                             skip_connection2,
                             skip_connection3,
                             skip_connection4,
                             skip_connection5))
        return skips


class FullyConnectedBlock(nn.Module):
    def __init__(self, hidden, out_dim=1, dropout_rate=.30):
        super(FullyConnectedBlock, self).__init__()
        self.hidden = hidden
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate

        self.seq = nn.Sequential(
            nn.LazyLinear(self.hidden),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_rate),
            nn.LazyLinear(out_dim),
            nn.Tanh())

    def forward(self, inputs):
        outputs = self.seq(inputs)
        return outputs


class WaveNetSharpeLoss(nn.Module):

    def __init__(self, in_channels, out_channels, kernal_size, hidden_size, dropOutRate=.30, device='cuda'):
        super(WaveNetSharpeLoss, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernal_size = kernal_size
        self.hidden_size = hidden_size
        self.dropOutRate = dropOutRate
        self.device = device

        self.CausalConvFirst = CausalConv1d(in_channels=self.in_channels,
                                            out_channels=self.out_channels,
                                            kernel_size=self.kernal_size,
                                            dilation=1, device=self.device)

        self.ResStacks = ResStack(in_channels=self.out_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=2, device=device)

        self.tanh1 = nn.Tanh()

        # drop out
        self.drop1 = nn.Dropout(p=self.dropOutRate)
        self.flattener = nn.Flatten()

        # last we have our full connected block
        self.fcBlock = FullyConnectedBlock(hidden=self.hidden_size,
                                           out_dim=1,
                                           dropout_rate=self.dropOutRate)

    def forward(self, inputs):
        # pass through the first casual convolutional layer
        outputs = self.CausalConvFirst(inputs)

        # now we go into the residual block
        skip_connections = self.ResStacks(outputs)
        skip_connections = torch.sum(skip_connections, dim=0)
        skip_connections = self.tanh1(skip_connections)

        # flatten before last fully connected layer
        skip_connections = self.flattener(skip_connections)
        skip_connections = self.fcBlock(skip_connections)
        return skip_connections