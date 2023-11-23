import torch
import torch.nn as nn


class RegressionLoss(nn.Module):
  def __init__(self):
    super(RegressionLoss, self).__init__()

  def forward(self, input, target):
    M = torch.prod(input.shape)

    difference = input - target

    return 1 / M * torch.sum(torch.pow(difference, 2))


class BinaryLoss(nn.Module):
  def __init__(self):
    super(BinaryLoss, self).__init__()

  def forward(self, input, target):
    M = torch.prod(input.shape)

    # agnostic to cuda or cpu
    indicator = 0 * target
    indicator[target > 0] = 1

    return -1 / M * torch.sum(indicator * torch.log(input) + (1 - indicator) * torch.log(1 - input))


class AverageReturnsLoss(nn.Module):
  def __init__(self, target_sigma = 0.15):
    super(AverageReturnsLoss, self).__init__()
    self.target_sigma = target_sigma

  def forward(self, input, target):
    M = torch.prod(input.shape)

    R = input * self.target_sigma * target

    return -1 / M * torch.sum(R)


class SharpeLoss(nn.Module):
  def __init__(self, target_sigma = 0.15):
    super(SharpeLoss, self).__init__()
    self.target_sigma = target_sigma

  def forward(self, input, target):
    M = torch.prod(input.shape)

    R = input * self.target_sigma * target
    R_squared = torch.pow(R, 2)

    mu_R = 1 / M * torch.sum(R)
    mu_R_squared = torch.pow(mu_R, 2)

    return -mu_R * torch.sqrt(252) / torch.sqrt(torch.sum(R_squared) / M - mu_R_squared)
