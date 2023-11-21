import torch
import torch.nn as nn


# input: 1-d tensor of the predictions
# target: 1-d tensor of r/sigma


class RegressionLoss(nn.Module):
  def forward(self, input, target):
    M = input.shape[0]

    difference = input - target

    return 1 / M * torch.sum(torch.pow(difference, 2))


class BinaryLoss(nn.Module):
  def forward(self, input, target):
    M = input.shape[0]

    # agnostic to cuda or cpu
    indicator = 0 * target
    indicator[target > 0] = 1

    return -1 / M * torch.sum(indicator * torch.log(input) + (1 - indicator) * torch.log(1 - input))


class AverageReturnsLoss(nn.Module):
  def __init__(self, target_sigma = 0.15):
    self.target_sigma = target_sigma

  def forward(self, input, target):
    M = input.shape[0]

    R = input * self.target_sigma * target

    return -1 / M * torch.sum(R)


class SharpeLoss(nn.Module):
  def __init__(self, target_sigma = 0.15):
    self.target_sigma = target_sigma

  def forward(self, input, target):
    M = input.shape[0]

    R = input * self.target_sigma * target
    R_squared = R * R

    mu_R = 1 / M * torch.sum(R)
    mu_R_squared = mu_R * mu_R

    return -mu_R * torch.sqrt(252) / torch.sqrt(torch.sum(R_squared) / M - mu_R_squared)
