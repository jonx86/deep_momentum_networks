import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SharpeLossTargetOnly(nn.Module):

    def __init__(self, risk_trgt=.15):
        super(SharpeLossTargetOnly, self).__init__()
        self.risk_trgt = risk_trgt

    def forward(self, input, target):
        """
        input and target are both 1-d
        """

        ret = input * self.risk_trgt * target
        uR = torch.mean(ret)

        r_squared = torch.mean(torch.pow(ret, 2))
        uR_squared = torch.pow(uR, 2)

        sharpe = uR * torch.sqrt(torch.tensor(252)) / torch.sqrt(r_squared - uR_squared)
        return -sharpe
    

class RetLossTargetOlnly(nn.Module):
     def __init__(self, risk_trgt=.15):
        super(RetLossTargetOlnly, self).__init__()
        self.risk_trgt = risk_trgt

     def forward(self, input, target):
        """
        input and target are both 1-d
        """

        ret = input * self.risk_trgt * target
        uR = torch.mean(ret)

        return -uR

class SharpeLoss(nn.Module):
    def __init__(self, risk_trgt=.15):
        super(SharpeLoss, self).__init__()
        self.risk_trgt = risk_trgt

    def forward(self, input, target):
        """
        target is 2d and contains r,t+1 and sigma
        Input is Xt 1d
        """

        ret = target[:, 0].clone()
        sigma = target[:, 1].clone()

        #print(ret.shape)
        #print(sigma.shape)

        rets = input * (self.risk_trgt/sigma * torch.sqrt(torch.tensor(252))) * ret
        ret1 = torch.mean(rets)
        ret2 = torch.mean(torch.pow(rets, 2))

        sharpe = torch.sqrt(torch.tensor(252)) * ret1 / torch.sqrt(ret2 - torch.pow(ret1, 2))
        return -sharpe

    

class RetLoss(nn.Module):
    def __init__(self, risk_trgt=15):
        super(RetLoss, self).__init__()
        self.risk_trgt = risk_trgt

    def forward(self, input, target):
        """
        target is 2d and contains rt+1, and sigma
        Input is Xt and is 1d
        """

        # return and sigma
        ret = target[:, 0].clone()
        sigma = target[:, 1].clone()

        # portfolio ret, given the trading rule and risk scaling position sizes by 
        # a risk target
        pf_ret = (input * self.risk_trgt * ret) / (np.sqrt(252) * sigma)
        pf_ew_ret = torch.mean(pf_ret)

        return -pf_ew_ret


class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()

    def forward(pred:torch.tensor, target:torch.tensor):
        # pred and target are both 1d
        loss = F.mse_loss(pred, target) # lower loss is better and we are maximizing
        return -loss
    

class BinaryClassificationLoss(nn.Module):
    def __init__(self):
        super(BinaryClassificationLoss, self).__init__()
    
    def forward(pred: torch.tensor, target:torch.tensor):
        # pred and target are both 1d
        loss = F.binary_cross_entropy(pred, target)
        return -loss

    

if __name__ == "__main__":
    import pandas as pd
  
    s = pd.Series()
    for x in range(500):
        i = np.random.randn(72, 1)
        i = torch.tensor(i)
        i = torch.tanh(i)

        r = np.linspace(-.05, .05, 72)
        sigmas = np.linspace(.0001, .03, 72)
        target = np.vstack([r, sigmas]).reshape(72, 2)
        target = torch.tensor(target)

        l = SharpeLossTargetOnly()
        f = l.forward(i, torch.tensor(r/sigmas))
        print(f)

        extracted_val = float(f.cpu().numpy())
        s.loc[x] = extracted_val
    
    # plots the losses
    s.plot(kind='hist', bins=25, title='dist of losses')




        

        







