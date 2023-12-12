import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SharpeLoss(nn.Module):

    def __init__(self, risk_trgt=.15):
        super(SharpeLoss, self).__init__()
        self.risk_trgt = risk_trgt

    def forward(self, input, target):
        """
        input and target are both 1-d
        """

        if (len(input.shape)>2) and (len(target.shape)>2):
            input = torch.flatten(input)
            target = torch.flatten(target)

        ret = input * self.risk_trgt * target
        uR = torch.mean(ret)

        r_squared = torch.mean(torch.pow(ret, 2))
        uR_squared = torch.pow(uR, 2)

        sharpe = uR * torch.sqrt(torch.tensor(252)) / torch.sqrt(r_squared - uR_squared)
        return -sharpe
    

class RetLoss(nn.Module):
     def __init__(self, risk_trgt=.15):
        super(RetLoss, self).__init__()
        self.risk_trgt = risk_trgt

     def forward(self, input, target):
        """
        input and target are both 1-d
        """
        if (len(input.shape)>2) and (len(target.shape)>2):
            input = torch.flatten(input)
            target = torch.flatten(target)

        ret = input * self.risk_trgt * target
        uR = torch.mean(ret)

        return -uR


class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()

    def forward(self, pred, target):
        # pred and target are both 1d
        loss = F.mse_loss(pred, target) # lower loss is better and we are maximizing
        return loss
    

class BinaryClassificationLoss(nn.Module):
    def __init__(self):
        super(BinaryClassificationLoss, self).__init__()
    
    def forward(self, pred, target):
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

        l = SharpeLoss()
        f = l.forward(i, torch.tensor(r/sigmas))
        print(f)

        extracted_val = float(f.cpu().numpy())
        s.loc[x] = extracted_val
    
    # plots the losses
    s.plot(kind='hist', bins=25, title='dist of losses')


    




        

        







