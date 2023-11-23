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
        target is 2d and contains r,t+1 and sigma
        Input is Xt 1d
        """

        ret = target[:, 0].clone()
        sigma = target[:, 1].clone()

        ret = (input * self.risk_trgt * ret)/(torch.sqrt(torch.tensor(252)) * sigma)
        ret1 = torch.mean(ret)
        ret2 = torch.mean(ret ** 2)

        sharpe = (ret1 * 252)/(torch.sqrt((ret2-(ret1 **2 ))) * torch.sqrt(torch.tensor(252)))
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
    

class BinaryClassificationLoss(nn.Model):
    def __init__(self):
        super(BinaryClassificationLoss, self).__init__()
    
    def forward(pred: torch.tensor, target:torch.tensor):
        # pred and target are both 1d
        loss = F.binary_cross_entropy(pred, target)
        return -loss

    

if __name__ == "__main__":
  

    for _ in range(50):
        i = np.random.randn(72, 1)
        i = torch.tensor(i)
        i = torch.tanh(i)

        r = np.linspace(-.05, .05, 72)
        sigmas = np.linspace(.0001, .03, 72)
        target = np.vstack([r, sigmas]).reshape(72, 2)
        target = torch.tensor(target)

        l = SharpeLoss()
        f = l.forward(i, target)
        print(f)




        

        







