import torch
import torch.nn as nn
import math

class FocalLoss(nn.Module):

    def __init__(self, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, y, target):
        logp = self.ce(y, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class MultiLoss(nn.Module):
    def __init__(self, focal_gamma=2, alpha1=0.5, alpha2=0.5):
        super(MultiLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.fl = FocalLoss(gamma=focal_gamma)
        
        self.alpha1 = alpha1
        self.alpha2 = alpha2
    
    def forward(self, y, target):
        lossVal = self.alpha1 * self.ce(y, target) + self.alpha2 * self.fl(y, target)
        return lossVal