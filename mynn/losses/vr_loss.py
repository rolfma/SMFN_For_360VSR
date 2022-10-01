import torch
import math


class l1_weight_loss(torch.nn.Module):

    def __init__(self):
        super(l1_weight_loss, self).__init__()
        self.eps = 1e-12

    def forward(self, x, y, weight=None):
        if weight is None:
            equ = torch.ones_like(x).to('cuda')
        else:
            equ = weight

        diff = x - y
        loss = torch.sum(
            torch.sqrt(diff * diff + self.eps) * equ) / torch.sum(equ)
        return loss


class mse_weight_loss(torch.nn.Module):

    def __init__(self):
        super(mse_weight_loss, self).__init__()

    def forward(self, x, y, weight=None):
        if weight is None:
            equ = torch.ones_like(x).to('cuda')
        else:
            equ = weight

        diff = x - y
        loss = torch.sum((diff * diff) * equ) / torch.sum(equ)
        return loss