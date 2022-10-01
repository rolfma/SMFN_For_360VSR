import torch
import torch.nn as nn


class mse_weight_loss(nn.Module):
    def __init__(self):
        super(mse_weight_loss, self).__init__()

    def forward(self, x, y, weight=None):
        if weight is None:
            w = torch.ones_like(x)
        else:
            w = weight

        diff = x - y
        loss = torch.sum((diff * diff) * w) / torch.sum(w)
        return loss