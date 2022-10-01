import torch.nn as nn
from mynn.ops import PartialConv2d

conv = PartialConv2d(3, 3, 3)
print(isinstance(conv, nn.Conv2d))