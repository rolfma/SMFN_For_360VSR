import torch
from mynn.models.myexp_model import ParallelNet
from mynn.utils import HookTool

hook_tool = HookTool()
net = ParallelNet()
data = torch.randn([1, 3, 64, 64])
out1, out2, out3 = net(data)
print(out1.shape, out2.shape, out3.shape)