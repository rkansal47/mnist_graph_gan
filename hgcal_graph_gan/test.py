import torch

x = torch.tensor([[1, 2],[3, 4],[5, 6]])
# x = x.unsqueeze(2)
x

x = x.repeat(1,4)
x.view(3, 4, 2)
