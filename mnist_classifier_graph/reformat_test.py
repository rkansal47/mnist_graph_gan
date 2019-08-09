import torch
import numpy as np

x = torch.rand(2, 5, 2)
s, indx = x.sort(1)
