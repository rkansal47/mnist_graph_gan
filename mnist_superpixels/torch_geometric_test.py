import torch
import torch_geometric

from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import Data

import torch_geometric.transforms as T

dataset = torch.load('dataset/training.pt')
data = Data(dataset)

datast = MNISTSuperpixels(".", True)

dataset5 = MNISTSuperpixels(".", True, transform=T.Polar())

dataset[0].shape

dataset[1].shape

dataset[2].shape

dataset[3].shape

dataset[4].shape

data = Data(x=dataset[0],edge_index=dataset[1],pos=dataset[3])

data

data.edge_index

torch.unique(data.edge_index[0])

torch.unique(data.edge_index[1])

data.edge_index[0,:75]
data.pos[0]

dataset2 = torch.load('processed/training.pt')



dataset2[0]
dataset2[1]
