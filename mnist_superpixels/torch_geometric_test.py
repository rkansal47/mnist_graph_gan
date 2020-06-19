import torch
import torch_geometric

from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

import torch_geometric.transforms as T

dataset = torch.load('../mnist_superpixels/raw/training.pt')
data = Data(dataset)

datast = MNISTSuperpixels(".", True)

datast

dataset5 = MNISTSuperpixels(".", True, transform=T.Cartesian())

train_loader = DataLoader(dataset5, batch_size=2)

for dat in train_loader:
    print(dat)
    #print(dat.x)
    print(dat.edge_attr)
    print(len(dat.edge_attr))
    print(len(dat.edge_index))
    break

dat.edge_attr

dat.edge_index.shape

dat.edge_attr.shape

dat.x.shape

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

dataset2 = torch.load('../mnist_superpixels/processed/training.pt')



dataset2[0]
dataset2[1]
