import torch
import torch_geometric

from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

import torch_geometric.transforms as T

dataset = torch.load('dataset/cartesian/raw/test.pt')

dataset[2]

ints = dataset[0]
coords = dataset[3]

# ints = ints[dataset[4]==3]
# coords = coords[dataset[4]==3]

dataset[1][:, :10]
dataset[3][0][:8]

ints = ints-0.5
coords = (coords-13.5)/27

X = torch.cat((coords, ints.unsqueeze(2)), 2)

X.shape

batch_size = 10000

Xpos = X[:,:,:2]
Xpos.shape

x1 = Xpos.repeat(1, 1, 75).reshape(batch_size, 75*75, 2)
x2 = Xpos.repeat(1, 75, 1)

norms = torch.norm(x2 - x1 + 1e-12, dim=2).reshape(batch_size, 75, 75)

cutoff = 0.3245

neighborhood = torch.nonzero(norms < cutoff)

neighborhood = neighborhood[neighborhood[:, 1] != neighborhood[:, 2]] #remove self-loops

unique, counts = torch.unique(neighborhood[:, 0], return_counts=True)
counts = torch.cat((torch.tensor([0]), counts.cumsum(0)))
counts


edge_index = neighborhood[:,1:].transpose(0,1)

edge_index[:, :20]

dataset[1][:, :20]

dataset[2]



def prefilter(data):
    return data.y == 3

tgdataset = MNISTSuperpixels("dataset/cartesian", False, transform=T.Cartesian(), pre_filter=prefilter)
tgdataset

tgloader = DataLoader(tgdataset, batch_size=batch_size)

for dat in tgloader:
    print(dat)
    break

dat.edge_index[:, :30]


dat.edge_index[:]
(16-21)/x+0.5 = 0.21
(16.07-20.9)/(0.217-0.5)
-(5.03-4.65)/17.07+0.5



dat.edge_attr
dat.pos[dat.edge_index[0,0]]-dat.pos[dat.edge_index[1,0]]

dat.edge_index

dat.pos[0]
dat.pos[3]
dat.pos[8]
dat.pos[10]

x=torch.tensor([0,6,2,2,3,1,0,1,0])

u, inv = torch.unique(x, sorted=True, return_inverse=True)
inv.new_empty(u.size(0)).scatter_(0,inv, torch.arange(inv.size(0)))


dat.pos[:75]

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
