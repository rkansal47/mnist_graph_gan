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
cutoff = 0.32178

pos = X[:,:,:2]

x1 = pos.repeat(1, 1, 75).reshape(batch_size, 75*75, 2)
x2 = pos.repeat(1, 75, 1)

diff_norms = torch.norm(x2 - x1 + 1e-12, dim=2)

diff = x2-x1
diff = diff[diff_norms < cutoff]

norms = diff_norms.reshape(batch_size, 75, 75)
neighborhood = torch.nonzero(norms < cutoff, as_tuple=False)
edge_attr = diff[neighborhood[:, 1] != neighborhood[:, 2]]

neighborhood = neighborhood[neighborhood[:, 1] != neighborhood[:, 2]] #remove self-loops
unique, counts = torch.unique(neighborhood[:, 0], return_counts=True)
edge_slices = torch.cat((torch.tensor([0]), counts.cumsum(0)))
edge_index = neighborhood[:,1:].transpose(0,1)

for i in range(batch_size):
    start_index = edge_slices[i]
    end_index = edge_slices[i+1]
    max = torch.max(edge_attr[start_index:end_index])
    edge_attr[start_index:end_index] /= 2*max

edge_attr += 0.5

x = X[:,:,2].reshape(batch_size*75, 1)+0.5
pos = 27*pos.reshape(batch_size*75, 2)+13.5

x[:40]

edge_index[:, :20]
dataset[1][:, :20]

edge_index.shape
dataset[1].shape

edge_slices.shape
dataset[2].shape

edge_attr.shape

edge_attr

edge_slices
dataset[2]





def prefilter(data):
    return data.y == 3

prefilter=None

tgdataset = MNISTSuperpixels("dataset/cartesian", False, transform=T.Cartesian(), pre_filter=prefilter)
tgdataset

tgloader = DataLoader(tgdataset, batch_size=10, shuffle=False)

for dat in tgloader:
    print(dat)
    break

dat

dat.batch

batch_size = 10
zeros = torch.zeros(batch_size*75, dtype=int)
zeros[torch.arange(batch_size)*75] = 1
batch = torch.cumsum(zeros, 0)-1

batch

dat

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

dat.edge_attr

dat.edge_index

edge_index

dataset2[0]
dataset2[1]




datat = {'x':x, 'pos':pos, 'edge_attr':edge_attr, 'edge_index':edge_index}


tdc = dict(x=x)


testtuple = list()

testtuple.append(torch.zeros(5))
testtuple.append(torch.ones(2))

torch.cat(testtuple)

testtuple

datat.x = x

class tgData():
    def __init__(self, x, pos, edge_index, edge_attr):
        self.x = x
        self.pos = pos
