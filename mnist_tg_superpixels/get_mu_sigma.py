# Getting mu and sigma of activation features of GCNN classifier for the FID score

import torch
from torch import nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F

from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T

from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (graclus, max_pool, global_mean_pool)
from torch_geometric.nn import GMMConv

from tqdm import tqdm

import numpy as np

from os import listdir, mkdir
from os.path import exists, dirname, realpath

import sys
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cutoff = 0.32178


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class MoNet(torch.nn.Module):
    def __init__(self, kernel_size):
        super(MoNet, self).__init__()
        self.conv1 = GMMConv(1, 32, dim=2, kernel_size=kernel_size)
        self.conv2 = GMMConv(32, 64, dim=2, kernel_size=kernel_size)
        self.conv3 = GMMConv(64, 64, dim=2, kernel_size=kernel_size)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, data):
        row, col = data.edge_index
        data.edge_attr = (data.pos[col] - data.pos[row]) / (2 * 28 * cutoff) + 0.5

        # print(data.edge_index.shape)
        # print(data.edge_index[:, -20:])

        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        row, col = data.edge_index
        data.edge_attr = (data.pos[col] - data.pos[row]) / (2 * 28 * cutoff) + 0.5

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        row, col = data.edge_index
        data.edge_attr = (data.pos[col] - data.pos[row]) / (2 * 28 * cutoff) + 0.5

        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))

        x = global_mean_pool(data.x, data.batch)
        return self.fc1(x)

        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, GMMConv):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()


def add_bool_arg(parser, name, help, default=False, no_name=None):
    varname = '_'.join(name.split('-'))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=varname, action='store_true', help=help)
    if(no_name is None):
        no_name = 'no-' + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument('--' + no_name, dest=varname, action='store_false', help=no_help)
    parser.set_defaults(**{varname: default})


dir_path = dirname(realpath(__file__))
parser = argparse.ArgumentParser()

add_bool_arg(parser, "n", "run on nautilus cluster", default=False)
parser.add_argument("--batch-size", type=int, default=128, help="batch size")
parser.add_argument("--num", type=int, default=3, help="number to train on")

args = parser.parse_args()

model_path = dir_path + "/cmodels/12_global_edge_attr_test/C_100.pt"
dataset_path = dir_path + '/dataset/cartesian/'


def pf(data):
    return data.y == args.num


pre_filter = pf if args.num != -1 else None

train_dataset = MNISTSuperpixels(dataset_path, True, pre_transform=T.Cartesian(), pre_filter=pre_filter)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

print("loaded data")

pretrained_model = torch.load(model_path, map_location=device)
C = MoNet(25)
C.load_state_dict(pretrained_model.state_dict())

torch.save(pretrained_model.state_dict(), "../mnist_superpixels/eval/C_state_dict.pt")

print("loaded model)")

# TEST WITH CLASSIFICATION FIRST

for batch_ndx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
    if(batch_ndx == 0):
        activations = C(data)
    else:
        activations = torch.cat((C(data), activations), axis=0)

activations = activations.cpu().detach().numpy()

print(activations.shape)

mu = np.mean(activations, axis=0)
sigma = np.cov(activations, rowvar=False)

print(mu)
print(mu.shape)
print(sigma)
print(sigma.shape)

np.savetxt("../mnist_superpixels/eval/mu2.txt", mu)
np.savetxt("../mnist_superpixels/eval/sigma2.txt", sigma)
