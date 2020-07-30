import torch
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (graclus, max_pool, global_mean_pool)
from torch_geometric.nn import GMMConv

class ConvGenerator(torch.nn.Module):
    def __init__(self, kernel_size=25, node_latent_dim=16, dropout=0.5, wgan=False, device='cpu'):
        super(MoNet, self).__init__()
        self.conv1 = GMMConv(node_latent_dim, 32, dim=2, kernel_size=kernel_size)
        self.conv2 = GMMConv(32, 64, dim=2, kernel_size=kernel_size)
        self.conv3 = GMMConv(64, 64, dim=2, kernel_size=kernel_size)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 3)
        self.dropout = dropout
        self.device = device
        self.wgan = wgan

    def forward(self, data):
        data.edge_att
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))


        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))

        x = global_mean_pool(data.x, data.batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        y = self.fc2(x)

        if(self.wgan):
            return y

        return torch.sigmoid(y)


class MoNetDiscriminator(torch.nn.Module):
    def __init__(self, kernel_size, dropout=0.5, wgan=False, device='cpu'):
        super(MoNet, self).__init__()
        self.conv1 = GMMConv(1, 32, dim=2, kernel_size=kernel_size)
        self.conv2 = GMMConv(32, 64, dim=2, kernel_size=kernel_size)
        self.conv3 = GMMConv(64, 64, dim=2, kernel_size=kernel_size)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 1)
        self.dropout = dropout
        self.device = device
        self.wgan = wgan

    def normalized_cut_2d(edge_index, pos):
        row, col = edge_index
        edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
        return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))

        x = F.elu(self.fc1(data.x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.fc2(x)
        y = global_mean_pool(x, data.batch)

        if(self.wgan):
            return y

        return torch.sigmoid(y)
