import torch
from torch import nn

import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import EdgeConv
from torch_cluster import knn_graph

class ParticleNetEdgeNet(nn.Module):
    def __init__(self, in_size, layer_size):
        super(ParticleNetEdgeNet, self).__init__()

        layers = []

        layers.append(nn.Linear(in_size, layer_size))
        layers.append(nn.BatchNorm1d(layer_size))
        layers.append(nn.ReLU())

        for i in range(2):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.model)

class ParticleNet(nn.Module):
    def __init__(self, args):
        super(ParticleNet, self).__init__()
        self.args = args
        self.k = 16
        self.num_edge_convs = 3
        self.kernel_sizes = [64, 128, 256]
        self.fc1_size = 256
        self.num_classes = 5
        self.dropout = 0.1
        self.num_features = 3

        self.edge_nets = nn.ModuleList()
        self.edge_convs = nn.ModuleList()

        self.edge_nets.append(ParticleNetEdgeNet(self.num_features, self.kernel_sizes[0]))
        self.edge_convs.append(EdgeConv(self.edge_nets[-1], aggr='mean'))

        for i in range(self.num_edge_convs - 1):
            self.edge_nets.append(ParticleNetEdgeNet(self.kernel_sizes[i + 1], self.kernel_sizes[i + 1]))
            self.edge_convs.append(EdgeConv(self.edge_nets[-1], aggr='mean'))

        self.fc1 = nn.Sequential(nn.Linear(self.kernel_sizes[-1], self.fc1_size), nn.ReLU(), nn.Dropout(p=self.dropout))
        self.fc2 = nn.Linear(self.fc1_size, self.num_classes)

        logging.info("edge nets: ")
        logging.info(self.edge_nets)

        logging.info("edge_convs: ")
        logging.info(self.edge_convs)

        logging.info("fc1: ")
        logging.info(self.fc1)

        logging.info("fc2: ")
        logging.info(self.fc2)


    def forward(self, x):
        x = F.leaky_relu(self.dense(x), negative_slope=self.args.leaky_relu_alpha)

        batch_size = x.size(0)
        x = x.reshape(batch_size * self.args.num_hits, self.num_features)
        zeros = torch.zeros(batch_size * self.args.num_hits, dtype=int).to(self.args.device)
        zeros[torch.arange(batch_size) * self.args.num_hits] = 1
        batch = torch.cumsum(zeros, 0) - 1

        for i in range(self.num_edge_convs):
            edge_index = knn_graph(x, 1, batch) if i == 0 else edge_index = knn_graph(x, 1, batch)
            edge_attr = x[edge_index[0]] - x[edge_index[1]]
            x = F.leaky_relu(self.bn_layers[i](self.layers[i](x, edge_index, edge_attr)), negative_slope=self.args.leaky_relu_alpha)

        return x.reshape(batch_size, self.args.num_hits, self.args.node_feat_size)
