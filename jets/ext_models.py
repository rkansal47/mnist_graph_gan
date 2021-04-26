import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import NNConv
from torch_cluster import knn_graph

import logging

class rGANG(nn.Module):
    def __init__(self, args):
        super(rGANG, self).__init__()
        self.args = args

        self.args.rgang_fc.insert(0, self.args.latent_dim)

        layers = []
        for i in range(len(self.args.rgang_fc) - 1):
            layers.append(nn.Linear(self.args.rgang_fc[i], self.args.rgang_fc[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=args.leaky_relu_alpha))

        layers.append(nn.Linear(self.args.rgang_fc[-1], self.args.num_hits * self.args.node_feat_size))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

        logging.info("rGAN generator: \n {}".format(self.model))

    def forward(self, x, labels=None, epoch=None):
        return self.model(x).reshape(-1, self.args.num_hits, self.args.node_feat_size)


class rGAND(nn.Module):
    def __init__(self, args):
        super(rGAND, self).__init__()
        self.args = args

        self.args.rgand_sfc.insert(0, self.args.node_feat_size)

        layers = []
        for i in range(len(self.args.rgand_sfc) - 1):
            layers.append(nn.Conv1d(self.args.rgand_sfc[i], self.args.rgand_sfc[i + 1], 1))
            layers.append(nn.LeakyReLU(negative_slope=args.leaky_relu_alpha))

        self.sfc = nn.Sequential(*layers)

        self.args.rgand_fc.insert(0, self.args.rgand_sfc[-1])

        layers = []
        for i in range(len(self.args.rgand_fc) - 1):
            layers.append(nn.Linear(self.args.rgand_fc[i], self.args.rgand_fc[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=args.leaky_relu_alpha))

        layers.append(nn.Linear(self.args.rgand_fc[-1], 1))
        layers.append(nn.Sigmoid())

        self.fc = nn.Sequential(*layers)

        logging.info("rGAND sfc: \n {}".format(self.sfc))
        logging.info("rGAND fc: \n {}".format(self.fc))

    def forward(self, x, labels=None, epoch=None):
        x = x.reshape(-1, self.args.node_feat_size, 1)
        x = self.sfc(x)
        x = torch.max(x.reshape(-1, self.args.num_hits, self.args.rgand_sfc[-1]), 1)[0]
        return self.fc(x)


class GraphCNNGANG(nn.Module):
    def __init__(self, args):
        super(GraphCNNGANG, self).__init__()
        self.args = args

        self.dense = nn.Linear(self.args.latent_dim, self.args.num_hits * self.args.graphcnng_layers[0])

        self.layers = nn.ModuleList()
        self.edge_weights = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(self.args.graphcnng_layers) - 1):
            self.edge_weights.append(nn.Linear(self.args.graphcnng_layers[i], self.args.graphcnng_layers[i] * self.args.graphcnng_layers[i + 1]))
            self.layers.append(NNConv(self.args.graphcnng_layers[i], self.args.graphcnng_layers[i + 1], self.edge_weights[i], aggr='mean', root_weight=True, bias=True))
            self.bn_layers.append(torch_geometric.nn.BatchNorm(self.args.graphcnng_layers[i + 1]))

        self.edge_weights.append(nn.Linear(self.args.graphcnng_layers[-1], self.args.graphcnng_layers[-1] * self.args.node_feat_size))
        self.layers.append(NNConv(self.args.graphcnng_layers[-1], self.args.node_feat_size, self.edge_weights[-1], aggr='mean', root_weight=True, bias=True))
        self.bn_layers.append(torch_geometric.nn.BatchNorm(self.args.node_feat_size))

        logging.info("dense: ")
        logging.info(self.dense)

        logging.info("edge_weights: ")
        logging.info(self.edge_weights)

        logging.info("layers: ")
        logging.info(self.layers)

        logging.info("bn layers: ")
        logging.info(self.bn_layers)


    def forward(self, x, labels=None, epoch=None):
        x = F.leaky_relu(self.dense(x), negative_slope=self.args.leaky_relu_alpha)

        batch_size = x.size(0)
        x = x.reshape(batch_size * self.args.num_hits, self.args.graphcnng_layers[0])
        zeros = torch.zeros(batch_size * self.args.num_hits, dtype=int).to(self.args.device)
        zeros[torch.arange(batch_size) * self.args.num_hits] = 1
        batch = torch.cumsum(zeros, 0) - 1

        for i in range(len(self.layers)):
            edge_index = knn_graph(x, 1, batch)
            edge_attr = x[edge_index[0]] - x[edge_index[1]]
            x = F.leaky_relu(self.bn_layers[i](self.layers[i](x, edge_index, edge_attr)), negative_slope=self.args.leaky_relu_alpha)

        return x.reshape(batch_size, self.args.num_hits, self.args.node_feat_size)
