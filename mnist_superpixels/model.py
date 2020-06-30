import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (graclus, max_pool, global_mean_pool)
from torch_geometric.nn import GMMConv
import torch_geometric.transforms as T


class Graph_Generator(nn.Module):
    def __init__(self, node_size, fe_hidden_size, fe_out_size, fn_hidden_size, fn_num_layers, mp_iters, num_hits, dropout, alpha, hidden_node_size=64, int_diffs=False, pos_diffs=False, gru=True, batch_norm=False, device='cpu'):
        super(Graph_Generator, self).__init__()
        self.node_size = node_size
        self.fe_hidden_size = fe_hidden_size
        self.fe_out_size = fe_out_size
        self.fn_hidden_size = fn_hidden_size
        self.num_hits = num_hits
        self.alpha = alpha
        self.fn_num_layers = fn_num_layers
        self.mp_iters = mp_iters
        self.hidden_node_size = hidden_node_size
        self.gru = gru
        self.device = device
        self.int_diffs = int_diffs
        self.pos_diffs = pos_diffs
        self.batch_norm = batch_norm

        if(int_diffs and pos_diffs):
            self.fe_in_size = 2 * hidden_node_size + 2
        elif(int_diffs or pos_diffs):
            self.fe_in_size = 2 * hidden_node_size + 1
        else:
            self.fe_in_size = 2 * hidden_node_size

        self.fe1 = nn.ModuleList()
        self.fe2 = nn.ModuleList()
        self.fn1 = nn.ModuleList()
        self.fn2 = nn.ModuleList()

        if(batch_norm):
            self.bne1 = nn.ModuleList()
            self.bne2 = nn.ModuleList()
            self.bnn = nn.ModuleList()

        for i in range(mp_iters):
            self.fe1.append(nn.Linear(self.fe_in_size, fe_hidden_size))
            self.fe2.append(nn.Linear(fe_hidden_size, fe_out_size))

            if(batch_norm):
                self.bne1.append(nn.BatchNorm1d(fe_hidden_size))
                self.bne2.append(nn.BatchNorm1d(fe_out_size))

            if(self.gru):
                self.fn1.append(GRU(fe_out_size + hidden_node_size, fn_hidden_size, fn_num_layers, dropout))
                self.fn2.append(nn.Linear(fn_hidden_size, hidden_node_size))
            else:
                fni1 = nn.ModuleList()
                fni1.append(nn.Linear(fe_out_size + hidden_node_size, fn_hidden_size))
                for i in range(fn_num_layers - 1):
                    fni1.append(nn.Linear(fn_hidden_size, fn_hidden_size))

                self.fn1.append(fni1)
                self.fn2.append(nn.Linear(fn_hidden_size, hidden_node_size))

                if(batch_norm):
                    bnni = nn.ModuleList()
                    for i in range(fn_num_layers):
                        bnni.append(nn.BatchNorm1d(fn_hidden_size))
                    self.bnn.append(bnni)

    def forward(self, x):
        batch_size = x.shape[0]

        if(self.gru):
            hidden = self.initHidden(batch_size)  # since this is done on the CPU, slows down training by ~3x

        for i in range(self.mp_iters):
            A = self.getA(x, batch_size)
            A = F.leaky_relu(self.fe1[i](A), negative_slope=self.alpha)
            if(self.batch_norm): A = self.bne1[i](A)

            A = F.leaky_relu(self.fe2[i](A), negative_slope=self.alpha)
            if(self.batch_norm): A = self.bne2[i](A)

            A = torch.mean(A.view(batch_size, self.num_hits, self.num_hits, self.fe_out_size), 2)

            x = torch.cat((A, x), 2)
            x = x.view(batch_size * self.num_hits, self.fe_out_size + self.hidden_node_size)

            if(self.gru):
                x, hidden = self.fn1[i](x, hidden)
            else:
                for j in range(self.fn_num_layers):
                    x = F.leaky_relu(self.fn1[i][j](x), negative_slope=self.alpha)
                    if(self.batch_norm): x = self.bnn[i][j](x)

            x = torch.tanh(self.fn2[i](x))
            x = x.view(batch_size, self.num_hits, self.hidden_node_size)

        x = x[:, :, :self.node_size]

        return x

    def getA(self, x, batch_size):
        x1 = x.repeat(1, 1, self.num_hits).view(batch_size, self.num_hits * self.num_hits, self.hidden_node_size)
        x2 = x.repeat(1, self.num_hits, 1)

        if(self.int_diffs):
            dists = torch.norm(x2[:, :, :2] - x1[:, :, :2] + 1e-12, dim=2).unsqueeze(2)
            int_diffs = 1 - ((x2[:, :, 2] - x1[:, :, 2])).unsqueeze(2)
            A = (torch.cat((x1, x2, dists, int_diffs), 2)).view(batch_size * self.num_hits * self.num_hits, self.fe_in_size)
        elif(self.pos_diffs):
            dists = torch.norm(x2[:, :, :2] - x1[:, :, :2] + 1e-12, dim=2).unsqueeze(2)
            A = torch.cat((x1, x2, dists), 2).view(batch_size * self.num_hits * self.num_hits, self.fe_in_size)
        else:
            A = torch.cat((x1, x2), 2).view(batch_size * self.num_hits * self.num_hits, self.fe_in_size)

        return A

    def initHidden(self, batch_size):
        return torch.zeros(self.fn_num_layers, batch_size * self.num_hits, self.fn_hidden_size).to(self.device)


class Graph_Discriminator(nn.Module):
    def __init__(self, node_size, fe_hidden_size, fe_out_size, fn_hidden_size, fn_num_layers, mp_iters, num_hits, dropout, alpha, hidden_node_size=64, wgan=False, int_diffs=False, pos_diffs=False, gru=False, batch_norm=False, device='cpu'):
        super(Graph_Discriminator, self).__init__()
        self.node_size = node_size
        self.hidden_node_size = hidden_node_size
        self.fe_hidden_size = fe_hidden_size
        self.fe_out_size = fe_out_size
        self.num_hits = num_hits
        self.alpha = alpha
        self.dropout = dropout
        self.fn_num_layers = fn_num_layers
        self.fn_hidden_size = fn_hidden_size
        self.mp_iters = mp_iters
        self.wgan = wgan
        self.gru = gru
        self.device = device
        self.int_diffs = int_diffs
        self.pos_diffs = pos_diffs
        self.dropout = nn.Dropout(p=dropout)
        self.batch_norm = batch_norm

        if(int_diffs and pos_diffs):
            self.fe_in_size = 2 * hidden_node_size + 2
        elif(int_diffs or pos_diffs):
            self.fe_in_size = 2 * hidden_node_size + 1
        else:
            self.fe_in_size = 2 * hidden_node_size

        self.fe1 = nn.ModuleList()
        self.fe2 = nn.ModuleList()
        self.fn1 = nn.ModuleList()
        self.fn2 = nn.ModuleList()

        if(batch_norm):
            self.bne1 = nn.ModuleList()
            self.bne2 = nn.ModuleList()
            self.bnn = nn.ModuleList()

        for i in range(mp_iters):
            self.fe1.append(nn.Linear(self.fe_in_size, fe_hidden_size))
            self.fe2.append(nn.Linear(fe_hidden_size, fe_out_size))

            if(batch_norm):
                self.bne1.append(nn.BatchNorm1d(fe_hidden_size))
                self.bne2.append(nn.BatchNorm1d(fe_out_size))

            if(self.gru):
                self.fn1.append(GRU(fe_out_size + hidden_node_size, fn_hidden_size, fn_num_layers, dropout))
                self.fn2.append(nn.Linear(fn_hidden_size, hidden_node_size))
            else:
                fni1 = nn.ModuleList()
                fni1.append(nn.Linear(fe_out_size + hidden_node_size, fn_hidden_size))
                for i in range(fn_num_layers - 1):
                    fni1.append(nn.Linear(fn_hidden_size, fn_hidden_size))

                self.fn1.append(fni1)
                self.fn2.append(nn.Linear(fn_hidden_size, hidden_node_size))

                if(batch_norm):
                    bnni = nn.ModuleList()
                    for i in range(fn_num_layers):
                        bnni.append(nn.BatchNorm1d(fn_hidden_size))
                    self.bnn.append(bnni)

    def forward(self, x):
        batch_size = x.shape[0]
        if(self.gru):
            hidden = self.initHidden(batch_size)

        x = F.pad(x, (0, self.hidden_node_size - self.node_size, 0, 0, 0, 0))

        for i in range(self.mp_iters):
            A = self.getA(x, batch_size)

            A = F.leaky_relu(self.fe1[i](A), negative_slope=self.alpha)
            if(self.batch_norm): A = self.bne1[i](A)
            A = self.dropout(A)

            A = F.leaky_relu(self.fe2[i](A), negative_slope=self.alpha)
            if(self.batch_norm): A = self.bne2[i](A)
            A = self.dropout(A)

            A = torch.mean(A.view(batch_size, self.num_hits, self.num_hits, self.fe_out_size), 2)
            x = torch.cat((A, x), 2)
            x = x.view(batch_size * self.num_hits, self.fe_out_size + self.hidden_node_size)

            if(self.gru):
                x, hidden = self.fn1[i](x, hidden)
            else:
                for j in range(self.fn_num_layers):
                    x = F.leaky_relu(self.fn1[i][j](x), negative_slope=self.alpha)
                    if(self.batch_norm): x = self.bnn[i][j](x)
                    x = self.dropout(x)

            x = self.dropout(torch.tanh(self.fn2[i](x)))
            x = x.view(batch_size, self.num_hits, self.hidden_node_size)

        x = torch.mean(x[:, :, :1], 1)

        if(self.wgan):
            return x

        return torch.sigmoid(x)

    def getA(self, x, batch_size):
        x1 = x.repeat(1, 1, self.num_hits).view(batch_size, self.num_hits * self.num_hits, self.hidden_node_size)
        x2 = x.repeat(1, self.num_hits, 1)

        dists = torch.norm(x2[:, :, :2] - x1[:, :, :2] + 1e-12, dim=2).unsqueeze(2)

        if(self.int_diffs):
            dists = torch.norm(x2[:, :, :2] - x1[:, :, :2] + 1e-12, dim=2).unsqueeze(2)
            int_diffs = 1 - ((x2[:, :, 2] - x1[:, :, 2])).unsqueeze(2)
            A = (torch.cat((x1, x2, dists, int_diffs), 2)).view(batch_size * self.num_hits * self.num_hits, self.fe_in_size)
        elif(self.pos_diffs):
            dists = torch.norm(x2[:, :, :2] - x1[:, :, :2] + 1e-12, dim=2).unsqueeze(2)
            A = torch.cat((x1, x2, dists), 2).view(batch_size * self.num_hits * self.num_hits, self.fe_in_size)
        else:
            A = torch.cat((x1, x2), 2).view(batch_size * self.num_hits * self.num_hits, self.fe_in_size)

        return A

    def initHidden(self, batch_size):
        return torch.zeros(self.fn_num_layers, batch_size * self.num_hits, self.fn_hidden_size).to(self.device)


class GRU(nn.Module):
    def __init__(self, input_size, fn_hidden_size, num_layers, dropout):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.fn_hidden_size = fn_hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()

        self.layers.append(GRUCell(input_size, fn_hidden_size))
        for i in range(num_layers - 1):
            self.layers.append(GRUCell(fn_hidden_size, fn_hidden_size))

    def forward(self, x, hidden):
        x = x.squeeze()
        hidden[0] = F.dropout(self.layers[0](x, hidden[0].clone()), p=self.dropout)

        for i in range(1, self.num_layers):
            hidden[i] = F.dropout(self.layers[i](hidden[i - 1].clone(), hidden[i].clone()), p=self.dropout)

        return hidden[-1].unsqueeze(1).clone(), hidden


class GRUCell(nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, fn_hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.fn_hidden_size = fn_hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * fn_hidden_size, bias=bias)
        self.h2h = nn.Linear(fn_hidden_size, 3 * fn_hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.parameters():
            w.data.uniform_(-0.1, 0.1)

    def forward(self, x, hidden):

        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class Gaussian_Discriminator(nn.Module):
    def __init__(self, node_size, fe_hidden_size, fe_out_size, fn_hidden_size, fn_num_layers, mp_iters, num_hits, dropout, alpha, kernel_size, hidden_node_size=64, wgan=False, int_diffs=False, gru=False, device='cpu'):
        super(Gaussian_Discriminator, self).__init__()
        self.node_size = node_size
        self.hidden_node_size = hidden_node_size
        self.fe_hidden_size = fe_hidden_size
        self.fe_out_size = fe_out_size
        self.num_hits = num_hits
        self.alpha = alpha
        self.dropout = dropout
        self.fn_num_layers = fn_num_layers
        self.fn_hidden_size = fn_hidden_size
        self.mp_iters = mp_iters
        self.wgan = wgan
        self.gru = gru
        self.kernel_size = kernel_size
        self.device = device

        self.fn = nn.Linear(hidden_node_size, hidden_node_size)
        self.fc = nn.Linear(hidden_node_size, 1)

        self.mu = Parameter(torch.Tensor(kernel_size, 2).to(self.device))
        self.sigma = Parameter(torch.Tensor(kernel_size, 2).to(self.device))

        self.kernel_weight = Parameter(torch.Tensor(kernel_size).to(self.device))

        self.glorot(self.mu)
        self.glorot(self.sigma)
        self.kernel_weight.data.uniform_(0, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.pad(x, (0, self.hidden_node_size - self.node_size, 0, 0, 0, 0))

        for i in range(self.mp_iters):
            x1 = x.repeat(1, 1, self.num_hits).view(batch_size, self.num_hits * self.num_hits, self.hidden_node_size)
            y = x.repeat(1, self.num_hits, 1)

            u = y[:, :, :2] - x1[:, :, :2]
            y = self.fn(y)

            # print("test")
            # print(y.shape)

            y2 = torch.zeros(y.shape).to(self.device)

            for j in range(self.kernel_size):
                w = self.weights(u, j)
                # print(w)
                # print(w.shape)
                # print(y.shape)

                y2 += w.unsqueeze(-1) * self.kernel_weight[j] * y

            x = torch.sum(y2.view(batch_size, self.num_hits, self.num_hits, self.hidden_node_size), 2)
            x = x.view(batch_size, self.num_hits, self.hidden_node_size)

        y = torch.tanh(self.fc(x))
        y = torch.mean(y, 1)

        if(self.wgan):
            return y

        return torch.sigmoid(y)

    def weights(self, u, j):
        return torch.exp(torch.sum(((u - self.mu[j]) ** 2) * self.sigma[j], dim=-1))

    def initHidden(self, batch_size):
        return torch.zeros(self.fn_num_layers, batch_size * self.num_hits, self.fn_hidden_size).to(self.device)

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def zeros(self, tensor):
        if tensor is not None:
            tensor.data.fill_(0)


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class MoNet(torch.nn.Module):
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

    def forward(self, data):
        cutoff = 0.32178
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
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        y = self.fc2(x)

        if(self.wgan):
            return y

        return torch.sigmoid(y)
