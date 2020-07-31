import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (graclus, max_pool, global_mean_pool)
from torch_geometric.nn import GMMConv
import torch_geometric.transforms as T

from spectral_normalization import SpectralNorm


class Graph_GAN(nn.Module):
    def __init__(self, gen, args):
        super(Graph_GAN, self).__init__()
        self.args = args

        self.G = gen
        self.D = not gen

        self.test = 10

        self.args.spectral_norm = self.args.spectral_norm_gen if self.G else self.args.spectral_norm_disc
        self.args.batch_norm = self.args.batch_norm_gen if self.G else self.args.batch_norm_disc
        self.args.mp_iters = self.args.mp_iters_gen if self.G else self.args.mp_iters_disc

        if(self.args.int_diffs and self.args.pos_diffs):
            self.args.fe_in_size = 2 * self.args.hidden_node_size + 2
        elif(self.args.int_diffs or self.args.pos_diffs):
            self.args.fe_in_size = 2 * self.args.hidden_node_size + 1
        else:
            self.args.fe_in_size = 2 * self.args.hidden_node_size

        self.args.fe_out_size = self.args.fe[-1]

        self.args.fe.insert(0, self.args.fe_in_size)
        self.args.fn.insert(0, self.args.fe_out_size + self.args.hidden_node_size)
        self.args.fn.append(self.args.hidden_node_size)

        if(self.args.dea):
            self.args.fnd.insert(0, self.args.hidden_node_size)
            self.args.fnd.append(1)

        self.fe = nn.ModuleList()
        self.fn = nn.ModuleList()

        if(self.args.batch_norm):
            self.bne = nn.ModuleList()
            self.bnn = nn.ModuleList()

        for i in range(self.args.mp_iters):
            # edge network
            fe_iter = nn.ModuleList()
            if self.args.batch_norm: bne = nn.ModuleList()
            for j in range(len(self.args.fe) - 1):
                linear = nn.Linear(self.args.fe[j], self.args.fe[j + 1])
                # if self.args.spectral_norm: linear = SpectralNorm(linear)
                # fe_iter.append(SpectralNorm(linear) if self.args.spectral_norm else linear)
                fe_iter.append(linear)
                if self.args.batch_norm: bne.append(nn.BatchNorm1d(self.args.fe[j + 1]))

            self.fe.append(fe_iter)
            if self.args.batch_norm: self.bne.append(bne)

            # node network
            fn_iter = nn.ModuleList()
            if self.args.batch_norm: bnn = nn.ModuleList()
            for j in range(len(self.args.fn) - 1):
                linear = nn.Linear(self.args.fn[j], self.args.fn[j + 1])
                # if self.args.spectral_norm: linear = SpectralNorm(linear)
                fn_iter.append(linear)
                if self.args.batch_norm: bnn.append(nn.BatchNorm1d(self.args.fn[j + 1]))

            self.fn.append(fn_iter)
            if self.args.batch_norm: self.bnn.append(bnn)

        if(self.args.dea):
            self.fnd = nn.ModuleList()
            self.bnd = nn.ModuleList()
            for i in range(len(self.args.fnd) - 1):
                linear = nn.Linear(self.args.fnd[i], self.args.fnd[i + 1])
                # if self.args.spectral_norm: linear = SpectralNorm(linear)
                self.fnd.append(linear)
                if self.args.batch_norm: self.bnd.append(nn.BatchNorm1d(self.args.fnd[i + 1]))

        p = self.args.gen_dropout if self.G else self.args.disc_dropout
        self.dropout = nn.Dropout(p=p)

        # self.init_params()

        if self.args.spectral_norm:
            for ml in self.fe:
                for i in range(len(ml)):
                    ml[i] = SpectralNorm(ml[i])

            for ml in self.fn:
                for i in range(len(ml)):
                    ml[i] = SpectralNorm(ml[i])

            if self.args.dea:
                for i in range(len(self.fnd)):
                    self.fnd[i] = SpectralNorm(self.fnd[i])
        # print("after")

        print("fe: ")
        print(self.fe)

        print("fn: ")
        print(self.fn)

        if(self.args.dea):
            print("fnd: ")
            print(self.fnd)

    def forward(self, x):
        batch_size = x.shape[0]

        if(self.D): x = F.pad(x, (0, self.args.hidden_node_size - self.args.node_feat_size, 0, 0, 0, 0))

        for i in range(self.args.mp_iters):

            # message passing
            A = self.getA(x, batch_size)

            for j in range(len(self.fe[i])):
                A = F.leaky_relu(self.fe[i][j](A), negative_slope=self.args.leaky_relu_alpha)
                if(self.args.batch_norm): A = self.bne[i][j](A)  # try before activation
                # if(self.args.spectral_norm): A = SpectralNorm(A)
                A = self.dropout(A)

            # message aggregation into new features
            A = A.view(batch_size, self.args.num_hits, self.args.num_hits, self.args.fe_out_size)
            A = torch.sum(A, 2) if self.args.sum else torch.mean(A, 2)
            x = torch.cat((A, x), 2).view(batch_size * self.args.num_hits, self.args.fe_out_size + self.args.hidden_node_size)

            for j in range(len(self.fn[i]) - 1):
                x = F.leaky_relu(self.fn[i][j](x), negative_slope=self.args.leaky_relu_alpha)
                if(self.args.batch_norm): x = self.bnn[i][j](x)
                # if(self.args.spectral_norm): x = SpectralNorm(x)
                x = self.dropout(x)

            x = self.dropout(self.fn[i][-1](x))
            x = x.view(batch_size, self.args.num_hits, self.args.hidden_node_size)

        # print(x)

        if(self.G):
            x = torch.tanh(x[:, :, :self.args.node_feat_size])
            return x
        else:
            if(self.args.dea):
                x = torch.sum(x, 1) if self.args.sum else torch.mean(x, 1)
                for i in range(len(self.fnd) - 1):
                    x = F.leaky_relu(self.fnd[i](x), negative_slope=self.args.leaky_relu_alpha)
                    if(self.args.batch_norm): x = self.bnd[i](x)
                    # if(self.args.spectral_norm): x = SpectralNorm(x)
                    x = self.dropout(x)
                x = self.dropout(self.fnd[-1](x))
            else:
                x = torch.sum(x[:, :, :1], 1) if self.args.sum else torch.mean(x[:, :, :1], 1)
                # print(    x)

            # print(x)
            # print(torch.sigmoid(x))
            # return x if (self.args.loss == 'w' or self.args.loss == 'hinge') else torch.sigmoid(x)
            return torch.sigmoid(x)

    def getA(self, x, batch_size):
        x1 = x.repeat(1, 1, self.args.num_hits).view(batch_size, self.args.num_hits * self.args.num_hits, self.args.hidden_node_size)
        x2 = x.repeat(1, self.args.num_hits, 1)

        if(self.args.int_diffs):
            dists = torch.norm(x2[:, :, :2] - x1[:, :, :2] + 1e-12, dim=2).unsqueeze(2)
            int_diffs = 1 - ((x2[:, :, 2] - x1[:, :, 2])).unsqueeze(2)
            A = (torch.cat((x1, x2, dists, int_diffs), 2)).view(batch_size * self.args.num_hits * self.args.num_hits, self.args.fe_in_size)
        elif(self.args.pos_diffs):
            dists = torch.norm(x2[:, :, :2] - x1[:, :, :2] + 1e-12, dim=2).unsqueeze(2)
            A = torch.cat((x1, x2, dists), 2).view(batch_size * self.args.num_hits * self.args.num_hits, self.args.fe_in_size)
        else:
            A = torch.cat((x1, x2), 2).view(batch_size * self.args.num_hits * self.args.num_hits, self.args.fe_in_size)

        return A

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # print(m)
                torch.nn.init.xavier_uniform(m.weight, 0.1)

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def assigntest(self, boom):
        self.test = boom

    def printtest(self):
        print("Test: ")
        print(self.test)


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

            A = torch.sum(A.view(batch_size, self.num_hits, self.num_hits, self.fe_out_size), 2)

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

            A = torch.sum(A.view(batch_size, self.num_hits, self.num_hits, self.fe_out_size), 2)
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


class GaussianGenerator(torch.nn.Module):
    def __init__(self, args):
        super(GaussianGenerator, self).__init__()
        self.args = args
        self.conv1 = GMMConv(self.args.channels[0], 5 * self.args.channels[1], dim=2, kernel_size=args.kernel_size)
        self.conv2 = GMMConv(self.args.channels[1], 3 * self.args.channels[2], dim=2, kernel_size=args.kernel_size)
        self.conv3 = GMMConv(self.args.channels[2], self.args.channels[3], dim=2, kernel_size=args.kernel_size)
        self.pos1fc1 = torch.nn.Linear(2 + self.args.channels[1], 128)
        self.pos1fc2 = torch.nn.Linear(128, 64)
        self.pos1fc3 = torch.nn.Linear(64, 2)
        self.pos2fc1 = torch.nn.Linear(2 + self.args.channels[2], 128)
        self.pos2fc2 = torch.nn.Linear(128, 64)
        self.pos2fc3 = torch.nn.Linear(64, 2)

    def forward(self, data):
        data.edge_index, data.edge_attr = self.getA(data.pos, 5)
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr)).view(-1, self.args.channels[1])

        p = torch.cat((data.pos.repeat(1, 5).view(-1, 2), data.x), 1)
        p = F.elu(self.pos1fc1(p))
        p = F.elu(self.pos1fc2(p))
        data.pos = F.elu(self.pos1fc3(p))

        data.edge_index, data.edge_attr = self.getA(data.pos, 25)
        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr)).view(-1, self.args.channels[2])

        p = torch.cat((data.pos.repeat(1, 3).view(-1, 2), data.x), 1)
        p = F.elu(self.pos2fc1(p))
        p = F.elu(self.pos2fc2(p))
        data.pos = F.elu(self.pos2fc3(p))

        data.edge_index, data.edge_attr = self.getA(data.pos, 75)
        data.x = F.tanh(self.conv3(data.x, data.edge_index, data.edge_attr)).view(-1, self.args.channels[3])

        return data

    def getA(self, pos, num_nodes):
        posb = pos.view(-1, num_nodes, 2)
        batch_size = posb.size(0)

        x1 = posb.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, 2)
        x2 = posb.repeat(1, num_nodes, 1)

        diff_norms = torch.norm(x2 - x1 + 1e-12, dim=2)
        norms = diff_norms.view(batch_size, num_nodes, num_nodes)
        neighborhood = torch.nonzero(norms < self.args.cutoff, as_tuple=False)
        neighborhood = neighborhood[neighborhood[:, 1] != neighborhood[:, 2]]  # remove self-loops
        edge_index = (neighborhood[:, 1:] + (neighborhood[:, 0] * num_nodes).view(-1, 1)).transpose(0, 1)

        row, col = edge_index
        edge_attr = (pos[col] - pos[row]) / self.args.cutoff + 0.5

        # print("A")
        #
        # print(edge_index.shape)
        # print(edge_index)
        # print(edge_index[:, -20:])
        #
        # print(edge_attr.shape)
        # print(edge_attr)

        return edge_index.contiguous(), edge_attr


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class MoNet(torch.nn.Module):
    def __init__(self, args):
        super(MoNet, self).__init__()
        self.args = args
        self.conv1 = GMMConv(1, 32, dim=2, kernel_size=self.args.kernel_size)
        self.conv2 = GMMConv(32, 64, dim=2, kernel_size=self.args.kernel_size)
        self.conv3 = GMMConv(64, 64, dim=2, kernel_size=self.args.kernel_size)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        row, col = data.edge_index
        data.edge_attr = (data.pos[col] - data.pos[row]) / (2 * self.args.cutoff) + 0.5

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        row, col = data.edge_index
        data.edge_attr = (data.pos[col] - data.pos[row]) / (2 * self.args.cutoff) + 0.5

        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))

        x = global_mean_pool(data.x, data.batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.args.disc_dropout)
        y = self.fc2(x)

        if(self.args.wgan):
            return y

        return torch.sigmoid(y)
