import torch
import torch.nn as nn
import torch.nn.functional as F

class Graph_Generator(nn.Module):
    def __init__(self, node_size, fe_hidden_size, fe_out_size, hidden_size, num_gru_layers, iters, num_hits, dropout, alpha, hidden_node_size=64, int_diffs=False):
        super(Graph_Generator, self).__init__()
        self.node_size = node_size
        self.fe_hidden_size = fe_hidden_size
        self.fe_out_size = fe_out_size
        self.hidden_size = hidden_size
        self.num_hits = num_hits
        self.alpha = alpha
        self.num_gru_layers = num_gru_layers
        self.iters = iters
        self.hidden_node_size = hidden_node_size

        self.fe_in_size = 2*hidden_node_size+2 if int_diffs else 2*hidden_node_size+1
        self.use_int_diffs = int_diffs

        self.fe1 = nn.Linear(self.fe_in_size, fe_hidden_size)
        self.fe2 = nn.Linear(fe_hidden_size, fe_out_size)

        self.fn1 = GRU(fe_out_size + hidden_node_size, hidden_size, num_gru_layers, dropout)
        self.fn2 = nn.Linear(hidden_size, hidden_node_size)

    def forward(self, x):
        batch_size = x.shape[0]
        hidden = self.initHidden(batch_size)

        for i in range(self.iters):
            A = self.getA(x, batch_size)

            A = F.leaky_relu(self.fe1(A), negative_slope=self.alpha)
            A = F.leaky_relu(self.fe2(A), negative_slope=self.alpha)
            A = torch.sum(A.view(batch_size, self.num_hits, self.num_hits, self.fe_out_size), 2)

            x = torch.cat((A, x), 2)
            del A

            x = x.view(batch_size*self.num_hits, 1, self.fe_out_size + self.hidden_node_size)

            x, hidden = self.fn1(x, hidden)
            x = torch.tanh(self.fn2(x))
            x = x.view(batch_size, self.num_hits, self.hidden_node_size)

        x = x[:,:,:self.node_size]

        return x

    def getA(self, x, batch_size):
        x1 = x.repeat(1, 1, self.num_hits).view(batch_size, self.num_hits*self.num_hits, self.hidden_node_size)
        x2 = x.repeat(1, self.num_hits, 1)

        dists = torch.norm(x2[:, :, :2]-x1[:, :, :2], dim=2).unsqueeze(2)

        if(self.use_int_diffs):
            # int_diffs = ((x2[:, :, 2]-x1[:, :, 2])**2).unsqueeze(2)
            # A = ((1-int_diffs)*torch.cat((x1, x2, dists, int_diffs), 2)).view(batch_size*self.num_hits*self.num_hits, self.fe_in_size)
            int_diffs = ((x2[:, :, 2]-x1[:, :, 2])).unsqueeze(2)
            A = (torch.cat((x1, x2, dists, int_diffs), 2)).view(batch_size*self.num_hits*self.num_hits, self.fe_in_size)
        else:
            A = torch.cat((x1, x2, dists), 2).view(batch_size*self.num_hits*self.num_hits, self.fe_ins_size)
        return A

    def initHidden(self, batch_size):
        return torch.zeros(self.num_gru_layers, batch_size*self.num_hits, self.hidden_size).cuda()

class Graph_Discriminator(nn.Module):
    def __init__(self, node_size, fe_hidden_size, fe_out_size, hidden_size, num_gru_layers, iters, num_hits, dropout, alpha, hidden_node_size=64, wgan=False, int_diffs=False):
        super(Graph_Discriminator, self).__init__()
        self.node_size = node_size
        self.hidden_node_size = hidden_node_size
        self.fe_hidden_size = fe_hidden_size
        self.fe_out_size = fe_out_size
        self.num_hits = num_hits
        self.alpha = alpha
        self.dropout = dropout
        self.num_gru_layers = num_gru_layers
        self.hidden_size = hidden_size
        self.iters = iters
        self.wgan = wgan

        self.fe_in_size = 2*hidden_node_size+2 if int_diffs else 2*hidden_node_size+1
        self.use_int_diffs = int_diffs

        self.fe1 = nn.Linear(self.fe_in_size, fe_hidden_size)
        self.fe2 = nn.Linear(fe_hidden_size, fe_out_size)

        self.fn1 = GRU(fe_out_size + hidden_node_size, hidden_size, num_gru_layers, dropout)
        self.fn2 = nn.Linear(hidden_size, hidden_node_size)

    def forward(self, x):
        batch_size = x.shape[0]
        hidden = self.initHidden(batch_size)

        x = F.pad(x, (0,self.hidden_node_size - self.node_size,0,0,0,0))

        for i in range(self.iters):
            A = self.getA(x, batch_size)

            A = F.leaky_relu(self.fe1(A), negative_slope=self.alpha)
            A = F.leaky_relu(self.fe2(A), negative_slope=self.alpha)
            A = torch.sum(A.view(batch_size, self.num_hits, self.num_hits, self.fe_out_size), 2)

            x = torch.cat((A, x), 2)
            del A

            x = x.view(batch_size*self.num_hits, 1, self.fe_out_size + self.hidden_node_size)

            x, hidden = self.fn1(x, hidden)
            x = torch.tanh(self.fn2(x))
            x = x.view(batch_size, self.num_hits, self.hidden_node_size)

        x = torch.mean(x[:,:,:1], 1)

        if(self.wgan):
            return x
        return torch.sigmoid(x)

    def getA(self, x, batch_size):
        x1 = x.repeat(1, 1, self.num_hits).view(batch_size, self.num_hits*self.num_hits, self.hidden_node_size)
        x2 = x.repeat(1, self.num_hits, 1)

        dists = torch.norm(x2[:, :, :2]-x1[:, :, :2], dim=2).unsqueeze(2)

        if(self.use_int_diffs):
            # int_diffs = ((x2[:, :, 2]-x1[:, :, 2])**2).unsqueeze(2)
            # A = ((1-int_diffs)*torch.cat((x1, x2, dists, int_diffs), 2)).view(batch_size*self.num_hits*self.num_hits, self.fe_in_size)
            int_diffs = ((x2[:, :, 2]-x1[:, :, 2])).unsqueeze(2)
            A = (torch.cat((x1, x2, dists, int_diffs), 2)).view(batch_size*self.num_hits*self.num_hits, self.fe_in_size)
        else:
            A = torch.cat((x1, x2, dists), 2).view(batch_size*self.num_hits*self.num_hits, self.fe_ins_size)

        return A

    def initHidden(self, batch_size):
        return torch.zeros(self.num_gru_layers, batch_size*self.num_hits, self.hidden_size).cuda()

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()

        self.layers.append(GRUCell(input_size, hidden_size))
        for i in range(num_layers - 1):
            self.layers.append(GRUCell(hidden_size, hidden_size))

    def forward(self, x, hidden):
        x = x.squeeze()
        hidden[0] = F.dropout(self.layers[0](x, hidden[0].clone()), p = self.dropout)

        for i in range(1, self.num_layers):
            hidden[i] = F.dropout(self.layers[i](hidden[i-1].clone(), hidden[i].clone()), p = self.dropout)

        return hidden[-1].unsqueeze(1).clone(), hidden

class GRUCell(nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
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
