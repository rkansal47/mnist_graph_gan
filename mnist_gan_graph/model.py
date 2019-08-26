import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple_GRU(nn.Module):
    def __init__(self, node_size, fe_out_size, hidden_size, num_gru_layers, iters, num_hits, dropout, alpha, hidden_node_size=64):
        super(Simple_GRU, self).__init__()
        self.node_size = node_size
        self.fe_out_size = fe_out_size
        self.hidden_size = hidden_size
        self.num_hits = num_hits
        self.alpha = alpha
        self.num_gru_layers = num_gru_layers
        self.iters = iters
        self.hidden_node_size = hidden_node_size

        self.fe1 = nn.ModuleList()
        self.fe2 = nn.ModuleList()
        for i in range(iters):
            self.fe1.append(nn.Linear(2*node_size + 1, 128))
            self.fe2.append(nn.Linear(128, fe_out_size))

        self.fn1 = GRU(fe_out_size + node_size, hidden_size, num_gru_layers, batch_first=True, dropout=dropout)
        self.fn2 = nn.Linear(hidden_size, node_size)

    def forward(self, x):
        batch_size = x.shape[0]
        hidden = self.initHidden(batch_size)

        for i in range(self.iters):
            x1 = x.repeat(1, 1, self.num_hits).view(batch_size, self.num_hits*self.num_hits, self.hidden_node_size)
            x2 = x.repeat(1, self.num_hits, 1)

            norms = torch.norm(x2[:, :, :2]-x1[:, :, :2], dim=2).unsqueeze(2)
            pairs = torch.cat((x1, x2, norms), 2).view(batch_size*self.num_hits*self.num_hits, 2*self.hidden_node_size + 1)

            del x1
            del x2
            del norms

            av = F.leaky_relu(self.fe1(pairs), negative_slope=self.alpha)

            del pairs

            av = F.leaky_relu(self.fe2(av), negative_slope=self.alpha)

            av = torch.sum(av.view(batch_size, self.num_hits, self.num_hits, self.fe_out_size), 2)

            x = torch.cat((av, x), 2)
            del av

            x = x.view(batch_size*self.num_hits, 1, self.fe_out_size + self.hidden_node_size)

            x, hidden = self.fn1(x, hidden)
            x = torch.tanh(self.fn2(x))
            x = x.view(batch_size, self.num_hits, self.hidden_node_size)

        x = x[:,:,:self.node_size]

        return x

    def initHidden(self, batch_size):
        return torch.zeros(self.num_gru_layers, batch_size*self.num_hits, self.hidden_size).cuda()

class Graph_Discriminator(nn.Module):
    def __init__(self, node_size, fe_out_size, hidden_size, num_gru_layers, iters, num_hits, dropout, alpha, same_params, hidden_node_size=64, wgan=False):
        super(Graph_Discriminator, self).__init__()
        self.node_size = node_size
        self.hidden_node_size = hidden_node_size
        self.fe_out_size = fe_out_size
        self.num_hits = num_hits
        self.alpha = alpha
        self.dropout = dropout
        self.same_params = same_params
        self.num_gru_layers = num_gru_layers
        self.hidden_size = hidden_size
        self.iters = iters
        self.wgan = wgan

        if(same_params):
            # self.fe11 = nn.Linear(2*node_size + 1, 128)
            # self.fe21 = nn.Linear(128, fe_out_size)
            #
            # self.fn11 = nn.Linear(fe_out_size + node_size, hidden_size)
            # self.fn21 = nn.Linear(hidden_size, self.hidden_node_size-node_size)
            #
            # self.fe12 = nn.Linear(2*hidden_node_size + 1, 128)
            # self.fe22 = nn.Linear(128, fe_out_size)
            #
            # self.fn12 = nn.GRU(fe_out_size + hidden_node_size, hidden_size, num_gru_layers, batch_first=True, dropout=dropout)
            # self.fn22 = nn.Linear(hidden_size, 1)

            self.fe1 = nn.Linear(2*hidden_node_size + 1, 128)
            self.fe2 = nn.Linear(128, fe_out_size)

            self.fn1 = GRU(fe_out_size + hidden_node_size, hidden_size, num_gru_layers, batch_first=True, dropout=dropout)
            self.fn2 = nn.Linear(hidden_size, hidden_node_size)
        else:
            self.fe1 = nn.ModuleList()
            self.fe2 = nn.ModuleList()

            self.fn1 = nn.ModuleList()
            self.fn2 = nn.ModuleList()

            self.node_sizes = [self.node_size, 16, 32]

            self.fe1.append(nn.Linear(2*node_size + 1, 128))
            self.fe2.append(nn.Linear(128, fe_out_size))

            self.fn1.append(nn.Linear(fe_out_size + node_size, 128))
            self.fn2.append(nn.Linear(128, self.node_sizes[1]-node_size))

            self.fe1.append(nn.Linear(2*self.node_sizes[1]+1, 128))
            self.fe2.append(nn.Linear(128, fe_out_size))

            self.fn1.append(nn.Linear(fe_out_size + self.node_sizes[1], 128))
            self.fn2.append(nn.Linear(128, self.node_sizes[2]-node_size))

            self.fe1.append(nn.Linear(2*self.node_sizes[2]+1, 128))
            self.fe2.append(nn.Linear(128, fe_out_size))

            self.fn1.append(nn.Linear(fe_out_size + self.node_sizes[2], 128))
            self.fn2.append(nn.Linear(128, 1))

    def forward(self, x):
        batch_size = x.shape[0]
        hidden = self.initHidden(batch_size)

        if(self.same_params):
            # x1 = x.repeat(1, 1, self.num_hits).view(batch_size, self.num_hits*self.num_hits, self.node_size)
            # x2 = x.repeat(1, self.num_hits, 1)
            # norms = torch.norm(x2[:, :, :2]-x1[:, :, :2], dim=2).unsqueeze(2)
            # av = torch.cat((x1, x2, norms), 2).view(batch_size*self.num_hits*self.num_hits, 2*self.node_size + 1)
            # del x1
            # del x2
            # del norms
            # av = F.dropout(F.leaky_relu(self.fe11(av), negative_slope=self.alpha), p=self.dropout)
            # av = F.dropout(F.leaky_relu(self.fe21(av), negative_slope=self.alpha), p=self.dropout)
            # av = torch.sum(av.view(batch_size, self.num_hits, self.num_hits, self.fe_out_size), 2)
            # y = torch.cat((av, x), 2)
            # del av
            # y = F.dropout(F.leaky_relu(self.fn11(y), negative_slope=self.alpha), p=self.dropout)
            # y = F.dropout(F.leaky_relu(self.fn21(y), negative_slope=self.alpha), p=self.dropout)
            # y = y.view(batch_size, self.num_hits, self.hidden_node_size - self.node_size)
            # x = torch.cat((x[:,:,:self.node_size], y), 2)
            #
            # for i in range(self.iters):
            #     x1 = x.repeat(1, 1, self.num_hits).view(batch_size, self.num_hits*self.num_hits, self.hidden_node_size)
            #     x2 = x.repeat(1, self.num_hits, 1)
            #     norms = torch.norm(x2[:, :, :2]-x1[:, :, :2], dim=2).unsqueeze(2)
            #     av = torch.cat((x1, x2, norms), 2).view(batch_size*self.num_hits*self.num_hits, 2*self.hidden_node_size + 1)
            #     del x1
            #     del x2
            #     del norms
            #     av = F.dropout(F.leaky_relu(self.fe12(av), negative_slope=self.alpha), p=self.dropout)
            #     av = F.dropout(F.leaky_relu(self.fe22(av), negative_slope=self.alpha), p=self.dropout)
            #     av = torch.sum(av.view(batch_size, self.num_hits, self.num_hits, self.fe_out_size), 2)
            #     y = torch.cat((av, x), 2)
            #     del av
            #     y, hidden = self.fn12(y, hidden)
            #
            #     if(i<self.iters-1):
            #         y = F.dropout(F.leaky_relu(self.fn21(y), negative_slope=self.alpha), p=self.dropout)
            #         y = y.view(batch_size, self.num_hits, self.hidden_node_size - self.node_size)
            #         x = torch.cat((x[:,:,:self.node_size], y), 2)
            #     else:
            #         y = F.dropout(F.tanh(self.fn22(y)), p=self.dropout)
            #         x = y.view(batch_size, self.num_hits)

            # print(x)
            x = F.pad(x, (0,self.hidden_node_size - self.node_size,0,0,0,0))
            # print(x)

            for i in range(self.iters):
                x1 = x.repeat(1, 1, self.num_hits).view(batch_size, self.num_hits*self.num_hits, self.hidden_node_size)
                x2 = x.repeat(1, self.num_hits, 1)

                norms = torch.norm(x2[:, :, :2]-x1[:, :, :2], dim=2).unsqueeze(2)
                pairs = torch.cat((x1, x2, norms), 2).view(batch_size*self.num_hits*self.num_hits, 2*self.hidden_node_size + 1)

                del x1
                del x2
                del norms

                av = F.leaky_relu(self.fe1(pairs), negative_slope=self.alpha)

                del pairs

                av = F.leaky_relu(self.fe2(av), negative_slope=self.alpha)

                av = torch.sum(av.view(batch_size, self.num_hits, self.num_hits, self.fe_out_size), 2)

                x = torch.cat((av, x), 2)
                del av

                x = x.view(batch_size*self.num_hits, 1, self.fe_out_size + self.hidden_node_size)

                # print(x)
                # print(x.shape)

                x, hidden = self.fn1(x, hidden)
                x = torch.tanh(self.fn2(x))
                x = x.view(batch_size, self.num_hits, self.hidden_node_size)

            x = torch.mean(x[:,:,:1], 1)
            # print(x)

            if(self.wgan):
                return x
            return torch.sigmoid(x)

        else:
            for i in range(self.iters):
                x1 = x.repeat(1, 1, self.num_hits).view(batch_size, self.num_hits*self.num_hits, self.node_sizes[i])
                x2 = x.repeat(1, self.num_hits, 1)

                norms = torch.norm(x2[:, :, :2]-x1[:, :, :2], dim=2).unsqueeze(2)
                av = torch.cat((x1, x2, norms), 2).view(batch_size*self.num_hits*self.num_hits, 2*self.node_sizes[i] + 1)

                del x1
                del x2
                del norms

                av = F.dropout(F.leaky_relu(self.fe1[i](av), negative_slope=self.alpha), p=self.dropout)
                av = F.dropout(F.leaky_relu(self.fe2[i](av), negative_slope=self.alpha), p=self.dropout)

                # av = F.dropout(torch.tanh(self.fe1[i](av)), p=self.dropout)
                # av = F.dropout(torch.tanh(self.fe2[i](av)), p=self.dropout)

                av = torch.sum(av.view(batch_size, self.num_hits, self.num_hits, self.fe_out_size), 2)

                y = torch.cat((av, x), 2)

                del av

                # y = F.dropout(F.leaky_relu(self.fn1[i](y), negative_slope=self.alpha), p=self.dropout)
                # y = F.leaky_relu(self.fn2[i](y), negative_slope=self.alpha)

                y = F.dropout(torch.tanh(self.fn1[i](y)), p=self.dropout)
                y = torch.tanh(self.fn2[i](y))

                if(i<self.iters-1):
                    y = y.view(batch_size, self.num_hits, self.node_sizes[i+1] - self.node_size)
                    x = torch.cat((x[:,:,:2], y), 2)
                else:
                    x = y.view(batch_size, self.num_hits)

        x = torch.sigmoid(torch.mean(x, 1, keepdim=True))

        return x

    def initHidden(self, batch_size):
        return torch.zeros(self.num_gru_layers, batch_size*self.num_hits, self.hidden_size).cuda()

class Critic(nn.Module):
    def __init__(self, input_shape, dropout, batch_size, wgan=False):
        super(Critic, self).__init__()
        self.batch_size = batch_size

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(input_shape[1], 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.out = nn.Linear(64*input_shape[0]/4, 1)
        self.wgan = wgan
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # x = self.sort(input)
        x = input
        x = x.permute(0, 2, 1)
        x = self.max_pool(self.leaky_relu(self.conv1(x)))
        x = self.max_pool(self.leaky_relu(self.conv2(x)))
        x = x.reshape(input.shape[0], -1)
        x = self.out(x)
        if(self.wgan):
            return x
        else:
            return self.sigmoid(x)

    def sort(self, x):
        s, indx = torch.sort(x[:, :, 0])
        for i in range(len(x)):
            x[i] = x[i][indx[i]]
        return x

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUCell, self).__init__()
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
        hidden[:, 0] = F.dropout(self.layers[0](x, hidden[:, 0]), p = self.dropout)

        for i in range(1, self.num_layers):
            hidden[:, i] = F.dropout(self.layers[i](hidden[:, i-1], hidden[:, i]), p = self.dropout)

        return hidden[:, -1].unsqueeze(1), hidden

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
        std = 1.0 / torch.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):

        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy
