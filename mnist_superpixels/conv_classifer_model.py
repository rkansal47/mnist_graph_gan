import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class Gaussian_Classifier(nn.Module):
    def __init__(self, node_size, fe_hidden_size, fe_out_size, mp_hidden_size, mp_num_layers, iters, num_hits, dropout, alpha, kernel_size, hidden_node_size=64, wgan=False, int_diffs=False, gru=False):
        super(Gaussian_Classifier, self).__init__()
        self.node_size = node_size
        self.hidden_node_size = hidden_node_size
        self.fe_hidden_size = fe_hidden_size
        self.fe_out_size = fe_out_size
        self.num_hits = num_hits
        self.alpha = alpha
        self.dropout = dropout
        self.mp_num_layers = mp_num_layers
        self.mp_hidden_size = mp_hidden_size
        self.iters = iters
        self.wgan = wgan
        self.gru = gru
        self.kernel_size = kernel_size

        self.fn = nn.Linear(hidden_node_size, hidden_node_size)
        self.fc1 = nn.Linear(hidden_node_size, 50)
        self.fc2 = nn.Linear(50, 10)

        self.mu = Parameter(torch.Tensor(kernel_size, 2).cuda())
        self.sigma = Parameter(torch.Tensor(kernel_size, 2).cuda())

        self.kernel_weight = Parameter(torch.Tensor(kernel_size).cuda())

        self.glorot(self.mu)
        self.glorot(self.sigma)
        self.kernel_weight.data.uniform_(0, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.pad(x, (0,self.hidden_node_size - self.node_size,0,0,0,0))

        for i in range(self.iters):
            x1 = x.repeat(1, 1, self.num_hits).view(batch_size, self.num_hits*self.num_hits, self.hidden_node_size)
            y = x.repeat(1, self.num_hits, 1)

            u = y[:,:,:2]-x1[:,:,:2]
            y = self.fn(y)
            y2 = torch.zeros(y.shape).cuda()

            for j in range(self.kernel_size):
                w = self.weights(u, j)
                y2 += w.unsqueeze(-1)*self.kernel_weight[j]*y

            x = torch.sum(y2.view(batch_size, self.num_hits, self.num_hits, self.hidden_node_size), 2)
            x = x.view(batch_size, self.num_hits, self.hidden_node_size)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # print(x)
        # print(x.shape)
        x = torch.mean(x, 1)
        # print(x)
        return F.log_softmax(x)

    def weights(self, u, j):
        return torch.exp(torch.sum((u-self.mu[j])**2*self.sigma[j], dim=-1))

    def initHidden(self, batch_size):
        return torch.zeros(self.mp_num_layers, batch_size*self.num_hits, self.mp_hidden_size).cuda()

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def zeros(self, tensor):
        if tensor is not None:
            tensor.data.fill_(0)
