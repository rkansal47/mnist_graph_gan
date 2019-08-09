import torch
import torch.nn as nn
import torch.nn.functional as F
from gc_layer import GraphConvolution

class GCN_classifier(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, adjoint_scaling=29):
        super(GCN_classifier, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.adjoint_scaling = adjoint_scaling

    def forward(self, x):
        print("Getting adj")
        adj = self.get_adj(x)
        print("Got adj")
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def get_adj(self, x):
        dim = x.shape[1]
        # print(x.shape)
        adj = torch.zeros((len(x), dim, dim)).cuda()

        for n in range(len(x)):
            for i in range(dim):
                for j in range(i+1, dim):
                    adj[n, i, j] = self.inv_dist(x[n, i], x[n, j])
            adj[n] += adj[n].t() + torch.eye(dim).cuda()
        return adj

    def inv_dist(self, x1, x2):
        # print(x1)
        # print(x2)
        dist = torch.sqrt((x2[0]-x1[0])**2 + (x2[1]-x1[1])**2)
        return 1.0/dist/self.adjoint_scaling
