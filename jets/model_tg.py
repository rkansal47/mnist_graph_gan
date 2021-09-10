import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import MessagePassing

class MP(MessagePassing):
    def __init(self, args, fe, fn, dropout, aggr='add'):
        super(MP, self).__init__(aggr=aggr)  # "Add" aggregation (Step 5).

        fe_layers = []
        for i in range(len(fe) - 1):
            fe_layers.append(nn.Linear(fe[i], fe[i + 1]))
            fe_layers.append(nn.LeakyReLU(negative_slope=args.leaky_relu_alpha))
            fe_layers.append(nn.Dropout(p=dropout))

        self.fe_mlp = torch.Sequential(*fe_layers)

        fn_layers = []
        for i in range(len(fn) - 1):
            fe_layers.append(nn.Linear(fn[i], fn[i + 1]))
            fe_layers.append(nn.LeakyReLU(negative_slope=args.leaky_relu_alpha))
            fe_layers.append(nn.Dropout(p=dropout))

        self.fn_mlp = torch.Sequential(*fn_layers)

    def forward(self, x):
        

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)
        return self.fe_mlp(tmp)
