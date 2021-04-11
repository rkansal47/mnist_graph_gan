import torch.nn as nn
import logging


class rGANG(nn.Module):
    def __init__(self, args):
        super(rGANG, self).__init__()
        self.args = args

        layers = []
        for i in range(len(self.args.rgang_fc) - 1):
            layers.append(nn.Linear(self.args.rgang_fc[i], self.args.rgang_fc[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=self.args.leaky_relu_alpha))

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

        self.args.rgand_fc.insert(0, self.args.num_hits * self.args.node_feat_size)

        layers = []
        for i in range(len(self.args.rgand_fc) - 1):
            layers.append(nn.Linear(self.args.rgand_fc[i], self.args.rgand_fc[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=self.args.leaky_relu_alpha))

        layers.append(nn.Linear(self.args.rgand_fc[-1], 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

        logging.info("rGAN discriminator: \n {}".format(self.model))


    def forward(self, x, labels=None, epoch=None):
        return self.model(x.reshape(-1, self.args.num_hits * self.args.node_feat_size))
