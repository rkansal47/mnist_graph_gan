import logging

import torch
from torch import nn
import itertools

from torch.autograd import Variable

import numpy as np


# from https://github.com/jmduarte/JEDInet-code/blob/master/python/JetImageClassifier_IN_FinalTraining.py
class JEDINet(nn.Module):
    def __init__(self, device, n_constituents=30, n_targets=5, node_feat_size=3, hidden=50, De=12, Do=6,
                 fr_activation=0, fo_activation=2, fc_activation=2, optimizer=0, verbose=False):
        super(JEDINet, self).__init__()
        self.hidden = hidden
        self.P = node_feat_size
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.n_targets = n_targets
        self.fr_activation = fr_activation
        self.fo_activation = fo_activation
        self.fc_activation = fc_activation
        self.optimizer = optimizer
        self.verbose = verbose
        self.device = device
        self.assign_matrices()

        self.sum_O = 0
        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden).to(self.device)
        self.fr2 = nn.Linear(self.hidden, int(self.hidden / 2)).to(self.device)
        self.fr3 = nn.Linear(int(self.hidden / 2), self.De).to(self.device)
        self.fo1 = nn.Linear(self.P + self.Dx + self.De, self.hidden).to(self.device)
        self.fo2 = nn.Linear(self.hidden, int(self.hidden / 2)).to(self.device)
        self.fo3 = nn.Linear(int(self.hidden / 2), self.Do).to(self.device)
        if self.sum_O:
            self.fc1 = nn.Linear(self.Do * 1, self.hidden).to(self.device)
        else:
            self.fc1 = nn.Linear(self.Do * self.N, self.hidden).to(self.device)
        self.fc2 = nn.Linear(self.hidden, int(self.hidden / 2)).to(self.device)
        self.fc3 = nn.Linear(int(self.hidden / 2), self.n_targets).to(self.device)

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = Variable(self.Rr).to(self.device)
        self.Rs = Variable(self.Rs).to(self.device)

    def forward(self, x):
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        if self.fr_activation == 2:
            B = nn.functional.selu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.selu(self.fr2(B))
            E = nn.functional.selu(self.fr3(B).view(-1, self.Nr, self.De))
        elif self.fr_activation == 1:
            B = nn.functional.elu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.elu(self.fr2(B))
            E = nn.functional.elu(self.fr3(B).view(-1, self.Nr, self.De))
        else:
            B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.relu(self.fr2(B))
            E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        C = torch.cat([x, Ebar], 1)
        del Ebar
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        if self.fo_activation == 2:
            C = nn.functional.selu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.selu(self.fo2(C))
            O = nn.functional.selu(self.fo3(C).view(-1, self.N, self.Do))
        elif self.fo_activation == 1:
            C = nn.functional.elu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.elu(self.fo2(C))
            O = nn.functional.elu(self.fo3(C).view(-1, self.N, self.Do))
        else:
            C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.relu(self.fo2(C))
            O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C

        ## sum over the O matrix
        if self.sum_O:
            O = torch.sum(O, dim=1)

        ### Classification MLP ###
        if self.fc_activation == 2:
            if self.sum_O:
                N = nn.functional.selu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.selu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.selu(self.fc2(N))
        elif self.fc_activation ==1:
            if self.sum_O:
                N = nn.functional.elu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.elu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.elu(self.fc2(N))
        else:
            if self.sum_O:
                N = nn.functional.relu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.relu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.relu(self.fc2(N))
        del O

        #N = nn.functional.relu(self.fc3(N))
        N = self.fc3(N)
        return N

    def tmul(self, x, y):  # Takes (I * J * K)(K * L) -> I * J * L
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.reshape(-1, x_shape[2]), y).reshape(-1, x_shape[1], y_shape[1])
