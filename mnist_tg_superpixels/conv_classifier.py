import setGPU

# from profile import profile
# from time import sleep

import torch
from torch_geometric.data import DataLoader
import torch.nn.functional as F

from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T

from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (graclus, max_pool, global_mean_pool)

from torch_geometric.nn import GMMConv

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from tqdm import tqdm

import os
from os import listdir
from os.path import join, isdir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.cuda.set_device(0)
torch.manual_seed(4)
torch.autograd.set_detect_anomaly(True)

#Have to specify 'name' and 'start_epoch' if True
LOAD_MODEL = False
TRAIN=False

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

class MoNet(torch.nn.Module):
    def __init__(self, kernel_size):
        super(MoNet, self).__init__()
        self.conv1 = GMMConv(1, 32, dim=2, kernel_size=kernel_size)
        self.conv2 = GMMConv(32, 64, dim=2, kernel_size=kernel_size)
        self.conv3 = GMMConv(64, 64, dim=2, kernel_size=kernel_size)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))

        x = global_mean_pool(data.x, data.batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)

def init_dirs(args):
    dirs = listdir('.')
    if('cmodels' not in dirs):
        os.mkdir('./cmodels')
    if('closses' not in dirs):
        os.mkdir('./closses')
    if('cargs' not in dirs):
        os.mkdir('./cargs')
    if('dataset' not in dirs):
        os.mkdir('./dataset')

    del dirs

    onlydirs = [f for f in listdir('cmodels/') if isdir(join('cmodels/', f))]
    if (args.name in onlydirs):
        print("name already used")
        # if(not LOAD_MODEL):
            # sys.exit()
    else:
        os.mkdir('./closses/' + args.name)
        os.mkdir('./cmodels/' + args.name)

    del onlydirs

    f = open("cargs/" + args.name + ".txt", "w+")
    f.write(str(args))
    f.close()

def main(args):
    init_dirs(args)

    train_dataset = MNISTSuperpixels("./dataset", True, pre_transform=T.Cartesian())
    test_dataset = MNISTSuperpixels("./dataset", False, pre_transform=T.Cartesian())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    C = MoNet(args.kernel_size).to(device)
    C_optimizer = torch.optim.Adam(C.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    C_scheduler = torch.optim.lr_scheduler.StepLR(C_optimizer, args.decay_step, gamma=args.lr_decay)

    train_losses = []
    test_losses = []

    def plot_losses(epoch, train_losses, test_losses):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(train_losses)
        ax1.set_title('training')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(test_losses)
        ax2.set_title('testing')

        plt.savefig("closses/"+ args.name +"/"+ str(epoch) + ".png")
        plt.close()

    def save_model(epoch):
        torch.save(C, "cmodels/" + args.name + "/C_" + str(epoch) + ".pt")

    def train_C(data, y):
        C.train()
        C_optimizer.zero_grad()

        output = C(data)

        #nll_loss takes class labels as target, so one-hot encoding is not needed
        C_loss = F.nll_loss(output, y)

        C_loss.backward()
        C_optimizer.step()

        return C_loss.item()

    def test():
        C.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                output = C(data.to(device))
                test_loss += F.nll_loss(output, data.y, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(data.y.data.view_as(pred)).sum()

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

    for i in range(args.num_epochs):
        print("Epoch %d %s" % ((i+1), args.name))
        C_loss = 0
        for batch_ndx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            C_loss += train_C(data.to(device), data.y.to(device))

        train_losses.append(C_loss/len(train_loader))
        C_scheduler.step()

        test()
        if((i+1)%10==0):
            save_model(args.name, i+1)
            plot_losses(args.name, i+1, train_losses, test_losses)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dropout", type=float, default=0.2, help="fraction of dropout")
    parser.add_argument("--leaky-relu-alpha", type=float, default=0.2, help="leaky relu alpha")

    parser.add_argument("--num-epochs", type=int, default=300, help="number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=0.99)
    parser.add_argument('--decay_step', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument("--kernel-size", type=int, default=10, help="graph convolutional layer kernel size")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")

    parser.add_argument("--name", type=str, default="41", help="name or tag for model; will be appended with other info")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
