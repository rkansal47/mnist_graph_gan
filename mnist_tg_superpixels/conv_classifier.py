#import setGPU
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

from os import listdir, mkdir
from os.path import exists, dirname, realpath

import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.cuda.set_device(0)
torch.manual_seed(4)
torch.autograd.set_detect_anomaly(True)

#Have to specify 'name' and 'start_epoch' if True
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
        row, col = data.edge_index
        data.edge_attr = (data.pos[col]-data.pos[row])/(2*28*cutoff) + 0.5

        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        row, col = data.edge_index
        data.edge_attr = (data.pos[col]-data.pos[row])/(2*28*cutoff) + 0.5

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        row, col = data.edge_index
        data.edge_attr = (data.pos[col]-data.pos[row])/(2*28*cutoff) + 0.5

        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))

        x = global_mean_pool(data.x, data.batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)

def init_dirs(args):
    args.model_path = args.dir_path + '/cmodels/'
    args.losses_path = args.dir_path + '/closses/'
    args.args_path = args.dir_path + '/cargs/'
    args.out_path = args.dir_path + '/cout/'
    args.dataset_path = args.dir_path + '/dataset/cartesian/' if args.cartesian else args.dir_path + '/dataset/polar/'

    if(not exists(args.model_path)):
        mkdir(args.model_path)
    if(not exists(args.losses_path)):
        mkdir(args.losses_path)
    if(not exists(args.args_path)):
        mkdir(args.args_path)
    if(not exists(args.out_path)):
        mkdir(args.out_path)
    if(not exists(args.dataset_path)):
        mkdir(args.dir_path + '/dataset')
        mkdir(args.dir_path + '/dataset/cartesian')
        mkdir(args.dir_path + '/dataset/polar')

    prev_models = [f[:-4] for f in listdir(args.args_path)] #removing txt part

    if (args.name in prev_models):
        print("name already used")
        # if(not args.load_model):
        #     sys.exit()
    else:
        mkdir(args.losses_path + args.name)
        mkdir(args.model_path + args.name)

    if(not args.load_model):
        f = open(args.args_path + args.name + ".txt", "w+")
        f.write(str(vars(args)))
        f.close()
        return args
    else:
        # f = open(args.args_path + args.name + ".txt", "r")
        # args2 = eval(f.read())
        # f.close()
        # args2.load_model = True
        # args2.start_epoch = args.start_epoch
        # return args2
        return args

def main(args):
    args = init_dirs(args)

    pt = T.Cartesian() if args.cartesian else T.Polar()

    train_dataset = MNISTSuperpixels(args.dataset_path, True, pre_transform=pt)
    test_dataset = MNISTSuperpixels(args.dataset_path, False, pre_transform=pt)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    if(args.load_model):
        C = torch.load(args.model_path + name + "/C_" + str(start_epoch) + ".pt").to(device)
    else:
        C = MoNet(args.kernel_size).to(device)

    C_optimizer = torch.optim.Adam(C.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if(args.scheduler):
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

        plt.savefig(args.losses_path + args.name +"/"+ str(epoch) + ".png")
        plt.close()

    def save_model(epoch):
        torch.save(C, args.model_path + args.name + "/C_" + str(epoch) + ".pt")

    def train_C(data, y):
        C.train()
        C_optimizer.zero_grad()

        output = C(data)

        #nll_loss takes class labels as target, so one-hot encoding is not needed
        C_loss = F.nll_loss(output, y)

        C_loss.backward()
        C_optimizer.step()

        return C_loss.item()

    def test(epoch):
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

        f = open(args.out_path + args.name + '.txt', 'a')
        s = "After {} epochs, on test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(epoch, test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset))
        f.write(s)
        f.close()


    for i in range(args.start_epoch, args.num_epochs):
        print("Epoch %d %s" % ((i+1), args.name))
        C_loss = 0
        test(i)
        for batch_ndx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            C_loss += train_C(data.to(device), data.y.to(device))

        train_losses.append(C_loss/len(train_loader))

        if(args.scheduler):
            C_scheduler.step()

        if((i+1)%10==0):
            save_model(i+1)
            plot_losses(i+1, train_losses, test_losses)

    test(args.num_epochs)

def add_bool_arg(parser, name, help, default=False):
    varname = '_'.join(name.split('-')) # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=varname, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=varname, action='store_false', help="don't " + help)
    parser.set_defaults(**{varname:default})

def parse_args():
    import argparse

    dir_path = dirname(realpath(__file__))

    parser = argparse.ArgumentParser()

    parser.add_argument("--dir-path", type=str, default=dir_path, help="path where dataset and output will be stored")

    add_bool_arg(parser, "load-model", "load a pretrained model", default=False)
    parser.add_argument("--start-epoch", type=int, default=0, help="which epoch to start training on (only makes sense if loading a model)")

    parser.add_argument("--dropout", type=float, default=0.5, help="fraction of dropout")

    parser.add_argument("--num-epochs", type=int, default=300, help="number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-decay', type=float, default=0.99)
    parser.add_argument('--decay-step', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    add_bool_arg(parser, "scheduler", "use a optimization scheduler", default=True)

    parse_coords = parser.add_mutually_exclusive_group(required=False)
    parse_coords.add_argument('--cartesian', dest="cartesian", action='store_true', help="use cartesian coordinates")
    parse_coords.add_argument('--polar', dest="cartesian", action='store_false', help="use polar coordinates")
    parser.set_defaults(cartesian=True)

    parser.add_argument("--kernel-size", type=int, default=25, help="graph convolutional layer kernel size")
    parser.add_argument("--batch-size", type=int, default=10, help="batch size")

    parser.add_argument("--name", type=str, default="test", help="name or tag for model; will be appended with other info")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
