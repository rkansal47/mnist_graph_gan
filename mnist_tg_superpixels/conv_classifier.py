# import setGPU
import torch
from torch_geometric.data import DataLoader as tgDataLoader
import torch.nn.functional as F

from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T

from graph_dataset_mnist import MNISTGraphDataset
from torch.utils.data import DataLoader

from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (graclus, max_pool, global_mean_pool)
from torch_geometric.nn import GMMConv

import matplotlib.pyplot as plt

from tqdm import tqdm

from os import listdir, mkdir
from os.path import exists, dirname, realpath

from torch_geometric.data import Batch, Data

import sys

plt.switch_backend('agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.cuda.set_device(0)
torch.manual_seed(4)
torch.autograd.set_detect_anomaly(True)

# Have to specify 'name' and 'start_epoch' if True
TRAIN = False

cutoff = 0.32178


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


# transform my format to torch_geometric's
def tg_transform(args, X):
    batch_size = X.size(0)

    pos = X[:, :, :2]

    x1 = pos.repeat(1, 1, args.num_hits).reshape(batch_size, args.num_hits * args.num_hits, 2)
    x2 = pos.repeat(1, args.num_hits, 1)

    diff_norms = torch.norm(x2 - x1 + 1e-12, dim=2)

    norms = diff_norms.reshape(batch_size, args.num_hits, args.num_hits)
    neighborhood = torch.nonzero(norms < args.cutoff, as_tuple=False)

    neighborhood = neighborhood[neighborhood[:, 1] != neighborhood[:, 2]]  # remove self-loops
    unique, counts = torch.unique(neighborhood[:, 0], return_counts=True)
    edge_index = (neighborhood[:, 1:] + (neighborhood[:, 0] * args.num_hits).view(-1, 1)).transpose(0, 1)

    x = X[:, :, 2].reshape(batch_size * args.num_hits, 1) + 0.5
    pos = 28 * pos.reshape(batch_size * args.num_hits, 2) + 14

    row, col = edge_index
    edge_attr = (pos[col] - pos[row]) / (2 * 28 * args.cutoff) + 0.5

    zeros = torch.zeros(batch_size * args.num_hits, dtype=int).to(args.device)
    zeros[torch.arange(batch_size) * args.num_hits] = 1
    batch = torch.cumsum(zeros, 0) - 1

    return Batch(batch=batch, x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr, y=None, pos=pos)


def parse_args():
    import argparse

    dir_path = dirname(realpath(__file__))

    parser = argparse.ArgumentParser()

    parser.add_argument("--dir-path", type=str, default=dir_path, help="path where dataset and output will be stored")
    add_bool_arg(parser, "n", "run on nautilus cluster", default=False)

    add_bool_arg(parser, "load-model", "load a pretrained model", default=False)
    parser.add_argument("--start-epoch", type=int, default=0, help="which epoch to start training on (only makes sense if loading a model)")

    parser.add_argument("--dataset", type=str, default="sp", help="sp = superpixels, sm = sparse mnist, jets = jets obviously")

    parser.add_argument("--num_hits", type=int, default=75, help="num nodes in graph")

    parser.add_argument("--dropout", type=float, default=0.5, help="fraction of dropout")

    parser.add_argument("--num-epochs", type=int, default=300, help="number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-decay', type=float, default=0.99)
    parser.add_argument('--decay-step', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    add_bool_arg(parser, "scheduler", "use a optimization scheduler", default=False)

    parse_coords = parser.add_mutually_exclusive_group(required=False)
    parse_coords.add_argument('--cartesian', dest="cartesian", action='store_true', help="use cartesian coordinates")
    parse_coords.add_argument('--polar', dest="cartesian", action='store_false', help="use polar coordinates")
    parser.set_defaults(cartesian=True)

    parser.add_argument("--kernel-size", type=int, default=25, help="graph convolutional layer kernel size")
    parser.add_argument("--batch-size", type=int, default=10, help="batch size")

    parser.add_argument("--cutoff", type=float, default=0.32178, help="cutoff edge distance")  # found empirically to match closest to Superpixels

    parser.add_argument("--name", type=str, default="test", help="name or tag for model; will be appended with other info")
    args = parser.parse_args()

    if(args.n):
        args.dir_path = "/graphganvol/mnist_graph_gan/mnist_tg_superpixels"

    return args


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
        data.edge_attr = (data.pos[col] - data.pos[row]) / (2 * 28 * cutoff) + 0.5

        # print(data.edge_index.shape)
        # print(data.edge_index[:, -20:])

        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        row, col = data.edge_index
        data.edge_attr = (data.pos[col] - data.pos[row]) / (2 * 28 * cutoff) + 0.5

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        row, col = data.edge_index
        data.edge_attr = (data.pos[col] - data.pos[row]) / (2 * 28 * cutoff) + 0.5

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
    if args.dataset == 'sp':
        args.dataset_path = args.dir_path + '/dataset/sp/cartesian/' if args.cartesian else args.dir_path + '/dataset/sp/polar/'
    else: args.dataset_path = args.dir_path + '/dataset/' + args.dataset + '/'

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
        mkdir(args.dir_path + '/dataset/sp')
        mkdir(args.dir_path + '/dataset/sp/cartesian')
        mkdir(args.dir_path + '/dataset/sp/polar')
        mkdir(args.dir_path + '/dataset/sm')
        mkdir(args.dir_path + '/dataset/jets')
        if args.dataset == 'sm':
            print("downloading dataset")
            import requests
            r = requests.get('https://pjreddie.com/media/files/mnist_train.csv', allow_redirects=True)
            open(args.dataset_path + 'mnist_train.csv', 'wb').write(r.content)
            r = requests.get('https://pjreddie.com/media/files/mnist_test.csv', allow_redirects=True)
            open(args.dataset_path + 'mnist_test.csv', 'wb').write(r.content)

    prev_models = [f[:-4] for f in listdir(args.args_path)]  # removing txt part

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
    else:
        temp = args.start_epoch, args.num_epochs
        f = open(args.args_path + args.name + ".txt", "r")
        args_dict = vars(args)
        load_args_dict = eval(f.read())
        for key in load_args_dict:
            args_dict[key] = load_args_dict[key]

        args = objectview(args_dict)
        f.close()
        args.load_model = True
        args.start_epoch, args.num_epochs = temp

    args.device = device
    return args


def main(args):
    args = init_dirs(args)

    pt = T.Cartesian() if args.cartesian else T.Polar()

    if args.dataset == 'sp':
        train_dataset = MNISTSuperpixels(args.dataset_path, True, pre_transform=pt)
        test_dataset = MNISTSuperpixels(args.dataset_path, False, pre_transform=pt)
        train_loader = tgDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = tgDataLoader(test_dataset, batch_size=args.batch_size)
    elif args.dataset == 'sm':
        train_dataset = MNISTGraphDataset(args.dataset_path, args.num_hits, train=True)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True)
        test_dataset = MNISTGraphDataset(args.dataset_path, args.num_hits, train=False)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True)

    if(args.load_model):
        C = torch.load(args.model_path + args.name + "/C_" + str(args.start_epoch) + ".pt").to(device)
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

        plt.savefig(args.losses_path + args.name + "/" + str(epoch) + ".png")
        plt.close()

    def save_model(epoch):
        torch.save(C, args.model_path + args.name + "/C_" + str(epoch) + ".pt")

    def train_C(data, y):
        C.train()
        C_optimizer.zero_grad()

        output = C(data)

        # nll_loss takes class labels as target, so one-hot encoding is not needed
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
                if args.dataset == 'sp':
                    output = C(data.to(device))
                    y = data.y
                elif args.dataset == 'sm':
                    output = C(tg_transform(args, data[0]).to(device))
                    y = data[1]

                test_loss += F.nll_loss(output, y, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(y.data.view_as(pred)).sum()

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        print('test')

        f = open(args.out_path + args.name + '.txt', 'a')
        print(args.out_path + args.name + '.txt')
        s = "After {} epochs, on test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(epoch, test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset))
        print(s)
        f.write(s)
        f.close()

    for i in range(args.start_epoch, args.num_epochs):
        print("Epoch %d %s" % ((i + 1), args.name))
        C_loss = 0
        test(i)
        for batch_ndx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            if args.dataset == 'sp':
                C_loss += train_C(data.to(device), data.y.to(device))
            elif args.dataset == 'sm':
                C_loss += train_C(tg_transform(args, data[0]).to(device), data[1].to(device))

        train_losses.append(C_loss / len(train_loader))

        if(args.scheduler):
            C_scheduler.step()

        if((i + 1) % 10 == 0):
            save_model(i + 1)
            plot_losses(i + 1, train_losses, test_losses)

    test(args.num_epochs)


def add_bool_arg(parser, name, help, default=False):
    varname = '_'.join(name.split('-'))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=varname, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=varname, action='store_false', help="don't " + help)
    parser.set_defaults(**{varname: default})


if __name__ == "__main__":
    args = parse_args()
    main(args)
