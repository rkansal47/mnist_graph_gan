# import setGPU

# from profile import profile
# from time import sleep

import torch
from model import Graph_Generator, Graph_Discriminator, Gaussian_Discriminator, MoNet
from superpixels_dataset import SuperpixelsDataset
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from skimage.draw import draw

import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm

import numpy as np

from os import listdir, mkdir
from os.path import exists, dirname, realpath

import sys
import tarfile
import urllib

from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader as tgDataLoader
from torch_geometric.data import Batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#torch.cuda.set_device(0)

url = 'http://ls7-www.cs.uni-dortmund.de/cvpr_geometric_dl/mnist_superpixels.tar.gz'

#Have to specify 'name' and 'args.start_epoch' if True
GCNN = True
WGAN = False
LSGAN = True #WGAN must be false otherwise it'll just be WGAN
TRAIN = True
NUM = 3 #-1 means all numbers
INT_DIFFS = True
GRU = False

def main(args):
    torch.manual_seed(4)
    torch.autograd.set_detect_anomaly(True)

    name = [args.name]
    if WGAN:
        name.append('wgan')
    if GRU:
        name.append('gru')
    if GCNN:
        name.append('gcnn')

    # name.append('num_iters_{}'.format(args.num_iters))
    # name.append('num_critic_{}'.format(args.num_critic))
    args.name = '_'.join(name)

    args.model_path = args.dir_path + '/models/'
    args.losses_path = args.dir_path + '/losses/'
    args.args_path = args.dir_path + '/args/'
    args.figs_path = args.dir_path + '/figs/'
    args.dataset_path = args.dir_path + '/dataset/'

    if(not exists(args.model_path)):
        mkdir(args.model_path)
    if(not exists(args.losses_path)):
        mkdir(args.losses_path)
    if(not exists(args.args_path)):
        mkdir(args.args_path)
    if(not exists(args.figs_path)):
        mkdir(args.figs_path)
    if(not exists(args.dataset_path)):
        mkdir(args.dataset_path)
        try:
            # python2
            file_tmp = urllib.urlretrieve(url, filename=None)[0]
        except:
            # python3
            file_tmp = urllib.request.urlretrieve(url, filename=args.dataset)[0]

    prev_models = [f[:-4] for f in listdir(args.args_path)] #removing .txt

    if (args.name in prev_models):
        print("name already used")
        #if(not args.load_model):
        #    sys.exit()
    else:
        mkdir(args.losses_path + args.name)
        mkdir(args.model_path + args.name)
        mkdir(args.figs_path + args.name)

    if(not args.load_model):
        f = open(args.args_path + args.name + ".txt", "w+")
        f.write(str(vars(args)))
        f.close()
    #else:
        #f = open(args.args_path + args.name + ".txt", "r")
        #temp = args.start_epoch
        #args = eval(f.read())
        #f.close()
        #args.load_model = True
        #args.start_epoch = temp
        # return args2

    def pf(data):
        return data.y == args.num

    pre_filter = pf if args.num != -1 else None

    print("loading")

    #Change to True !!
    X = SuperpixelsDataset(args.dataset_path, args.num_hits, train=TRAIN, num=NUM, device=device)
    tgX = MNISTSuperpixels(args.dir_path, train=TRAIN, pre_transform=T.Cartesian(), pre_filter=pre_filter)


    X_loaded = DataLoader(X, shuffle=True, batch_size=args.batch_size, pin_memory=True)
    tgX_loaded = tgDataLoader(tgX, shuffle=True, batch_size=args.batch_size)

    print("loaded")

    if(args.load_model):
        G = torch.load(args.model_path + args.name + "/G_" + str(args.start_epoch) + ".pt")
        D = torch.load(args.model_path + args.name + "/D_" + str(args.start_epoch) + ".pt")
    else:
        G = Graph_Generator(args.node_feat_size, args.fe_hidden_size, args.fe_out_size, args.gru_hidden_size, args.gru_num_layers, args.num_iters, args.num_hits, args.dropout, args.leaky_relu_alpha, hidden_node_size=args.hidden_node_size, int_diffs=INT_DIFFS, gru=GRU, device=device).to(device)
        if(GCNN):
            D = MoNet(kernel_size=args.kernel_size, dropout=args.dropout, device=device).to(device)
            # D = Gaussian_Discriminator(args.node_feat_size, args.fe_hidden_size, args.fe_out_size, args.gru_hidden_size, args.gru_num_layers, args.num_iters, args.num_hits, args.dropout, args.leaky_relu_alpha, kernel_size=args.kernel_size, hidden_node_size=args.hidden_node_size, int_diffs=INT_DIFFS, gru=GRU).to(device)
        else:
            D = Graph_Discriminator(args.node_feat_size, args.fe_hidden_size, args.fe_out_size, args.gru_hidden_size, args.gru_num_layers, args.num_iters, args.num_hits, args.dropout, args.leaky_relu_alpha, hidden_node_size=args.hidden_node_size, int_diffs=INT_DIFFS, gru=GRU, device=device).to(device)

    print("Models loaded")

    if(WGAN):
        G_optimizer = optim.RMSprop(G.parameters(), lr = args.lr_gen)
        D_optimizer = optim.RMSprop(D.parameters(), lr = args.lr_disc)
    else:
        G_optimizer = optim.Adam(G.parameters(), lr = args.lr_gen, weight_decay=5e-4)
        D_optimizer = optim.Adam(D.parameters(), lr = args.lr_disc, weight_decay=5e-4)

    print("optimizers loaded")

    normal_dist = Normal(0, 0.2)

    def wasserstein_loss(y_out, y_true):
        return -torch.mean(y_out * y_true)

    if(WGAN):
        criterion = wasserstein_loss
    else:
        if(LSGAN):
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.BCELoss()

    # print(criterion(torch.tensor([1.0]),torch.tensor([-1.0])))

    def gen(num_samples, noise=0):
        if(noise == 0):
            noise = normal_dist.sample((num_samples, args.num_hits, args.hidden_node_size)).to(device)

        x = noise
        del noise

        x = G(x)
        return x

    # transform my format to torch_geometric's
    def tg_transform(X):
        batch_size = X.size(0)
        cutoff = 0.32178 #found empirically to match closest to Superpixels

        pos = X[:,:,:2]

        x1 = pos.repeat(1, 1, 75).reshape(batch_size, 75*75, 2)
        x2 = pos.repeat(1, 75, 1)

        diff_norms = torch.norm(x2 - x1 + 1e-12, dim=2)

        diff = x2-x1
        diff = diff[diff_norms < cutoff]

        norms = diff_norms.reshape(batch_size, 75, 75)
        neighborhood = torch.nonzero(norms < cutoff, as_tuple=False)
        edge_attr = diff[neighborhood[:, 1] != neighborhood[:, 2]]

        neighborhood = neighborhood[neighborhood[:, 1] != neighborhood[:, 2]] #remove self-loops
        unique, counts = torch.unique(neighborhood[:, 0], return_counts=True)
        edge_slices = torch.cat((torch.tensor([0]).to(device), counts.cumsum(0)))
        edge_index = neighborhood[:,1:].transpose(0,1)

        #normalizing edge attributes
        edge_attr_list = list()
        for i in range(batch_size):
            start_index = edge_slices[i]
            end_index = edge_slices[i+1]
            temp = diff[start_index:end_index]
            max = torch.max(temp)
            temp = temp/(2*max + 1e-12) + 0.5
            edge_attr_list.append(temp)

        edge_attr = torch.cat(edge_attr_list)

        x = X[:,:,2].reshape(batch_size*75, 1)+0.5
        pos = 27*pos.reshape(batch_size*75, 2)+13.5

        zeros = torch.zeros(batch_size*75, dtype=int).to(device)
        zeros[torch.arange(batch_size)*75] = 1
        batch = torch.cumsum(zeros, 0)-1

        return Batch(batch=batch, x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr, y=None, pos=pos)

    def draw_graph(graph, node_r, im_px):
        imd = im_px + node_r
        img = np.zeros((imd, imd), dtype=np.float)

        circles = []
        for node in graph:
            circles.append((draw.circle_perimeter(int(node[1]), int(node[0]), node_r), draw.disk((int(node[1]), int(node[0])), node_r), node[2]))

        for circle in circles:
            img[circle[1]] = circle[2]

        return img

    def save_sample_outputs(name, epoch, dlosses, glosses):
        print("drawing figs")
        fig = plt.figure(figsize=(10,10))

        num_ims = 100
        node_r = 30
        im_px = 1000

        gen_out = gen(args.batch_size).cpu().detach().numpy()

        for i in range(int(num_ims/args.batch_size)):
            gen_out = np.concatenate((gen_out, gen(args.batch_size).cpu().detach().numpy()), 0)

        gen_out = gen_out[:num_ims]

        gen_out[gen_out > 0.47] = 0.47
        gen_out[gen_out < -0.5] = -0.5

        gen_out = gen_out*[im_px, im_px, 1] + [(im_px+node_r)/2, (im_px+node_r)/2, 0.55]

        for i in range(1, num_ims+1):
            fig.add_subplot(10, 10, i)
            im_disp = draw_graph(gen_out[i-1], node_r, im_px)
            plt.imshow(im_disp, cmap=cm.gray_r, interpolation='nearest')
            plt.axis('off')

        plt.savefig(args.figs_path + args.name + "/" + str(epoch) + ".png")
        plt.close()

        plt.figure()
        plt.plot(dlosses, label='Discriminitive loss')
        plt.plot(glosses, label='Generative loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(args.losses_path + args.name +"/"+ str(epoch) + ".png")
        plt.close()

        print("saved figs")

    def save_models(name, epoch):
        torch.save(G, args.model_path + args.name + "/G_" + str(epoch) + ".pt")
        torch.save(D, args.model_path + args.name+ "/D_" + str(epoch) + ".pt")

    #from https://github.com/EmilienDupont/wgan-gp
    def gradient_penalty(real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1)
        alpha = alpha.expand_as(real_data).to(device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True).to(device)

        del alpha
        torch.cuda.empty_cache()

        # Calculate probability of interpolated examples
        prob_interpolated = D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones(prob_interpolated.size()).to(device), create_graph=True, retain_graph=True, allow_unused=True)[0].to(device)

        gradients = gradients.contiguous()

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return args.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def train_D(data):
        D.train()
        D_optimizer.zero_grad()

        run_batch_size = data.shape[0] if not GCNN else data.y.shape[0]

        if(not WGAN):
            Y_real = torch.ones(run_batch_size, 1).to(device)
            Y_fake = torch.zeros(run_batch_size, 1).to(device)

        try:
            D_real_output = D(data)
            gen_ims = gen(run_batch_size)

            if(GCNN):
                tg_gen_ims = tg_transform(gen_ims)

            D_fake_output = D(tg_gen_ims)

            if(WGAN):
                D_loss = D_fake_output.mean() - D_real_output.mean() + gradient_penalty(x, tg_gen_ims)
            else:
                D_real_loss = criterion(D_real_output, Y_real)
                D_fake_loss = criterion(D_fake_output, Y_fake)

                D_loss = D_real_loss + D_fake_loss
        except:
            print("Generated Images")
            print(gen_ims)

            print("Transformed Images")
            print(tg_gen_ims)

            print("Discriminator Output")
            print(D_fake_output)

            return


        D_loss.backward()
        D_optimizer.step()

        return D_loss.item()

    def train_G():
        G.train()
        G_optimizer.zero_grad()

        if(not WGAN):
            Y_real = torch.ones(args.batch_size, 1).to(device)

        gen_ims = gen(args.batch_size)

        if(GCNN):
            gen_ims = tg_transform(gen_ims)

        D_fake_output = D(gen_ims)

        if(WGAN):
            G_loss = -D_fake_output.mean()
        else:
            G_loss = criterion(D_fake_output, Y_real)

        G_loss.backward()
        G_optimizer.step()

        return G_loss.item()

    D_losses = []
    G_losses = []

    # save_models(name, 0)

    # save_sample_outputs(args.name, 0, D_losses, G_losses)

    # @profile
    def train():
        for i in range(args.start_epoch, args.num_epochs):
            print("Epoch %d %s" % ((i+1), args.name))
            D_loss = 0
            G_loss = 0
            loader = tgX_loaded if GCNN else X_loaded
            for batch_ndx, data in tqdm(enumerate(loader), total=len(loader)):
                if(batch_ndx > 0 and batch_ndx % (args.num_critic+1) == 0):
                    G_loss += train_G()
                else:
                    D_loss += train_D(data.to(device)) if GCNN else train_D(data[0].to(device))

            D_losses.append(D_loss/len(X_loaded)/2)
            G_losses.append(G_loss/len(X_loaded))

            if((i+1)%5==0):
                save_sample_outputs(args.name, i+1, D_losses, G_losses)

            if((i+1)%5==0):
                save_models(args.name, i+1)

    train()

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

    add_bool_arg(parser, "load-model", "load a pretrained model", default=False)
    parser.add_argument("--start-epoch", type=int, default=0, help="which epoch to start training on (only makes sense if loading a model)")

    parser.add_argument("--dir-path", type=str, default=dir_path, help="path where dataset and output will be stored")
    parser.add_argument("--node-feat-size", type=int, default=3, help="node feature size")
    parser.add_argument("--fe-hidden-size", type=int, default=128, help="edge network hidden layer size")
    parser.add_argument("--fe-out-size", type=int, default=256, help="edge network out size")
    parser.add_argument("--gru-hidden-size", type=int, default=256, help="GRU hidden size")
    parser.add_argument("--gru-num-layers", type=int, default=2, help="GRU number of layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="fraction of dropout")
    parser.add_argument("--leaky-relu-alpha", type=float, default=0.2, help="leaky relu alpha")
    parser.add_argument("--num-hits", type=int, default=75, help="number of hits")
    parser.add_argument("--num-epochs", type=int, default=2000, help="number of epochs to train")
    parser.add_argument("--lr-disc", type=float, default=1e-4, help="learning rate discriminator")
    parser.add_argument("--lr-gen", type=float, default=1e-4, help="learning rate generator")
    parser.add_argument("--num-critic", type=int, default=2, help="number of critic updates for each generator update")
    parser.add_argument("--num-iters", type=int, default=1, help="number of discriminator updates for each generator update")
    parser.add_argument("--hidden-node-size", type=int, default=64, help="latent vector size of each node (incl node feature size)")
    parser.add_argument("--kernel-size", type=int, default=10, help="graph convolutional layer kernel size")
    parser.add_argument("--num", type=int, default=3, help="number to train on")

    parser.add_argument("--batch-size", type=int, default=10, help="batch size")
    parser.add_argument("--gp-weight", type=float, default=10, help="WGAN generator penalty weight")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam optimizer beta1")
    parser.add_argument("--name", type=str, default="test", help="name or tag for model; will be appended with other info")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
