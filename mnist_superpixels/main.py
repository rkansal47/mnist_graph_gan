# import setGPU

import torch
from model import Graph_GAN, MoNet, GaussianGenerator  # , Graph_Generator, Graph_Discriminator, Gaussian_Discriminator
import utils, save_outputs, eval
from superpixels_dataset import SuperpixelsDataset
from graph_dataset_mnist import MNISTGraphDataset
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

import torch.optim as optim
from tqdm import tqdm

from os import listdir, mkdir
from os.path import exists, dirname, realpath

import sys
import argparse
from copy import deepcopy

from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader as tgDataLoader

import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    dir_path = dirname(realpath(__file__))

    parser = argparse.ArgumentParser()

    # utils.add_bool_arg(parser, "gru", "use GRUs", default=False)
    # parser.add_argument("--fe-hidden-size", type=int, default=128, help="edge network hidden layer size")
    # parser.add_argument("--fe-out-size", type=int, default=256, help="edge network out size")
    #
    # parser.add_argument("--fn-hidden-size", type=int, default=256, help="message passing hidden layers sizes")
    # parser.add_argument("--fn-num-layers", type=int, default=2, help="message passing number of layers in generator")

    # meta

    parser.add_argument("--name", type=str, default="test", help="name or tag for model; will be appended with other info")
    utils.add_bool_arg(parser, "train", "use training or testing dataset for model", default=True, no_name="test")
    parser.add_argument("--num", type=int, default=3, help="number to train on")

    utils.add_bool_arg(parser, "load-model", "load a pretrained model", default=False)
    parser.add_argument("--start-epoch", type=int, default=0, help="which epoch to start training on (only makes sense if loading a model)")
    parser.add_argument("--num-epochs", type=int, default=2000, help="number of epochs to train")

    parser.add_argument("--dir-path", type=str, default=dir_path, help="path where dataset and output will be stored")

    utils.add_bool_arg(parser, "sparse-mnist", "use sparse mnist dataset (as opposed to superpixels)", default=False)

    utils.add_bool_arg(parser, "n", "run on nautilus cluster", default=False)

    utils.add_bool_arg(parser, "save-zero", "save the initial figure", default=False)

    utils.add_bool_arg(parser, "debug", "debug mode", default=False)

    # architecture

    parser.add_argument("--num-hits", type=int, default=75, help="number of hits")
    parser.add_argument("--sd", type=float, default=0.2, help="standard deviation of noise")

    parser.add_argument("--node-feat-size", type=int, default=3, help="node feature size")
    parser.add_argument("--hidden-node-size", type=int, default=32, help="latent vector size of each node (incl node feature size)")

    parser.add_argument("--fn", type=int, nargs='*', default=[256, 256], help="hidden fn layers e.g. 256 256")
    parser.add_argument("--fe", type=int, nargs='+', default=[64, 128], help="hidden and output fe layers e.g. 64 128")
    parser.add_argument("--fnd", type=int, nargs='*', default=[256, 128], help="hidden disc output layers e.g. 256 128")
    parser.add_argument("--mp-iters-gen", type=int, default=2, help="number of message passing iterations in the generator")
    parser.add_argument("--mp-iters-disc", type=int, default=2, help="number of message passing iterations in the discriminator (if applicable)")
    parser.add_argument("--kernel-size", type=int, default=25, help="graph convolutional layer kernel size")
    utils.add_bool_arg(parser, "sum", "mean or sum in models", default=True, no_name="mean")

    utils.add_bool_arg(parser, "int-diffs", "use int diffs", default=False)
    utils.add_bool_arg(parser, "pos-diffs", "use pos diffs", default=True)

    parser.add_argument("--leaky-relu-alpha", type=float, default=0.2, help="leaky relu alpha")

    utils.add_bool_arg(parser, "gcnn", "use gcnn", default=False)
    parser.add_argument("--cutoff", type=float, default=0.32178, help="cutoff edge distance")  # found empirically to match closest to Superpixels

    utils.add_bool_arg(parser, "dea", "use early averaging discriminator", default=False)

    # optimization

    parser.add_argument("--optimizer", type=str, default="None", help="optimizer - options are adam, rmsprop, adadelta")
    parser.add_argument("--loss", type=str, default="ls", help="loss to use - options are og, ls, w, hinge")

    parser.add_argument("--lr-disc", type=float, default=1e-4, help="learning rate discriminator")
    parser.add_argument("--lr-gen", type=float, default=1e-4, help="learning rate generator")
    parser.add_argument("--beta1", type=float, default=0, help="Adam optimizer beta1")
    parser.add_argument("--beta2", type=float, default=0.9, help="Adam optimizer beta2")
    parser.add_argument("--batch-size", type=int, default=10, help="batch size")

    parser.add_argument("--num-critic", type=int, default=1, help="number of critic updates for each generator update")
    parser.add_argument("--num-gen", type=int, default=1, help="number of generator updates for each critic update (num-critic must be 1 for this to apply)")

    # regularization

    utils.add_bool_arg(parser, "batch-norm", "use batch normalization", default=False)
    utils.add_bool_arg(parser, "spectral-norm-disc", "use spectral normalization in discriminator", default=False)
    utils.add_bool_arg(parser, "spectral-norm-gen", "use spectral normalization in generator", default=False)

    parser.add_argument("--disc-dropout", type=float, default=0.5, help="fraction of discriminator dropout")
    parser.add_argument("--gen-dropout", type=float, default=0, help="fraction of generator dropout")

    utils.add_bool_arg(parser, "label-smoothing", "use label smotthing with discriminator", default=False)
    parser.add_argument("--label-noise", type=float, default=0, help="discriminator label noise (between 0 and 1)")

    utils.add_bool_arg(parser, "gp", "use gradient penalty", default=False)
    parser.add_argument("--gp-weight", type=float, default=10, help="WGAN generator penalty weight (if we are using gp)")

    utils.add_bool_arg(parser, "gom", "use gen only mode", default=False)
    utils.add_bool_arg(parser, "bgm", "use boost g mode", default=False)
    utils.add_bool_arg(parser, "rd", "use restart d mode", default=False)
    parser.add_argument("--bag", type=float, default=0.1, help="bag")

    parser.add_argument("--unrolled-steps", type=int, default=0, help="number of unrolled D steps for G training")

    # evaluation

    parser.add_argument("--fid-eval-size", type=int, default=8192, help="number of samples generated for evaluating fid")
    parser.add_argument("--fid-batch-size", type=int, default=128, help="batch size when generating samples for fid eval")

    args = parser.parse_args()

    if(not(args.loss == 'w' or args.loss == 'og' or args.loss == 'ls' or args.loss == 'hinge')):
        print("invalid loss")
        sys.exit()

    if(args.int_diffs and not args.pos_diffs):
        print("int_diffs = true and pos_diffs = false not supported yet")
        sys.exit()

    if(args.n):
        args.dir_path = "/graphganvol/mnist_graph_gan/mnist_superpixels"

    args.channels = [64, 32, 16, 1]

    return args


def init(args):
    torch.manual_seed(4)
    torch.autograd.set_detect_anomaly(True)

    args.device = device

    args.model_path = args.dir_path + '/models/'
    args.losses_path = args.dir_path + '/losses/'
    args.args_path = args.dir_path + '/args/'
    args.figs_path = args.dir_path + '/figs/'
    args.dataset_path = args.dir_path + '/raw/' if not args.sparse_mnist else args.dir_path + '/mnist_dataset/'
    args.err_path = args.dir_path + '/err/'
    args.eval_path = args.dir_path + '/eval/'

    if(not exists(args.model_path)):
        mkdir(args.model_path)
    if(not exists(args.losses_path)):
        mkdir(args.losses_path)
    if(not exists(args.args_path)):
        mkdir(args.args_path)
    if(not exists(args.figs_path)):
        mkdir(args.figs_path)
    if(not exists(args.err_path)):
        mkdir(args.err_path)
    if(not exists(args.dataset_path)):
        mkdir(args.dataset_path)
        print("Downloading dataset")
        if(not args.sparse_mnist):
            import tarfile, urllib
            url = 'http://ls7-www.cs.uni-dortmund.de/cvpr_geometric_dl/mnist_superpixels.tar.gz'
            try:
                # python2
                file_tmp = urllib.urlretrieve(url)[0]
            except:
                # python3
                file_tmp = urllib.request.urlretrieve(url)[0]

            tar = tarfile.open(file_tmp)
            tar.extractall(args.dataset_path)
        else:
            import requests
            r = requests.get('https://pjreddie.com/media/files/mnist_train.csv', allow_redirects=True)
            open(args.dataset_path + 'mnist_train.csv', 'wb').write(r.content)
            r = requests.get('https://pjreddie.com/media/files/mnist_test.csv', allow_redirects=True)
            open(args.dataset_path + 'mnist_test.csv', 'wb').write(r.content)

        print("Downloaded dataset")

    prev_models = [f[:-4] for f in listdir(args.args_path)]  # removing .txt

    if (args.name in prev_models):
        print("name already used")
        # if(not args.load_model):
        #    sys.exit()
    else:
        mkdir(args.losses_path + args.name)
        mkdir(args.model_path + args.name)
        mkdir(args.figs_path + args.name)

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

        args = utils.objectview(args_dict)
        f.close()
        args.load_model = True
        args.start_epoch, args.num_epochs = temp

    return args


def main(args):
    args = init(args)

    def pf(data):
        return data.y == args.num

    pre_filter = pf if args.num != -1 else None

    print("loading data")

    if(args.sparse_mnist):
        X = MNISTGraphDataset(args.dataset_path, args.num_hits, train=args.train, num=args.num)
        X_loaded = DataLoader(X, shuffle=True, batch_size=args.batch_size, pin_memory=True)
    else:
        if(args.gcnn):
            X = MNISTSuperpixels(args.dir_path, train=args.train, pre_transform=T.Cartesian(), pre_filter=pre_filter)
            X_loaded = tgDataLoader(X, shuffle=True, batch_size=args.batch_size)
        else:
            X = SuperpixelsDataset(args.dataset_path, args.num_hits, train=args.train, num=args.num)
            X_loaded = DataLoader(X, shuffle=True, batch_size=args.batch_size, pin_memory=True)

    print("loaded data")

    if(args.load_model):
        G = torch.load(args.model_path + args.name + "/G_" + str(args.start_epoch) + ".pt")
        D = torch.load(args.model_path + args.name + "/D_" + str(args.start_epoch) + ".pt")
    else:
        # G = Graph_Generator(args.node_feat_size, args.fe_hidden_size, args.fe_out_size, args.fn_hidden_size, args.fn_num_layers, args.mp_iters_gen, args.num_hits, args.gen_dropout, args.leaky_relu_alpha, hidden_node_size=args.hidden_node_size, int_diffs=args.int_diffs, pos_diffs=args.pos_diffs, gru=args.gru, batch_norm=args.batch_norm, device=device).to(args.device)
        if(args.gcnn):
            G = GaussianGenerator(args=deepcopy(args)).to(args.device)
            D = MoNet(args=deepcopy(args)).to(args.device)
            # D = Gaussian_Discriminator(args.node_feat_size, args.fe_hidden_size, args.fe_out_size, args.mp_hidden_size, args.mp_num_layers, args.num_iters, args.num_hits, args.dropout, args.leaky_relu_alpha, kernel_size=args.kernel_size, hidden_node_size=args.hidden_node_size, int_diffs=args.int_diffs, gru=GRU, batch_norm=args.batch_norm, device=device).to(args.device)
        else:
            # D = Graph_Discriminator(args.node_feat_size, args.fe_hidden_size, args.fe_out_size, args.fn_hidden_size, args.fn_num_layers, args.mp_iters_disc, args.num_hits, args.disc_dropout, args.leaky_relu_alpha, hidden_node_size=args.hidden_node_size, wgan=args.wgan, int_diffs=args.int_diffs, pos_diffs=args.pos_diffs, gru=args.gru, batch_norm=args.batch_norm, device=device).to(args.device)
            print("Generator")
            G = Graph_GAN(gen=True, args=deepcopy(args)).to(args.device)
            print("Discriminator")
            D = Graph_GAN(gen=False, args=deepcopy(args)).to(args.device)

    print("Models loaded")

    D_params_filter = filter(lambda p: p.requires_grad, D.parameters())  # spectral norm has untrainable params so this excludes those

    if(args.optimizer == 'rmsprop'):
        G_optimizer = optim.RMSprop(G.parameters(), lr=args.lr_gen)
        D_optimizer = optim.RMSprop(D_params_filter, lr=args.lr_disc)
    elif(args.optimizer == 'adadelta'):
        G_optimizer = optim.Adadelta(G.parameters(), lr=args.lr_gen)
        D_optimizer = optim.Adadelta(D_params_filter, lr=args.lr_disc)
    else:
        G_optimizer = optim.Adam(G.parameters(), lr=args.lr_gen, weight_decay=5e-4, betas=(args.beta1, args.beta2))
        D_optimizer = optim.Adam(D_params_filter, lr=args.lr_disc, weight_decay=5e-4, betas=(args.beta1, args.beta2))

    print("optimizers loaded")

    C, mu2, sigma2 = eval.load(args)

    normal_dist = Normal(torch.tensor(0.).to(args.device), torch.tensor(args.sd).to(args.device))

    def train_D(data, gen_data=None, unrolled=False):
        if args.debug: print("dtrain")
        D.train()
        D_optimizer.zero_grad()

        run_batch_size = data.shape[0] if not args.gcnn else data.y.shape[0]

        if gen_data is None:
            gen_data = utils.gen(args, G, normal_dist, run_batch_size)
            if(args.gcnn): gen_data = utils.convert_to_batch(args, gen_data, run_batch_size)

        D_real_output = D(data.clone())
        D_fake_output = D(gen_data)

        D_loss, D_loss_items = utils.calc_D_loss(args, D, data, gen_data, D_real_output, D_fake_output, run_batch_size)
        D_loss.backward(create_graph=unrolled)

        D_optimizer.step()
        return D_loss_items

    def train_G(data):
        if args.debug: print("gtrain")
        G.train()
        G_optimizer.zero_grad()

        gen_data = utils.gen(args, G, normal_dist, args.batch_size)
        if(args.gcnn): gen_data = utils.convert_to_batch(args, gen_data, args.batch_size)

        if(args.unrolled_steps > 0):
            D_backup = deepcopy(D)
            for i in range(args.unrolled_steps - 1):
                train_D(data, gen_data=gen_data, unrolled=True)

        D_fake_output = D(gen_data)

        G_loss = utils.calc_G_loss(args, D_fake_output)

        G_loss.backward()
        G_optimizer.step()

        if(args.unrolled_steps > 0):
            D.load(D_backup)

        return G_loss.item()

    losses = {}

    if(args.load_model):
        try:
            losses['D'] = np.loadtxt(args.losses_path + args.name + "/" + "D.txt").tolist()[:args.start_epoch]
            losses['Dr'] = np.loadtxt(args.losses_path + args.name + "/" + "Dr.txt").tolist()[:args.start_epoch]
            losses['Df'] = np.loadtxt(args.losses_path + args.name + "/" + "Df.txt").tolist()[:args.start_epoch]
            losses['G'] = np.loadtxt(args.losses_path + args.name + "/" + "G.txt").tolist()[:args.start_epoch]
            losses['fid'] = np.loadtxt(args.losses_path + args.name + "/" + "fid.txt").tolist()[:args.start_epoch]

            if(args.gp): losses['gp'] = np.loadtxt(args.losses_path + args.name + "/" + "gp.txt").tolist()[:args.start_epoch]
        except:
            losses['D'] = []
            losses['Dr'] = []
            losses['Df'] = []
            losses['G'] = []
            losses['fid'] = []

            if(args.gp): losses['gp'] = []

    else:
        losses['D'] = []
        losses['Dr'] = []
        losses['Df'] = []
        losses['G'] = []
        losses['fid'] = []

        if(args.gp): losses['gp'] = []

    if(args.save_zero):
        save_outputs.save_sample_outputs(args, D, G, normal_dist, args.name, 0, losses)

    def train():
        k = 0
        temp_ng = args.num_gen
        for i in range(args.start_epoch, args.num_epochs):
            print("Epoch %d %s" % ((i + 1), args.name))
            Dr_loss = 0
            Df_loss = 0
            G_loss = 0
            D_loss = 0
            gp_loss = 0
            losses['fid'].append(eval.get_fid(args, C, G, normal_dist, mu2, sigma2))
            lenX = len(X_loaded)
            for batch_ndx, data in tqdm(enumerate(X_loaded), total=lenX):
                data = data.to(args.device)
                if(args.gcnn):
                    data.pos = (data.pos - 14) / 28
                    row, col = data.edge_index
                    data.edge_attr = (data.pos[col] - data.pos[row]) / (2 * args.cutoff) + 0.5

                if(args.num_critic > 1):
                    D_loss_items = train_D(data)
                    D_loss += D_loss_items['D']
                    Dr_loss += D_loss_items['Dr']
                    Df_loss += D_loss_items['Df']
                    if(args.gp): gp_loss += D_loss_items['gp']

                    if((batch_ndx - 1) % args.num_critic == 0):
                        G_loss += train_G(data)
                else:
                    if(batch_ndx == 0 or (batch_ndx - 1) % args.num_gen == 0):
                        D_loss_items = train_D(data)
                        D_loss += D_loss_items['D']
                        Dr_loss += D_loss_items['Dr']
                        Df_loss += D_loss_items['Df']
                        if(args.gp): gp_loss += D_loss_items['gp']

                    G_loss += train_G(data)

                # if(batch_ndx == 10):
                #     return

            losses['D'].append(D_loss / (lenX / args.num_gen))
            losses['Dr'].append(Dr_loss / (lenX / args.num_gen))
            losses['Df'].append(Df_loss / (lenX / args.num_gen))
            losses['G'].append(G_loss / (lenX / args.num_critic))
            if(args.gp): losses['gp'].append(gp_loss / (lenX / args.num_gen))

            print("d loss: " + str(losses['D'][-1]))
            print("g loss: " + str(losses['G'][-1]))
            print("dr loss: " + str(losses['Dr'][-1]))
            print("df loss: " + str(losses['Df'][-1]))

            if(args.gp): print("gp loss: " + str(losses['gp'][-1]))

            gloss = losses['G'][-1]
            drloss = losses['Dr'][-1]
            dfloss = losses['Df'][-1]
            dloss = (drloss + dfloss) / 2

            if(args.bgm):
                if(i > 20 and gloss > dloss + args.bag):
                    print("num gen upping to 10")
                    args.num_gen = 10
                else:
                    print("num gen normal")
                    args.num_gen = temp_ng
            elif(args.gom):
                if(i > 20 and gloss > dloss + args.bag):
                    print("G loss too high - training G only")
                    j = 0
                    print("starting g loss: " + str(gloss))
                    print("starting d loss: " + str(dloss))

                    while(gloss > dloss + args.bag * 0.5):
                        print(j)
                        gloss = 0
                        for l in tqdm(range(lenX)):
                            gloss += train_G()

                        gloss /= lenX
                        print("g loss: " + str(gloss))
                        print("d loss: " + str(dloss))

                        losses['D'].append(dloss * 2)
                        losses['Dr'].append(drloss)
                        losses['Df'].append(dfloss)
                        losses['G'].append(gloss)

                        if(j % 5 == 0):
                            save_outputs.save_sample_outputs(args, D, G, normal_dist, args.name, i + 1, losses, k=k, j=j)

                        j += 1

                    k += 1
            elif(args.rd):
                if(i > 20 and gloss > dloss + args.bag):
                    print("gloss too high, resetting D params")
                    D.reset_params()

            if((i + 1) % 5 == 0):
                save_outputs.save_sample_outputs(args, D, G, normal_dist, args.name, i + 1, losses)

            if((i + 1) % 5 == 0):
                save_outputs.save_models(args, D, G, args.name, i + 1)

            if((i + 1) % 1 == 0):
                losses['fid'].append(eval.get_fid(args, C, G, normal_dist, mu2, sigma2))

    train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
