# import setGPU

import torch
from model import Graph_GAN, MoNet, GaussianGenerator  # , Graph_Generator, Graph_Discriminator, Gaussian_Discriminator
import utils, save_outputs, evaluation, augment
from superpixels_dataset import SuperpixelsDataset
from graph_dataset_mnist import MNISTGraphDataset
from acgd import ACGD
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
    parser.add_argument("--num", type=int, nargs='+', default=[3], help="number to train on")

    utils.add_bool_arg(parser, "load-model", "load a pretrained model", default=True)
    utils.add_bool_arg(parser, "override-args", "override original model args when loading with new args", default=False)
    parser.add_argument("--start-epoch", type=int, default=-1, help="which epoch to start training on, only applies if loading a model, by default start at the highest epoch model")
    parser.add_argument("--num-epochs", type=int, default=2000, help="number of epochs to train")

    parser.add_argument("--dir-path", type=str, default=dir_path, help="path where dataset and output will be stored")

    parser.add_argument("--num_samples", type=int, default=100, help="num samples to save every 5 epochs")

    utils.add_bool_arg(parser, "sparse-mnist", "use sparse mnist dataset (as opposed to superpixels)", default=False)

    utils.add_bool_arg(parser, "n", "run on nautilus cluster", default=False)
    utils.add_bool_arg(parser, "bottleneck", "use torch.utils.bottleneck settings", default=False)
    utils.add_bool_arg(parser, "lx", "run on lxplus", default=False)

    utils.add_bool_arg(parser, "save-zero", "save the initial figure", default=False)

    utils.add_bool_arg(parser, "debug", "debug mode", default=False)

    # architecture

    parser.add_argument("--num-hits", type=int, default=75, help="number of hits")
    parser.add_argument("--sd", type=float, default=0.2, help="standard deviation of noise")

    parser.add_argument("--node-feat-size", type=int, default=3, help="node feature size")
    parser.add_argument("--hidden-node-size", type=int, default=32, help="hidden vector size of each node (incl node feature size)")
    parser.add_argument("--latent-node-size", type=int, default=0, help="latent vector size of each node - 0 means same as hidden node size")

    parser.add_argument("--fn", type=int, nargs='*', default=[256, 256], help="hidden fn layers e.g. 256 256")
    parser.add_argument("--fe1g", type=int, nargs='*', default=0, help="hidden and output gen fe layers e.g. 64 128 in the first iteration - 0 means same as fe")
    parser.add_argument("--fe1d", type=int, nargs='*', default=0, help="hidden and output disc fe layers e.g. 64 128 in the first iteration - 0 means same as fe")
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
    utils.add_bool_arg(parser, "fcg", "use a fully connected graph", default=True)

    parser.add_argument("--glorot", type=float, default=0, help="gain of glorot - if zero then glorot not used")

    # optimization

    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer - options are adam, rmsprop, adadelta or acgd")
    parser.add_argument("--loss", type=str, default="ls", help="loss to use - options are og, ls, w, hinge")

    parser.add_argument("--lr-disc", type=float, default=1e-4, help="learning rate discriminator")
    parser.add_argument("--lr-gen", type=float, default=1e-4, help="learning rate generator")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam optimizer beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam optimizer beta2")
    parser.add_argument("--batch-size", type=int, default=10, help="batch size")

    parser.add_argument("--num-critic", type=int, default=1, help="number of critic updates for each generator update")
    parser.add_argument("--num-gen", type=int, default=1, help="number of generator updates for each critic update (num-critic must be 1 for this to apply)")

    # regularization

    utils.add_bool_arg(parser, "batch-norm-disc", "use batch normalization", default=False)
    utils.add_bool_arg(parser, "batch-norm-gen", "use batch normalization", default=False)
    utils.add_bool_arg(parser, "spectral-norm-disc", "use spectral normalization in discriminator", default=False)
    utils.add_bool_arg(parser, "spectral-norm-gen", "use spectral normalization in generator", default=False)

    parser.add_argument("--disc-dropout", type=float, default=0.5, help="fraction of discriminator dropout")
    parser.add_argument("--gen-dropout", type=float, default=0, help="fraction of generator dropout")

    utils.add_bool_arg(parser, "label-smoothing", "use label smoothing with discriminator", default=False)
    parser.add_argument("--label-noise", type=float, default=0, help="discriminator label noise (between 0 and 1)")

    # utils.add_bool_arg(parser, "gp", "use gradient penalty", default=False)
    parser.add_argument("--gp", type=float, default=0, help="WGAN generator penalty weight - 0 means not used")

    utils.add_bool_arg(parser, "gom", "use gen only mode", default=False)
    utils.add_bool_arg(parser, "bgm", "use boost g mode", default=False)
    utils.add_bool_arg(parser, "rd", "use restart d mode", default=False)
    parser.add_argument("--bag", type=float, default=0.1, help="bag")

    parser.add_argument("--unrolled-steps", type=int, default=0, help="number of unrolled D steps for G training")

    # augmentation

    # remember to add any new args to the if statement below
    utils.add_bool_arg(parser, "aug-t", "augment with translations", default=False)
    utils.add_bool_arg(parser, "aug-f", "augment with flips", default=False)
    utils.add_bool_arg(parser, "aug-r90", "augment with 90 deg rotations", default=False)
    utils.add_bool_arg(parser, "aug-s", "augment with scalings", default=False)
    parser.add_argument("--translate-ratio", type=float, default=0.125, help="random translate ratio")
    parser.add_argument("--scale-sd", type=float, default=0.125, help="random scale lognormal standard deviation")
    parser.add_argument("--translate-pn-ratio", type=float, default=0.05, help="random translate per node ratio")

    utils.add_bool_arg(parser, "adaptive-prob", "adaptive augment probability", default=False)
    parser.add_argument("--aug-prob", type=float, default=1.0, help="probability of being augmented")

    # evaluation

    utils.add_bool_arg(parser, "fid", "calc fid", default=True)
    parser.add_argument("--fid-eval-size", type=int, default=8192, help="number of samples generated for evaluating fid")
    parser.add_argument("--fid-batch-size", type=int, default=32, help="batch size when generating samples for fid eval")
    parser.add_argument("--gpu-batch", type=int, default=50, help="")

    args = parser.parse_args()

    if isinstance(args.num, list) and len(args.num) == 1:
        args.num = args.num[0]
    elif args.gcnn:
        print("multiple numbers and gcnn not support yet - exiting")
        sys.exit()
    elif isinstance(args.num, list):
        args.num = list(set(args.num))  # remove duplicates
        args.num.sort()
        print(args.num)

    if(args.aug_t or args.aug_f or args.aug_r90 or args.aug_s):
        args.augment = True
    else:
        args.augment = False

    if(not(args.loss == 'w' or args.loss == 'og' or args.loss == 'ls' or args.loss == 'hinge')):
        print("invalid loss - exiting")
        sys.exit()

    if(args.int_diffs and not args.pos_diffs):
        print("int_diffs = true and pos_diffs = false not supported yet - exiting")
        sys.exit()

    if(args.augment and args.gcnn):
        print("augmentation not implemented with GCNN yet - exiting")
        sys.exit()

    if(args.optimizer == 'acgd' and (args.num_critic != 1 or args.num_gen != 1)):
        print("acgd can't have num critic or num gen > 1 - exiting")
        sys.exit()

    if(args.n and args.lx):
        print("can't be on nautilus and lxplus both - exiting")
        sys.exit()

    if(args.num_samples != 100):
        print("save outputs not coded for anything other than 100 samples yet - exiting")
        sys.exit()

    if(args.n):
        args.dir_path = "/graphganvol/mnist_graph_gan/mnist_superpixels"
        args.save_zero = True

    if(args.bottleneck):
        args.save_zero = False

    if(args.lx):
        args.dir_path = "/eos/user/r/rkansal/mnist_graph_gan/mnist_superpixels"
        args.save_zero = True

    if(args.sparse_mnist and args.fid):
        print("no FID for sparse mnist yet")
        args.fid = False

    if(args.latent_node_size and args.latent_node_size < 2):
        print("latent node size can't be less than 2 - exiting")
        sys.exit()

    args.channels = [64, 32, 16, 1]

    return args


def init(args):
    torch.manual_seed(4)
    torch.autograd.set_detect_anomaly(True)

    args.model_path = args.dir_path + '/models/'
    args.losses_path = args.dir_path + '/losses/'
    args.args_path = args.dir_path + '/args/'
    args.figs_path = args.dir_path + '/figs/'
    args.dataset_path = args.dir_path + '/raw/' if not args.sparse_mnist else args.dir_path + '/mnist_dataset/'
    args.err_path = args.dir_path + '/err/'
    args.eval_path = args.dir_path + '/evaluation/'
    args.noise_path = args.dir_path + '/noise/'

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
    if(not exists(args.noise_path)):
        mkdir(args.noise_path)
    if(not exists(args.dataset_path)):
        mkdir(args.dataset_path)
        print("Downloading dataset")
        if(not args.sparse_mnist):
            import tarfile, urllib
            # url = 'http://ls7-www.cs.uni-dortmund.de/cvpr_geometric_dl/mnist_superpixels.tar.gz'
            url = 'https://ls7-www.cs.tu-dortmund.de/fileadmin/ls7-www/misc/cvpr/mnist_superpixels.tar.gz'
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

    if args.load_model:
        if args.start_epoch == -1:
            prev_models = [int(f[:-3].split('_')[-1]) for f in listdir(args.model_path + args.name + '/')]
            if len(prev_models):
                args.start_epoch = max(prev_models)
            else:
                print("No model to load from")
                args.start_epoch = 0
                args.load_model = False
    else:
        args.start_epoch = 0

    if(not args.load_model):
        f = open(args.args_path + args.name + ".txt", "w+")
        f.write(str(vars(args)))
        f.close()
    elif(not args.override_args):
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

    args.device = device

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

    # model

    if(args.load_model):
        G = torch.load(args.model_path + args.name + "/G_" + str(args.start_epoch) + ".pt", map_location=args.device)
        D = torch.load(args.model_path + args.name + "/D_" + str(args.start_epoch) + ".pt", map_location=args.device)

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

    # optimizer

    if args.spectral_norm_gen: G_params = filter(lambda p: p.requires_grad, G.parameters())
    else: G_params = G.parameters()

    if args.spectral_norm_gen: D_params = filter(lambda p: p.requires_grad, D.parameters())
    else: D_params = D.parameters()

    if(args.optimizer == 'rmsprop'):
        G_optimizer = optim.RMSprop(G_params, lr=args.lr_gen)
        D_optimizer = optim.RMSprop(D_params, lr=args.lr_disc)
    elif(args.optimizer == 'adadelta'):
        G_optimizer = optim.Adadelta(G_params, lr=args.lr_gen)
        D_optimizer = optim.Adadelta(D_params, lr=args.lr_disc)
    elif(args.optimizer == 'acgd'):
        optimizer = ACGD(max_params=G_params, min_params=D_params, lr_max=args.lr_gen, lr_min=args.lr_disc, device=args.device)
    elif(args.optimizer == 'adam' or args.optimizer == 'None'):
        G_optimizer = optim.Adam(G_params, lr=args.lr_gen, weight_decay=5e-4, betas=(args.beta1, args.beta2))
        D_optimizer = optim.Adam(D_params, lr=args.lr_disc, weight_decay=5e-4, betas=(args.beta1, args.beta2))

    if(args.load_model):
        try:
            if(not args.optimizer == 'acgd'):
                G_optimizer.load_state_dict(torch.load(args.model_path + args.name + "/G_optim_" + str(args.start_epoch) + ".pt", map_location=args.device))
                D_optimizer.load_state_dict(torch.load(args.model_path + args.name + "/D_optim_" + str(args.start_epoch) + ".pt", map_location=args.device))
            else:
                optimizer.load_state_dict(torch.load(args.model_path + args.name + "/optim_" + str(args.start_epoch) + ".pt", map_location=args.device))
        except:
            print("Error loading optimizer")

    print("optimizers loaded")

    if args.fid: C, mu2, sigma2 = evaluation.load(args, X_loaded)

    normal_dist = Normal(torch.tensor(0.).to(args.device), torch.tensor(args.sd).to(args.device))

    lns = args.latent_node_size if args.latent_node_size else args.hidden_node_size

    args.noise_file_name = "num_samples_" + str(args.num_samples) + "_num_nodes_" + str(args.num_hits) + "_latent_node_size_" + str(lns) + "_sd_" + str(args.sd) + ".pt"
    if args.gcnn: args.noise_file_name = "gcnn_" + args.noise_file_name

    noise_file_names = listdir(args.noise_path)

    if args.noise_file_name not in noise_file_names:
        if(args.gcnn):
            torch.save(normal_dist.sample((args.num_samples * 5, 2 + args.channels[0])), args.noise_path + args.noise_file_name)
        else:
            torch.save(normal_dist.sample((args.num_samples, args.num_hits, lns)), args.noise_path + args.noise_file_name)

    losses = {}

    if(args.load_model):
        try:
            losses['D'] = np.loadtxt(args.losses_path + args.name + "/" + "D.txt").tolist()[:args.start_epoch]
            losses['Dr'] = np.loadtxt(args.losses_path + args.name + "/" + "Dr.txt").tolist()[:args.start_epoch]
            losses['Df'] = np.loadtxt(args.losses_path + args.name + "/" + "Df.txt").tolist()[:args.start_epoch]
            losses['G'] = np.loadtxt(args.losses_path + args.name + "/" + "G.txt").tolist()[:args.start_epoch]
            if args.fid: losses['fid'] = np.loadtxt(args.losses_path + args.name + "/" + "fid.txt").tolist()[:args.start_epoch]
            if(args.gp): losses['gp'] = np.loadtxt(args.losses_path + args.name + "/" + "gp.txt").tolist()[:args.start_epoch]
        except:
            print("couldn't load losses")
            losses['D'] = []
            losses['Dr'] = []
            losses['Df'] = []
            losses['G'] = []
            if args.fid: losses['fid'] = []
            if(args.gp): losses['gp'] = []

    else:
        losses['D'] = []
        losses['Dr'] = []
        losses['Df'] = []
        losses['G'] = []
        if args.fid: losses['fid'] = []
        if(args.gp): losses['gp'] = []

    Y_real = torch.ones(args.batch_size, 1).to(args.device)
    Y_fake = torch.zeros(args.batch_size, 1).to(args.device)

    def train_D(data, gen_data=None, unrolled=False):
        if args.debug: print("dtrain")
        D.train()
        D_optimizer.zero_grad()

        run_batch_size = data.shape[0] if not args.gcnn else data.y.shape[0]

        if gen_data is None:
            gen_data = utils.gen(args, G, normal_dist, run_batch_size)
            if(args.gcnn): gen_data = utils.convert_to_batch(args, gen_data, run_batch_size)

        if args.augment:
            p = args.aug_prob if not args.adaptive_prob else losses['p'][-1]
            data = augment.augment(args, data, p)
            gen_data = augment.augment(args, gen_data, p)

        D_real_output = D(data.clone())
        D_fake_output = D(gen_data)

        D_loss, D_loss_items = utils.calc_D_loss(args, D, data, gen_data, D_real_output, D_fake_output, run_batch_size, Y_real, Y_fake)
        D_loss.backward(create_graph=unrolled)

        D_optimizer.step()
        return D_loss_items

    def train_G(data):
        if args.debug: print("gtrain")
        G.train()
        G_optimizer.zero_grad()

        gen_data = utils.gen(args, G, normal_dist, args.batch_size)
        if(args.gcnn): gen_data = utils.convert_to_batch(args, gen_data, args.batch_size)

        if args.augment:
            p = args.aug_prob if not args.adaptive_prob else losses['p'][-1]
            gen_data = augment.augment(args, gen_data, p)

        if(args.unrolled_steps > 0):
            D_backup = deepcopy(D)
            for i in range(args.unrolled_steps - 1):
                train_D(data, gen_data=gen_data, unrolled=True)

        D_fake_output = D(gen_data)

        G_loss = utils.calc_G_loss(args, D_fake_output, Y_real)

        G_loss.backward()
        G_optimizer.step()

        if(args.unrolled_steps > 0):
            D.load(D_backup)

        return G_loss.item()

    def train_acgd(data):
        if args.debug: print("acgd train")
        D.train()
        G.train()
        optimizer.zero_grad()

        run_batch_size = data.shape[0] if not args.gcnn else data.y.shape[0]

        gen_data = utils.gen(args, G, normal_dist, run_batch_size)
        if(args.gcnn): gen_data = utils.convert_to_batch(args, gen_data, run_batch_size)

        if args.augment:
            p = args.aug_prob if not args.adaptive_prob else losses['p'][-1]
            data = utils.rand_translate(args, data, p)
            gen_data = utils.rand_translate(args, gen_data, p)

        D_real_output = D(data.clone())
        D_fake_output = D(gen_data)

        D_loss, D_loss_items = utils.calc_D_loss(args, D, data, gen_data, D_real_output, D_fake_output, run_batch_size)

        optimizer.step(loss=D_loss)

        G.eval()
        with torch.no_grad():
            G_loss = utils.calc_G_loss(args, D_fake_output)

        return D_loss_items, G_loss.item()

    def train():
        k = 0
        temp_ng = args.num_gen
        if(args.fid): losses['fid'].append(evaluation.get_fid(args, C, G, normal_dist, mu2, sigma2))
        if(args.save_zero): save_outputs.save_sample_outputs(args, D, G, normal_dist, args.name, 0, losses)
        for i in range(args.start_epoch, args.num_epochs):
            print("Epoch %d %s" % ((i + 1), args.name))
            Dr_loss = 0
            Df_loss = 0
            G_loss = 0
            D_loss = 0
            gp_loss = 0
            lenX = len(X_loaded)
            for batch_ndx, data in tqdm(enumerate(X_loaded), total=lenX):
                data = data.to(args.device)
                if(args.gcnn):
                    data.pos = (data.pos - 14) / 28
                    row, col = data.edge_index
                    data.edge_attr = (data.pos[col] - data.pos[row]) / (2 * args.cutoff) + 0.5

                if(not args.optimizer == 'acgd'):
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
                else:
                    D_loss_items, G_loss_item = train_acgd(data)
                    D_loss += D_loss_items['D']
                    Dr_loss += D_loss_items['Dr']
                    Df_loss += D_loss_items['Df']
                    G_loss += G_loss_item

                if args.bottleneck:
                    if(batch_ndx == 10):
                        return

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
                optimizers = optimizer if args.optimizer == 'acgd' else (D_optimizer, G_optimizer)
                save_outputs.save_models(args, D, G, optimizers, args.name, i + 1)

            if(args.fid and (i + 1) % 1 == 0):
                losses['fid'].append(evaluation.get_fid(args, C, G, normal_dist, mu2, sigma2))

            if((i + 1) % 5 == 0):
                save_outputs.save_sample_outputs(args, D, G, normal_dist, args.name, i + 1, losses)

    train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
