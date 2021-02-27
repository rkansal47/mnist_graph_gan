import argparse
import sys
import utils

from os import listdir, mkdir
from os.path import exists, dirname, realpath

import torch

from model import Graph_GAN
from copy import deepcopy

import torch.optim as optim

import numpy as np

import logging


def parse_args():
    parser = argparse.ArgumentParser()

    # meta

    parser.add_argument("--name", type=str, default="test", help="name or tag for model; will be appended with other info")
    parser.add_argument("--dataset", type=str, default="jets", help="dataset to use", choices=['jets', 'sparse-mnist', 'superpixels'])

    utils.add_bool_arg(parser, "train", "use training or testing dataset for model", default=True, no_name="test")
    parser.add_argument("--ttsplit", type=float, default=0.85, help="ratio of train/test split")

    utils.add_bool_arg(parser, "load-model", "load a pretrained model", default=True)
    utils.add_bool_arg(parser, "override-load-check", "override check for whether name has already been used", default=False)
    utils.add_bool_arg(parser, "override-args", "override original model args when loading with new args", default=False)
    parser.add_argument("--start-epoch", type=int, default=-1, help="which epoch to start training on, only applies if loading a model, by default start at the highest epoch model")
    parser.add_argument("--num-epochs", type=int, default=2000, help="number of epochs to train")

    parser.add_argument("--dir-path", type=str, default="", help="path where dataset and output will be stored")

    parser.add_argument("--num-samples", type=int, default=100000, help="num samples to evaluate every 5 epochs")

    utils.add_bool_arg(parser, "n", "run on nautilus cluster", default=False)
    utils.add_bool_arg(parser, "bottleneck", "use torch.utils.bottleneck settings", default=False)
    utils.add_bool_arg(parser, "lx", "run on lxplus", default=False)

    utils.add_bool_arg(parser, "save-zero", "save the initial figure", default=False)
    utils.add_bool_arg(parser, "no-save-zero-or", "override --n save-zero default", default=False)
    parser.add_argument("--save-epochs", type=int, default=0, help="save outputs per how many epochs")
    parser.add_argument("--save-model-epochs", type=int, default=0, help="save models per how many epochs")

    utils.add_bool_arg(parser, "debug", "debug mode", default=False)
    utils.add_bool_arg(parser, "break-zero", "break after 1 iteration", default=False)
    utils.add_bool_arg(parser, "low-samples", "small number of samples for debugging", default=False)

    parser.add_argument("--jets", type=str, default="g", help="jet type", choices=['g', 't', 'w', 'z', 'q'])

    utils.add_bool_arg(parser, "real-only", "use jets with ony real particles", default=False)

    utils.add_bool_arg(parser, "multi-gpu", "use multiple gpus if possible", default=False)

    parser.add_argument("--log-file", type=str, default="", help='log file name - default is name of file in outs/ ; "stdout" prints to console')
    parser.add_argument("--log", type=str, default="INFO", help="log level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    parser.add_argument("--seed", type=int, default=4, help="torch seed")

    # architecture

    parser.add_argument("--num-hits", type=int, default=30, help="number of hits")
    parser.add_argument("--coords", type=str, default="polarrel", help="cartesian, polarrel or polarrelabspt", choices=['cartesian, polarrel, polarrelabspt'])

    parser.add_argument("--norm", type=float, default=1, help="normalizing max value of features to this value")

    parser.add_argument("--sd", type=float, default=0.2, help="standard deviation of noise")

    parser.add_argument("--node-feat-size", type=int, default=3, help="node feature size")
    parser.add_argument("--hidden-node-size", type=int, default=32, help="hidden vector size of each node (incl node feature size)")
    parser.add_argument("--latent-node-size", type=int, default=0, help="latent vector size of each node - 0 means same as hidden node size")

    parser.add_argument("--clabels", type=int, default=0, help="0 - no clabels, 1 - clabels with pt only, 2 - clabels with pt and eta", choices=[0, 1, 2])
    utils.add_bool_arg(parser, "clabels-fl", "use conditional labels in first layer", default=True)
    utils.add_bool_arg(parser, "clabels-hl", "use conditional labels in hidden layers", default=True)

    parser.add_argument("--fn", type=int, nargs='*', default=[256, 256], help="hidden fn layers e.g. 256 256")
    parser.add_argument("--fe1g", type=int, nargs='*', default=0, help="hidden and output gen fe layers e.g. 64 128 in the first iteration - 0 means same as fe")
    parser.add_argument("--fe1d", type=int, nargs='*', default=0, help="hidden and output disc fe layers e.g. 64 128 in the first iteration - 0 means same as fe")
    parser.add_argument("--fe", type=int, nargs='+', default=[96, 160, 192], help="hidden and output fe layers e.g. 64 128")
    parser.add_argument("--fnd", type=int, nargs='*', default=[128, 64], help="hidden disc output layers e.g. 128 128")
    parser.add_argument("--fmg", type=int, nargs='*', default=[64], help="mask network layers e.g. 64; input 0 for no intermediate layers")
    parser.add_argument("--mp-iters-gen", type=int, default=0, help="number of message passing iterations in the generator")
    parser.add_argument("--mp-iters-disc", type=int, default=0, help="number of message passing iterations in the discriminator (if applicable)")
    parser.add_argument("--mp-iters", type=int, default=2, help="number of message passing iterations in gen and disc both - will be overwritten by gen or disc specific args if given")
    utils.add_bool_arg(parser, "sum", "mean or sum in models", default=True, no_name="mean")

    utils.add_bool_arg(parser, "int-diffs", "use int diffs", default=False)
    utils.add_bool_arg(parser, "pos-diffs", "use pos diffs", default=True)
    # utils.add_bool_arg(parser, "scalar-diffs", "use scalar diff (as opposed to vector)", default=True)
    utils.add_bool_arg(parser, "deltar", "use delta r as an edge feature", default=True)
    utils.add_bool_arg(parser, "deltacoords", "use delta coords as edge features", default=False)

    parser.add_argument("--leaky-relu-alpha", type=float, default=0.2, help="leaky relu alpha")

    utils.add_bool_arg(parser, "dea", "use early averaging discriminator", default=False)

    utils.add_bool_arg(parser, "fully-connected", "use a fully connected graph", default=True)
    parser.add_argument("--num-knn", type=int, default=10, help="# of nearest nodes to connect to (if not fully connected)")
    utils.add_bool_arg(parser, "self-loops", "use self loops in graph - always true for fully connected", default=True)

    parser.add_argument("--glorot", type=float, default=0, help="gain of glorot - if zero then glorot not used")

    utils.add_bool_arg(parser, "gtanh", "use tanh for g output", default=True)
    # utils.add_bool_arg(parser, "dearlysigmoid", "use early sigmoid in d", default=False)

    utils.add_bool_arg(parser, "mask-feat", "add mask as fourth feature", default=False)
    utils.add_bool_arg(parser, "mask-feat-bin", "binary fourth feature", default=False)
    utils.add_bool_arg(parser, "mask-weights", "weight D nodes by mask", default=False)
    utils.add_bool_arg(parser, "mask-manual", "manually mask generated nodes with pT less than cutoff", default=False)
    utils.add_bool_arg(parser, "mask-exp", "exponentially decaying or binary mask; relevant only if mask-manual is true", default=False)
    utils.add_bool_arg(parser, "mask-real-only", "only use masking for real jets", default=False)
    utils.add_bool_arg(parser, "mask-learn", "learn mask from latent vars only use during gen", default=False)
    utils.add_bool_arg(parser, "mask-learn-bin", "binary or continuous learnt mask", default=True)
    utils.add_bool_arg(parser, "mask-learn-sep", "learn mask from separate noise vector", default=False)
    utils.add_bool_arg(parser, "mask-disc-sep", "separate disc network for # particles", default=False)
    utils.add_bool_arg(parser, "mask-fnd-np", "use num masked particles as an additional arg in D (dea will automatically be set true)", default=False)
    utils.add_bool_arg(parser, "mask-c", "conditional mask", default=False)
    utils.add_bool_arg(parser, "mask-fne-np", "pass num particles as features into fn and fe", default=False)
    parser.add_argument("--mask-epoch", type=int, default=0, help="# of epochs after which to start masking")

    utils.add_bool_arg(parser, "noise-padding", "use Gaussian noise instead of zero-padding for fake particles", default=False)

    # optimization

    parser.add_argument("--optimizer", type=str, default="rmsprop", help="pick optimizer", choices=['adam', 'rmsprop', 'adadelta', 'agcd'])
    parser.add_argument("--loss", type=str, default="ls", help="loss to use - options are og, ls, w, hinge", choices=['og', 'ls', 'w', 'hinge'])

    parser.add_argument("--lr-disc", type=float, default=3e-5, help="learning rate discriminator")
    parser.add_argument("--lr-gen", type=float, default=1e-5, help="learning rate generator")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam optimizer beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam optimizer beta2")
    parser.add_argument("--batch-size", type=int, default=0, help="batch size")

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

    parser.add_argument("--gp", type=float, default=0, help="WGAN generator penalty weight - 0 means not used")

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

    utils.add_bool_arg(parser, "fid", "calc fid", default=False)
    parser.add_argument("--fid-eval-size", type=int, default=8192, help="number of samples generated for evaluating fid")
    parser.add_argument("--fid-batch-size", type=int, default=32, help="batch size when generating samples for fid eval")
    parser.add_argument("--gpu-batch", type=int, default=50, help="")

    utils.add_bool_arg(parser, "w1", "calc w1", default=True)
    parser.add_argument("--w1-num-samples", type=int, nargs='+', default=[100, 1000, 10000], help='array of # of jet samples to test')
    parser.add_argument("--w1-tot-samples", type=int, default=100000, help='tot # of jets to generate to sample from')

    parser.add_argument("--jf", type=str, nargs='*', default=['mass', 'pt'], help='jet level features to evaluate')

    args = parser.parse_args()

    return args


def check_args(args):
    if args.real_only and (not args.jets == 't' or not args.num_hits == 30):
        logging.error("real only arg works only with 30p jets - exiting")
        sys.exit()

    if(args.aug_t or args.aug_f or args.aug_r90 or args.aug_s):
        args.augment = True
    else:
        args.augment = False

    # if not args.coords == 'polarrelabspt':
    #     logging.info("Can't have jet level features for this coordinate system")
    #     args.jf = False
    # elif len(args.jet_features):
    #     args.jf = True

    if(args.int_diffs):
        logging.error("int_diffs not supported yet - exiting")
        sys.exit()

    if(args.dataset == 'jets' and args.augment):
        logging.error("augmentation not implemented for jets yet - exiting")
        sys.exit()

    if(args.optimizer == 'acgd' and (args.num_critic != 1 or args.num_gen != 1)):
        logging.error("acgd can't have num critic or num gen > 1 - exiting")
        sys.exit()

    if(args.n and args.lx):
        logging.error("can't be on nautilus and lxplus both - exiting")
        sys.exit()

    if(args.latent_node_size and args.latent_node_size < 3):
        logging.error("latent node size can't be less than 2 - exiting")
        sys.exit()

    if args.debug:
        # args.save_zero = True
        args.low_samples = True

    if args.multi_gpu and args.loss != 'ls':
        logging.warning("multi gpu not implemented for non-mse loss")
        args.multi_gpu = False

    if torch.cuda.device_count() <= 1:
        args.multi_gpu = False

    if(args.bottleneck):
        args.save_zero = False

    if(args.batch_size == 0):
        if args.multi_gpu:
            if args.num_hits <= 30:
                args.batch_size = 128
            else:
                args.batch_size = 32
        else:
            if args.num_hits <= 30:
                args.batch_size = 256
            else:
                args.batch_size = 32

    if(args.n):
        if not (args.no_save_zero_or or args.num_hits == 100): args.save_zero = True

    if(args.lx):
        if not args.no_save_zero_or: args.save_zero = True

    if args.save_epochs == 0:
        if args.num_hits <= 30:
            args.save_epochs = 5
        else: args.save_epochs = 1

    if args.save_model_epochs == 0:
        if args.num_hits <= 30:
            args.save_model_epochs = 5
        else: args.save_model_epochs = 1

    if args.mask_fnd_np:
        logging.info("setting dea true due to mask-fnd-np arg")
        args.dea = True

    if not args.mp_iters_gen: args.mp_iters_gen = args.mp_iters
    if not args.mp_iters_disc: args.mp_iters_disc = args.mp_iters

    args.clabels_first_layer = args.clabels if args.clabels_fl else 0
    args.clabels_hidden_layers = args.clabels if args.clabels_hl else 0

    if args.mask_feat or args.mask_manual or args.mask_learn or args.mask_real_only or args.mask_c or args.mask_learn_sep: args.mask = True
    else: args.mask = False

    if args.noise_padding and not args.mask:
        logging.error("noise padding only works with masking - exiting")
        sys.exit()

    if args.mask_feat: args.node_feat_size += 1

    if args.mask_learn:
        if args.fmg == [0]:
            args.fmg = []

    if args.low_samples:
        args.w1_tot_samples = 1000
        args.w1_num_samples = [10, 100]
        args.num_samples = 1000



    return args


def init_project_dirs(args):
    if args.dir_path == "":
        if args.n: args.dir_path = "/graphganvol/mnist_graph_gan/jets"
        elif args.lx: args.dir_path = "/eos/user/r/rkansal/mnist_graph_gan/jets"
        else: args.dir_path = dirname(realpath(__file__))

    args_dict = vars(args)
    dirs = ['models', 'losses', 'args', 'figs', 'datasets', 'err', 'evaluation', 'outs', 'noise']
    for dir in dirs:
        args_dict[dir + '_path'] = args.dir_path + '/' + dir + '/'
        if not exists(args_dict[dir + '_path']):
            mkdir(args_dict[dir + '_path'])

    args = utils.objectview(args_dict)
    return args


def init_model_dirs(args):
    prev_models = [f[:-4] for f in listdir(args.args_path)]  # removing .txt

    if (args.name in prev_models):
        if args.name != "test" and not args.load_model and not args.override_load_check:
            logging.error("Name already used - exiting")
            sys.exit()

    try: mkdir(args.losses_path + args.name)
    except FileExistsError: logging.debug("losses dir exists")

    try: mkdir(args.models_path + args.name)
    except FileExistsError: logging.debug("models dir exists")

    try: mkdir(args.figs_path + args.name)
    except FileExistsError: logging.debug("figs dir exists")

    return args


def init_logging(args):
    if args.log_file == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    else:
        if args.log_file == "": args.log_file = args.name + ".txt"
        handler = logging.FileHandler(args.outs_path + args.log_file)

    level = getattr(logging, args.log.upper())

    handler.setLevel(level)
    handler.setFormatter(utils.CustomFormatter(args))

    logging.basicConfig(handlers=[handler], level=level, force=True)

    tqdm_out = utils.TqdmToLogger(logging.getLogger(), level=level)
    # print("print test")

    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    return args, tqdm_out


def load_args(args):
    if args.load_model:
        if args.start_epoch == -1:
            prev_models = [int(f[:-3].split('_')[-1]) for f in listdir(args.models_path + args.name + '/')]
            if len(prev_models):
                args.start_epoch = max(prev_models)
            else:
                logging.debug("No model to load from")
                args.start_epoch = 0
                args.load_model = False
        if args.start_epoch == 0: args.load_model = False
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

    return args


def init():
    args = parse_args()
    if args.debug:
        args.log = 'DEBUG'
        args.log_file = 'stdout'
    args = init_project_dirs(args)
    args, tqdm_out = init_logging(args)
    # args = init_logging(args)
    args = init_model_dirs(args)
    args = check_args(args)
    args = load_args(args)
    return args, tqdm_out


def models(args):
    G = Graph_GAN(gen=True, args=deepcopy(args))
    D = Graph_GAN(gen=False, args=deepcopy(args))

    if(args.load_model):
        try:
            G.load_state_dict(torch.load(args.models_path + args.name + "/G_" + str(args.start_epoch) + ".pt", map_location=args.device))
            D.load_state_dict(torch.load(args.models_path + args.name + "/D_" + str(args.start_epoch) + ".pt", map_location=args.device))
        except AttributeError:
            G = torch.load(args.models_path + args.name + "/G_" + str(args.start_epoch) + ".pt", map_location=args.device)
            D = torch.load(args.models_path + args.name + "/D_" + str(args.start_epoch) + ".pt", map_location=args.device)

    if args.multi_gpu:
        logging.info("Using", torch.cuda.device_count(), "GPUs")
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
        # G = DataParallelModel(G)
        # D = DataParallelModel(D)

    G = G.to(args.device)
    D = D.to(args.device)

    return G, D


def optimizers(args, G, D):
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
    elif(args.optimizer == 'adam' or args.optimizer == 'None'):
        G_optimizer = optim.Adam(G_params, lr=args.lr_gen, weight_decay=5e-4, betas=(args.beta1, args.beta2))
        D_optimizer = optim.Adam(D_params, lr=args.lr_disc, weight_decay=5e-4, betas=(args.beta1, args.beta2))

    if(args.load_model):
        G_optimizer.load_state_dict(torch.load(args.models_path + args.name + "/G_optim_" + str(args.start_epoch) + ".pt", map_location=args.device))
        D_optimizer.load_state_dict(torch.load(args.models_path + args.name + "/D_optim_" + str(args.start_epoch) + ".pt", map_location=args.device))

    return G_optimizer, D_optimizer


def losses(args):
    losses = {}

    if(args.load_model):
        losses['D'] = np.loadtxt(args.losses_path + args.name + "/" + "D.txt").tolist()[:args.start_epoch]
        losses['Dr'] = np.loadtxt(args.losses_path + args.name + "/" + "Dr.txt").tolist()[:args.start_epoch]
        losses['Df'] = np.loadtxt(args.losses_path + args.name + "/" + "Df.txt").tolist()[:args.start_epoch]
        losses['G'] = np.loadtxt(args.losses_path + args.name + "/" + "G.txt").tolist()[:args.start_epoch]
        if args.fid: losses['fid'] = np.loadtxt(args.losses_path + args.name + "/" + "fid.txt").tolist()[:args.start_epoch]
        if(args.gp): losses['gp'] = np.loadtxt(args.losses_path + args.name + "/" + "gp.txt").tolist()[:args.start_epoch]

        if args.w1:
            for k in range(len(args.w1_num_samples)):
                losses['w1_' + str(args.w1_num_samples[k]) + 'm'] = np.loadtxt(args.losses_path + args.name + "/w1_" + str(args.w1_num_samples[k]) + 'm.txt')
                losses['w1_' + str(args.w1_num_samples[k]) + 'std'] = np.loadtxt(args.losses_path + args.name + "/w1_" + str(args.w1_num_samples[k]) + 'std.txt')
                if losses['w1_' + str(args.w1_num_samples[k]) + 'm'].ndim == 1: np.expand_dims(losses['w1_' + str(args.w1_num_samples[k]) + 'm'], 0)
                if losses['w1_' + str(args.w1_num_samples[k]) + 'std'].ndim == 1: np.expand_dims(losses['w1_' + str(args.w1_num_samples[k]) + 'std'], 0)
                losses['w1_' + str(args.w1_num_samples[k]) + 'm'] = losses['w1_' + str(args.w1_num_samples[k]) + 'm'].tolist()[:int(args.start_epoch / args.save_epochs) + 1]
                losses['w1_' + str(args.w1_num_samples[k]) + 'std'] = losses['w1_' + str(args.w1_num_samples[k]) + 'std'].tolist()[:int(args.start_epoch / args.save_epochs) + 1]

            if args.jf:
                try:
                    for k in range(len(args.w1_num_samples)):
                        losses['w1j_' + str(args.w1_num_samples[k]) + 'm'] = np.loadtxt(args.losses_path + args.name + "/w1j_" + str(args.w1_num_samples[k]) + 'm.txt')
                        losses['w1j_' + str(args.w1_num_samples[k]) + 'std'] = np.loadtxt(args.losses_path + args.name + "/w1j_" + str(args.w1_num_samples[k]) + 'std.txt')
                        if losses['w1j_' + str(args.w1_num_samples[k]) + 'm'].ndim == 1: np.expand_dims(losses['w1j_' + str(args.w1_num_samples[k]) + 'm'], 0)
                        if losses['w1j_' + str(args.w1_num_samples[k]) + 'std'].ndim == 1: np.expand_dims(losses['w1j_' + str(args.w1_num_samples[k]) + 'std'], 0)
                        # remove
                        losses['w1j_' + str(args.w1_num_samples[k]) + 'm'] = losses['w1j_' + str(args.w1_num_samples[k]) + 'm'][:, :2]
                        losses['w1j_' + str(args.w1_num_samples[k]) + 'std'] = losses['w1j_' + str(args.w1_num_samples[k]) + 'std'][:, :2]
                        # ---
                        losses['w1j_' + str(args.w1_num_samples[k]) + 'm'] = losses['w1j_' + str(args.w1_num_samples[k]) + 'm'].tolist()[:int(args.start_epoch / args.save_epochs) + 1]
                        losses['w1j_' + str(args.w1_num_samples[k]) + 'std'] = losses['w1j_' + str(args.w1_num_samples[k]) + 'std'].tolist()[:int(args.start_epoch / args.save_epochs) + 1]
                except:
                    for k in range(len(args.w1_num_samples)):
                        losses['w1j_' + str(args.w1_num_samples[k]) + 'm'] = []
                        losses['w1j_' + str(args.w1_num_samples[k]) + 'std'] = []
    else:
        losses['D'] = []
        losses['Dr'] = []
        losses['Df'] = []
        losses['G'] = []

        if args.w1:
            for k in range(len(args.w1_num_samples)):
                losses['w1_' + str(args.w1_num_samples[k]) + 'm'] = []
                losses['w1_' + str(args.w1_num_samples[k]) + 'std'] = []

            if args.jf:
                for k in range(len(args.w1_num_samples)):
                    losses['w1j_' + str(args.w1_num_samples[k]) + 'm'] = []
                    losses['w1j_' + str(args.w1_num_samples[k]) + 'std'] = []

        if args.fid: losses['fid'] = []
        if(args.gp): losses['gp'] = []

    return losses
