# import setGPU

import torch
from model import Graph_GAN
import utils, save_outputs, evaluation, augment
from jets_dataset import JetsDataset
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

import torch.optim as optim
from tqdm import tqdm

from os import listdir, mkdir
from os.path import exists, dirname, realpath

import sys
import argparse
from copy import deepcopy

import numpy as np

from parallel import DataParallelModel, DataParallelCriterion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    dir_path = dirname(realpath(__file__))

    parser = argparse.ArgumentParser()

    # meta

    parser.add_argument("--name", type=str, default="test", help="name or tag for model; will be appended with other info")
    parser.add_argument("--dataset", type=str, default="jets", help="dataset to use", choices=['jets', 'sparse-mnist', 'superpixels'])

    utils.add_bool_arg(parser, "train", "use training or testing dataset for model", default=True, no_name="test")
    parser.add_argument("--ttsplit", type=float, default=0.85, help="ratio of train/test split")

    utils.add_bool_arg(parser, "load-model", "load a pretrained model", default=True)
    utils.add_bool_arg(parser, "override-args", "override original model args when loading with new args", default=False)
    parser.add_argument("--start-epoch", type=int, default=-1, help="which epoch to start training on, only applies if loading a model, by default start at the highest epoch model")
    parser.add_argument("--num-epochs", type=int, default=2000, help="number of epochs to train")

    parser.add_argument("--dir-path", type=str, default=dir_path, help="path where dataset and output will be stored")

    parser.add_argument("--num-samples", type=int, default=10000, help="num samples to evaluate every 5 epochs")

    utils.add_bool_arg(parser, "n", "run on nautilus cluster", default=False)
    utils.add_bool_arg(parser, "bottleneck", "use torch.utils.bottleneck settings", default=False)
    utils.add_bool_arg(parser, "lx", "run on lxplus", default=False)

    utils.add_bool_arg(parser, "save-zero", "save the initial figure", default=False)
    utils.add_bool_arg(parser, "no-save-zero-or", "override --n save-zero default", default=False)
    parser.add_argument("--save-epochs", type=int, default=5, help="save outputs per how many epochs")

    utils.add_bool_arg(parser, "debug", "debug mode", default=False)

    parser.add_argument("--jets", type=str, default="g", help="jet type", choices=['g', 't'])

    utils.add_bool_arg(parser, "real-only", "use jets with ony real particles", default=False)

    utils.add_bool_arg(parser, "multi-gpu", "use multiple gpus if possible", default=True)

    # architecture

    parser.add_argument("--num-hits", type=int, default=30, help="number of hits")
    parser.add_argument("--coords", type=str, default="polarrel", help="cartesian, polarrel or polarrelabspt", choices=['cartesian, polarrel, polarrelabspt'])

    parser.add_argument("--norm", type=float, default=1, help="normalizing max value of features to this value")

    parser.add_argument("--sd", type=float, default=0.2, help="standard deviation of noise")

    parser.add_argument("--node-feat-size", type=int, default=3, help="node feature size")
    parser.add_argument("--hidden-node-size", type=int, default=32, help="hidden vector size of each node (incl node feature size)")
    parser.add_argument("--latent-node-size", type=int, default=0, help="latent vector size of each node - 0 means same as hidden node size")

    parser.add_argument("--clabels", type=int, default=0, help="0 - no clabels, 1 - clabels with pt only, 2 - clabels with pt and detach", choices=[0, 1, 2])
    utils.add_bool_arg(parser, "clabels-fl", "use conditional labels in first layer", default=True)
    utils.add_bool_arg(parser, "clabels-hl", "use conditional labels in hidden layers", default=True)

    parser.add_argument("--fn", type=int, nargs='*', default=[256, 256], help="hidden fn layers e.g. 256 256")
    parser.add_argument("--fe1g", type=int, nargs='*', default=0, help="hidden and output gen fe layers e.g. 64 128 in the first iteration - 0 means same as fe")
    parser.add_argument("--fe1d", type=int, nargs='*', default=0, help="hidden and output disc fe layers e.g. 64 128 in the first iteration - 0 means same as fe")
    parser.add_argument("--fe", type=int, nargs='+', default=[96, 160, 192], help="hidden and output fe layers e.g. 64 128")
    parser.add_argument("--fnd", type=int, nargs='*', default=[128, 64], help="hidden disc output layers e.g. 128 128")
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
    utils.add_bool_arg(parser, "fcg", "use a fully connected graph", default=True)

    parser.add_argument("--glorot", type=float, default=0, help="gain of glorot - if zero then glorot not used")

    utils.add_bool_arg(parser, "gtanh", "use tanh for g output", default=True)
    # utils.add_bool_arg(parser, "dearlysigmoid", "use early sigmoid in d", default=False)

    utils.add_bool_arg(parser, "mask", "use masking for zero-padded particles", default=False)
    utils.add_bool_arg(parser, "mask-weights", "weight D nodes by mask", default=False)
    utils.add_bool_arg(parser, "mask-manual", "manually mask generated nodes with pT less than cutoff", default=False)
    utils.add_bool_arg(parser, "mask-exp", "exponentially decaying mask (instead of binary)", default=False)
    utils.add_bool_arg(parser, "mask-real-only", "only use masking for real jets", default=False)
    parser.add_argument("--mask-epoch", type=int, default=0, help="# of epochs after which to start masking")

    # optimization

    parser.add_argument("--optimizer", type=str, default="rmsprop", help="optimizer - options are adam, rmsprop, adadelta or acgd")
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

    parser.add_argument("--jf", type=str, nargs='*', default=['mass', 'pt'], help='jet level features to evaluate')

    args = parser.parse_args()

    if args.real_only and (not args.jets == 't' or not args.num_hits == 30):
        print("real only arg works only with 30p jets - exiting")
        sys.exit()

    if(args.aug_t or args.aug_f or args.aug_r90 or args.aug_s):
        args.augment = True
    else:
        args.augment = False

    # if not args.coords == 'polarrelabspt':
    #     print("Can't have jet level features for this coordinate system")
    #     args.jf = False
    # elif len(args.jet_features):
    #     args.jf = True

    if(args.int_diffs):
        print("int_diffs not supported yet - exiting")
        sys.exit()

    if(args.dataset == 'jets' and args.augment):
        print("augmentation not implemented for jets yet - exiting")
        sys.exit()

    if(args.optimizer == 'acgd' and (args.num_critic != 1 or args.num_gen != 1)):
        print("acgd can't have num critic or num gen > 1 - exiting")
        sys.exit()

    if(args.n and args.lx):
        print("can't be on nautilus and lxplus both - exiting")
        sys.exit()

    if(args.latent_node_size and args.latent_node_size < 3):
        print("latent node size can't be less than 2 - exiting")
        sys.exit()

    if args.multi_gpu and args.loss != 'ls':
        print("multi gpu not implemented for non-mse loss")
        args.multi_gpu = False

    if torch.cuda.device_count() <= 1:
        args.multi_gpu = False

    if(args.n):
        args.dir_path = "/graphganvol/mnist_graph_gan/jets"
        if not args.no_save_zero_or: args.save_zero = True

    if(args.bottleneck):
        args.save_zero = False

    if(args.lx):
        args.dir_path = "/eos/user/r/rkansal/mnist_graph_gan/jets"
        args.save_zero = True

    if(args.batch_size == 0):
        if args.multi_gpu:
            if args.num_hits == 30:
                args.batch_size = 128
            elif args.num_hits == 100:
                args.batch_size = 32
        else:
            if args.num_hits == 30:
                args.batch_size = 128
            elif args.num_hits == 100:
                args.batch_size = 32

    if not args.mp_iters_gen: args.mp_iters_gen = args.mp_iters
    if not args.mp_iters_disc: args.mp_iters_disc = args.mp_iters

    args.clabels_first_layer = args.clabels if args.clabels_fl else 0
    args.clabels_hidden_layers = args.clabels if args.clabels_hl else 0

    if args.mask:
        args.node_feat_size += 1
    else:
        args.mask_weights = False

    return args


def init(args):
    torch.manual_seed(4)
    torch.autograd.set_detect_anomaly(True)

    args.model_path = args.dir_path + '/models/'
    args.losses_path = args.dir_path + '/losses/'
    args.args_path = args.dir_path + '/args/'
    args.figs_path = args.dir_path + '/figs/'
    args.dataset_path = args.dir_path + '/datasets/'
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
    # if(not exists(args.noise_path)):
    #     mkdir(args.noise_path)
    if(not exists(args.dataset_path)):
        mkdir(args.dataset_path)

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

    args.device = device

    return args


def main(args):
    args = init(args)

    print("loading data")

    X = JetsDataset(args)
    X_loaded = DataLoader(X, shuffle=True, batch_size=args.batch_size, pin_memory=True)

    print("loaded data")

    # model

    if(args.load_model):
        G = torch.load(args.model_path + args.name + "/G_" + str(args.start_epoch) + ".pt", map_location=args.device)
        D = torch.load(args.model_path + args.name + "/D_" + str(args.start_epoch) + ".pt", map_location=args.device)
    else:
        G = Graph_GAN(gen=True, args=deepcopy(args))
        D = Graph_GAN(gen=False, args=deepcopy(args))

    if args.multi_gpu:
        print("Using", torch.cuda.device_count(), "GPUs")
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
        # G = DataParallelModel(G)
        # D = DataParallelModel(D)

    G = G.to(args.device)
    D = D.to(args.device)

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
    elif(args.optimizer == 'adam' or args.optimizer == 'None'):
        G_optimizer = optim.Adam(G_params, lr=args.lr_gen, weight_decay=5e-4, betas=(args.beta1, args.beta2))
        D_optimizer = optim.Adam(D_params, lr=args.lr_disc, weight_decay=5e-4, betas=(args.beta1, args.beta2))

    if(args.load_model):
        G_optimizer.load_state_dict(torch.load(args.model_path + args.name + "/G_optim_" + str(args.start_epoch) + ".pt", map_location=args.device))
        D_optimizer.load_state_dict(torch.load(args.model_path + args.name + "/D_optim_" + str(args.start_epoch) + ".pt", map_location=args.device))

    print("optimizers loaded")

    if args.fid: C, mu2, sigma2 = evaluation.load(args, X_loaded)

    normal_dist = Normal(torch.tensor(0.).to(args.device), torch.tensor(args.sd).to(args.device))
    losses = evaluation.init_losses(args)

    # lns = args.latent_node_size if args.latent_node_size else args.hidden_node_size
    #
    # args.noise_file_name = "num_samples_" + str(args.num_samples) + "_num_nodes_" + str(args.num_hits) + "_latent_node_size_" + str(lns) + "_sd_" + str(args.sd) + ".pt"
    #
    # noise_file_names = listdir(args.noise_path)
    #
    # if args.noise_file_name not in noise_file_names:
    #     torch.save(normal_dist.sample((args.num_samples, args.num_hits, lns)), args.noise_path + args.noise_file_name)

    Y_real = torch.ones(args.batch_size, 1).to(args.device)
    Y_fake = torch.zeros(args.batch_size, 1).to(args.device)

    mse = torch.nn.MSELoss()
    # if args.multi_gpu:
    #     mse = DataParallelCriterion(mse)

    def train_D(data, labels=None, gen_data=None, epoch=0):
        if args.debug: print("dtrain")
        D.train()
        D_optimizer.zero_grad()

        run_batch_size = data.shape[0]

        D_real_output = D(data.clone(), labels, epoch=epoch)

        if args.debug or run_batch_size != args.batch_size:
            print("D real output: ")
            print(D_real_output[:10])

        if gen_data is None:
            gen_data = utils.gen(args, G, normal_dist, run_batch_size, labels=labels)

        if args.augment:
            p = args.aug_prob if not args.adaptive_prob else losses['p'][-1]
            data = augment.augment(args, data, p)
            gen_data = augment.augment(args, gen_data, p)

        if args.debug or run_batch_size != args.batch_size:
            print("gen output: ")
            print(gen_data[:2, :10, :])

        D_fake_output = D(gen_data, labels, epoch=epoch)

        if args.debug or run_batch_size != args.batch_size:
            print("D fake output: ")
            print(D_fake_output[:10])

        D_loss, D_loss_items = utils.calc_D_loss(args, D, data, gen_data, D_real_output, D_fake_output, run_batch_size, Y_real, Y_fake, mse)
        D_loss.backward()

        D_optimizer.step()
        return D_loss_items

    def train_G(data, labels=None, epoch=0):
        if args.debug: print("gtrain")
        G.train()
        G_optimizer.zero_grad()

        run_batch_size = labels.shape[0] if labels is not None else args.batch_size

        gen_data = utils.gen(args, G, normal_dist, run_batch_size, labels=labels)

        if args.augment:
            p = args.aug_prob if not args.adaptive_prob else losses['p'][-1]
            gen_data = augment.augment(args, gen_data, p)

        D_fake_output = D(gen_data, labels, epoch=epoch)

        if args.debug:
            print("D fake output: ")
            print(D_fake_output[:10])

        G_loss = utils.calc_G_loss(args, D_fake_output, Y_real, run_batch_size, mse)

        G_loss.backward()
        G_optimizer.step()

        return G_loss.item()

    def train():
        if(args.fid): losses['fid'].append(evaluation.get_fid(args, C, G, normal_dist, mu2, sigma2))
        # if(args.w1): evaluation.calc_w1(args, X, G, normal_dist, losses)
        if(args.start_epoch == 0 and args.save_zero):
            save_outputs.save_sample_outputs(args, D, G, X[:args.num_samples][0], normal_dist, args.name, 0, losses, X_loaded=X_loaded)

        for i in range(args.start_epoch, args.num_epochs):
            print("Epoch %d %s" % ((i + 1), args.name))
            Dr_loss = 0
            Df_loss = 0
            G_loss = 0
            D_loss = 0
            gp_loss = 0
            lenX = len(X_loaded)
            for batch_ndx, data in tqdm(enumerate(X_loaded), total=lenX):
                if args.clabels:
                    labels = data[1].to(args.device)
                else: labels = None

                data = data[0].to(args.device)

                if args.num_critic > 1 or (batch_ndx == 0 or (batch_ndx - 1) % args.num_gen == 0):
                    D_loss_items = train_D(data, labels=labels, epoch=i)
                    D_loss += D_loss_items['D']
                    Dr_loss += D_loss_items['Dr']
                    Df_loss += D_loss_items['Df']
                    if(args.gp): gp_loss += D_loss_items['gp']

                if args.num_critic == 1 or (batch_ndx - 1) % args.num_critic == 0:
                    G_loss += train_G(data, labels=labels, epoch=i)

                # if(args.num_critic > 1):
                #     D_loss_items = train_D(data, labels=labels)
                #     D_loss += D_loss_items['D']
                #     Dr_loss += D_loss_items['Dr']
                #     Df_loss += D_loss_items['Df']
                #     if(args.gp): gp_loss += D_loss_items['gp']
                #
                #     if((batch_ndx - 1) % args.num_critic == 0):
                #         G_loss += train_G(data, labels=labels)
                # else:
                #     if(batch_ndx == 0 or (batch_ndx - 1) % args.num_gen == 0):
                #         D_loss_items = train_D(data, labels=labels)
                #         D_loss += D_loss_items['D']
                #         Dr_loss += D_loss_items['Dr']
                #         Df_loss += D_loss_items['Df']
                #         if(args.gp): gp_loss += D_loss_items['gp']
                #
                #     G_loss += train_G(data, labels=labels)

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

            if((i + 1) % 5 == 0):
                optimizers = (D_optimizer, G_optimizer)
                save_outputs.save_models(args, D, G, optimizers, args.name, i + 1)
                if args.w1: evaluation.calc_w1(args, X[:][0], G, normal_dist, losses, X_loaded=X_loaded)

            if(args.fid and (i + 1) % 1 == 0):
                losses['fid'].append(evaluation.get_fid(args, C, G, normal_dist, mu2, sigma2))

            if((i + 1) % args.save_epochs == 0):
                # mean, std = evaluation.calc_jsd(args, X, G, normal_dist)
                # print("JSD = " + str(mean) + " ± " + str(std))
                # losses['jsdm'].append(mean)
                # losses['jsdstd'].append(std)
                save_outputs.save_sample_outputs(args, D, G, X[:args.num_samples][0], normal_dist, args.name, i + 1, losses, X_loaded=X_loaded)

    train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
