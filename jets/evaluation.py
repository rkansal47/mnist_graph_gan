# Getting mu and sigma of activation features of GCNN classifier for the FID score

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (graclus, max_pool, global_mean_pool)
from torch_geometric.nn import GMMConv

from tqdm import tqdm

import numpy as np

import utils

from os import path

from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

import logging

cutoff = 0.32178

# need to turn this into a class eventually - can load and save mu and sigma, X features etc.


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
        return self.fc1(x)

        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


def get_mu2_sigma2(args, C, X_loaded, fullpath):
    logging.info("Getting mu2, sigma2")
    activations = 0
    for batch_ndx, data in tqdm(enumerate(X_loaded), total=len(X_loaded)):
        tg_data = utils.tg_transform(args, data.to(args.device))
        if(batch_ndx % args.gpu_batch == 0):
            if(batch_ndx == args.gpu_batch):
                np_activations = activations.cpu().detach().numpy()
            elif(batch_ndx > args.gpu_batch):
                np_activations = np.concatenate((np_activations, activations.cpu().detach().numpy()))
            activations = C(tg_data)
        else:
            activations = torch.cat((C(tg_data), activations), axis=0)
        # if batch_ndx == 113:
        #     break

    activations = np.concatenate((np_activations, activations.cpu().detach().numpy()))  # because torch doesn't have a built in function for calculating the covariance matrix

    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    np.savetxt(fullpath + "mu2.txt", mu)
    np.savetxt(fullpath + "sigma2.txt", sigma)

    return mu, sigma


def load(args, X_loaded):
    C = MoNet(25).to(args.device)
    C.load_state_dict(torch.load(args.evaluation_path + "C_state_dict.pt"))
    numstr = str(args.num) if args.num != -1 else "all_nums"
    dstr = "_sm_nh_" + str(args.num_hits) + "_" if args.sparse_mnist else "_sp_"
    fullpath = args.evaluation_path + numstr + dstr
    logging.debug(fullpath)
    if path.exists(fullpath + "mu2.txt"):
        mu2 = np.loadtxt(fullpath + "mu2.txt")
        sigma2 = np.loadtxt(fullpath + "sigma2.txt")
    else:
        mu2, sigma2 = get_mu2_sigma2(args, C, X_loaded, fullpath)
    return (C, mu2, sigma2)


# make sure to deepcopy G passing in
def get_fid(args, C, G, mu2, sigma2):
    logging.info("Evaluating GFD")
    G.eval()
    C.eval()
    num_iters = np.ceil(float(args.fid_eval_size) / float(args.fid_batch_size))
    with torch.no_grad():
        for i in tqdm(range(int(num_iters))):
            gen_data = utils.tg_transform(args, utils.gen(args, G, args.fid_batch_size))
            if(i == 0):
                activations = C(gen_data)
            else:
                activations = torch.cat((C(gen_data), activations), axis=0)

    activations = activations.cpu().detach().numpy()

    mu1 = np.mean(activations, axis=0)
    sigma1 = np.cov(activations, rowvar=False)

    fid = utils.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    logging.info("GFD:" + str(fid))

    return fid


rng = np.random.default_rng()


# make sure to deepcopy G passing in
def calc_jsd(args, X, G):
    logging.info("evaluating JSD")
    G.eval()

    bins = [np.arange(-1, 1, 0.02), np.arange(-1, 1, 0.02), np.arange(-1, 1, 0.01)]
    N = len(X)

    jsds = []

    for j in tqdm(range(10)):
        gen_out = utils.gen(args, G, num_samples=args.batch_size).cpu().detach().numpy()
        for i in range(int(args.num_samples / args.batch_size)):
            gen_out = np.concatenate((gen_out, utils.gen(args, G, num_samples=args.batch_size).cpu().detach().numpy()), 0)
        gen_out = gen_out[:args.num_samples]

        sample = X[rng.choice(N, size=args.num_samples, replace=False)].cpu().detach().numpy()
        jsd = []

        for i in range(3):
            hist1 = np.histogram(gen_out[:, :, i].reshape(-1), bins=bins[i], density=True)[0]
            hist2 = np.histogram(sample[:, :, i].reshape(-1), bins=bins[i], density=True)[0]
            jsd.append(jensenshannon(hist1, hist2))

        jsds.append(jsd)

    return np.mean(np.array(jsds), axis=0), np.std(np.array(jsds), axis=0)


# make sure to deepcopy G passing in
def calc_w1(args, X, G, losses, X_loaded=None):
    logging.info("Evaluating 1-WD")

    G.eval()
    gen_out = utils.gen_multi_batch(args, G, args.w1_tot_samples)

    logging.info("Generated Data")

    X_rn, mask_real = utils.unnorm_data(args, X.cpu().detach().numpy()[:args.w1_tot_samples], real=True)
    gen_out_rn, mask_gen = utils.unnorm_data(args, gen_out[:args.w1_tot_samples], real=False)

    logging.info("Unnormed data")

    if args.jf:
        realjf = utils.jet_features(X_rn, mask=mask_real)
        genjf = utils.jet_features(gen_out_rn, mask=mask_gen)

        realefp = utils.efp(args, X_rn, mask=mask_real, real=True)
        genefp = utils.efp(args, gen_out_rn, mask=mask_gen, real=False)

    num_batches = np.array(args.w1_tot_samples / np.array(args.w1_num_samples), dtype=int)

    for k in range(len(args.w1_num_samples)):
        logging.info("Num Samples: " + str(args.w1_num_samples[k]))
        w1s = []
        if args.jf: w1js = []
        for j in range(num_batches[k]):
            G_rand_sample = rng.choice(args.w1_tot_samples, size=args.w1_num_samples[k])
            X_rand_sample = rng.choice(args.w1_tot_samples, size=args.w1_num_samples[k])

            Gsample = gen_out_rn[G_rand_sample]
            Xsample = X_rn[X_rand_sample]

            if args.mask:
                mask_gen_sample = mask_gen[G_rand_sample]
                mask_real_sample = mask_real[X_rand_sample]
                parts_real = Xsample[mask_real_sample]
                parts_gen = Gsample[mask_gen_sample]
            else:
                parts_real = Xsample.reshape(-1, args.node_feat_size)
                parts_gen = Gsample.reshape(-1, args.node_feat_size)

            w1 = [wasserstein_distance(parts_real[:, i].reshape(-1), parts_gen[:, i].reshape(-1)) for i in range(3)]
            w1s.append(w1)

            if args.jf:
                realjf_sample = realjf[X_rand_sample]
                genjf_sample = genjf[G_rand_sample]

                realefp_sample = realefp[X_rand_sample]
                genefp_sample = genefp[X_rand_sample]

                w1jf = [wasserstein_distance(realjf_sample[:, i], genjf_sample[:, i]) for i in range(2)]
                w1jefp = [wasserstein_distance(realefp_sample[:, i], genefp_sample[:, i]) for i in range(5)]

                w1js.append([i for t in (w1jf, w1jefp) for i in t])

        losses['w1_' + str(args.w1_num_samples[k]) + 'm'].append(np.mean(np.array(w1s), axis=0))
        losses['w1_' + str(args.w1_num_samples[k]) + 'std'].append(np.std(np.array(w1s), axis=0))

        if args.jf:
            losses['w1j_' + str(args.w1_num_samples[k]) + 'm'].append(np.mean(np.array(w1js), axis=0))
            losses['w1j_' + str(args.w1_num_samples[k]) + 'std'].append(np.std(np.array(w1js), axis=0))

    return gen_out
