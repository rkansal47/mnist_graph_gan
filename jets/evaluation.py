# Getting mu and sigma of activation features of GCNN classifier for the FID score

import torch
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

import numpy as np

import utils

from os import path

from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from energyflow.emd import emds

import logging

from particlenet import ParticleNet


def calc_mu2_sigma2(args, C, X_loaded, fullpath, fjpnd=False):
    logging.info("Getting mu2, sigma2")

    C.eval()
    for i, jet in tqdm(enumerate(X_loaded), total=len(X_loaded)):
        if(i == 0):
            if not fjpnd: activations = C(jet[0][:, :, :3].to(args.device), ret_activations=True).cpu().detach()
            else: activations = torch.cat((C(jet[0][:, :, :3].to(args.device), ret_activations=True), jet[1][:, :args.clabels].to(args.device) * args.fjpnd_alpha), axis=1).cpu().detach()
        else:
            if not fjpnd: activations = torch.cat((C(jet[0][:, :, :3].to(args.device), ret_activations=True).cpu().detach(), activations), axis=0)
            else: activations = torch.cat((torch.cat((C(jet[0][:, :, :3].to(args.device), ret_activations=True), jet[1][:, :args.clabels].to(args.device) * args.fjpnd_alpha), axis=1).cpu().detach(), activations), axis=0)

    activations = activations.numpy()

    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    np.savetxt(fullpath + "mu2.txt", mu)
    np.savetxt(fullpath + "sigma2.txt", sigma)

    return mu, sigma


def get_C(args):
    C = ParticleNet(args.num_hits, args.node_feat_size, device=args.device).to(args.device)
    C.load_state_dict(torch.load(args.evaluation_path + "C_state_dict.pt", map_location=args.device))
    return C


def load(args, C, X_loaded=None):
    fullpath = args.evaluation_path + args.jets
    logging.debug(fullpath)
    if path.exists(fullpath + "mu2.txt") and path.exists(fullpath + "sigma2.txt"):
        mu2 = np.loadtxt(fullpath + "mu2.txt")
        sigma2 = np.loadtxt(fullpath + "sigma2.txt")
    else:
        mu2, sigma2 = calc_mu2_sigma2(args, C, X_loaded, fullpath, fjpnd=False)

    return (mu2, sigma2)


def load_fjpnd(args, C, X_loaded=None):
    fullpath = args.evaluation_path + "fjpnd_pt_" + args.jets
    logging.debug(fullpath)
    if args.clabels == 1:
        if path.exists(fullpath + "mu2.txt") and path.exists(fullpath + "sigma2.txt"):
            cmu2 = np.loadtxt(fullpath + "mu2.txt")
            csigma2 = np.loadtxt(fullpath + "sigma2.txt")
        else:
            cmu2, csigma2 = calc_mu2_sigma2(args, C, X_loaded, fullpath, fjpnd=True)
    elif args.clabels == 2:
        logging.info("FJPND not yet implemented for eta condition")

    return (cmu2, csigma2)


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

# make sure to change in save_outputs as well
pt_regions = [0, 1045, 1175, 3000]

# make sure to deepcopy G passing in
def calc_w1(args, X, G, losses, X_loaded=None, pcgan_args=None):
    logging.info("Evaluating 1-WD")

    G.eval()
    gen_out = utils.gen_multi_batch(args, G, args.eval_tot_samples, X_loaded=X_loaded, pcgan_args=pcgan_args)

    logging.info("Generated Data")

    X_rn, mask_real = utils.unnorm_data(args, X.cpu().detach().numpy()[:args.eval_tot_samples], real=True)
    gen_out_rn, mask_gen = utils.unnorm_data(args, gen_out[:args.eval_tot_samples], real=False)

    logging.info("Unnormed data")

    if args.jf:
        realjf = utils.jet_features(X_rn, mask=mask_real)
        logging.info("Obtained real jet features")

        genjf = utils.jet_features(gen_out_rn, mask=mask_gen)
        logging.info("Obtained gen jet features")

        if args.efp:
            realefp = utils.efp(args, X_rn, mask=mask_real, real=True)
            logging.info("Obtained Real EFPs")

            genefp = utils.efp(args, gen_out_rn, mask=mask_gen, real=False)
            logging.info("Obtained Gen EFPs")

    num_batches = np.array(args.eval_tot_samples / np.array(args.w1_num_samples), dtype=int)

    if args.clabels == 1: abs_labels = (labels[:args.eval_tot_samples, 0] * args.maxjf[0]).detach().numpy()

    for k in range(len(args.w1_num_samples)):
        logging.info("Num Samples: " + str(args.w1_num_samples[k]))
        w1s = []
        if args.jf: w1js = []

        if args.clabels:
            intra_w1s = []
            if args.jf: intra_w1js = []

        for j in range(num_batches[k]):
            G_rand_sample = rng.choice(args.eval_tot_samples, size=args.w1_num_samples[k])
            X_rand_sample = rng.choice(args.eval_tot_samples, size=args.w1_num_samples[k])

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

            if not len(parts_gen): w1 = [1, 1, 1]
            else: w1 = [wasserstein_distance(parts_real[:, i].reshape(-1), parts_gen[:, i].reshape(-1)) for i in range(3)]
            w1s.append(w1)

            if args.jf:
                realjf_sample = realjf[X_rand_sample]
                genjf_sample = genjf[G_rand_sample]
                w1jf = [wasserstein_distance(realjf_sample[:, i], genjf_sample[:, i]) for i in range(2)]

                if args.efp:
                    realefp_sample = realefp[X_rand_sample]
                    genefp_sample = genefp[X_rand_sample]
                    w1jefp = [wasserstein_distance(realefp_sample[:, i], genefp_sample[:, i]) for i in range(5)]
                    w1js.append([i for t in (w1jf, w1jefp) for i in t])
                else: w1js.append(w1jf)

            # Intra-W1
            if args.clabels == 1:
                num_regions = len(pt_regions) - 1
                w1_all = []
                w1j_all = []
                for i in range(num_regions):
                    cut = (abs_labels > pt_regions[i]) * (abs_labels < pt_regions[i + 1])
                    tot_cut = np.sum(cut)

                    Gcut = gen_out_rn[cut]
                    Xcut = X_rn[cut]

                    G_rand_sample = rng.choice(tot_cut, size=int(args.w1_num_samples[k] / num_regions))
                    X_rand_sample = rng.choice(tot_cut, size=int(args.w1_num_samples[k] / num_regions))

                    Gsample = Gcut[G_rand_sample]
                    Xsample = Xcut[X_rand_sample]

                    if args.mask:
                        mask_gen_sample = mask_gen[cut][G_rand_sample]
                        mask_real_sample = mask_real[cut][X_rand_sample]
                        parts_real = Xsample[mask_real_sample]
                        parts_gen = Gsample[mask_gen_sample]
                    else:
                        parts_real = Xsample.reshape(-1, args.node_feat_size)
                        parts_gen = Gsample.reshape(-1, args.node_feat_size)

                    if not len(parts_gen): w1 = [1, 1, 1]
                    else: w1 = [wasserstein_distance(parts_real[:, i].reshape(-1), parts_gen[:, i].reshape(-1)) for i in range(3)]
                    w1_all = w1_all + w1

                    if args.jf:
                        realjf_sample = realjf[cut][X_rand_sample]
                        genjf_sample = genjf[cut][G_rand_sample]
                        w1jf = [wasserstein_distance(realjf_sample[:, i], genjf_sample[:, i]) for i in range(2)]

                        if args.efp:
                            realefp_sample = realefp[cut][X_rand_sample]
                            genefp_sample = genefp[cut][X_rand_sample]
                            w1jefp = [wasserstein_distance(realefp_sample[:, i], genefp_sample[:, i]) for i in range(5)]
                            w1j_all = w1j_all + [i for t in (w1jf, w1jefp) for i in t]
                        else:
                            w1j_all = w1j_all + w1jf

                intra_w1s.append(w1_all)
                intra_w1js.append(w1j_all)

        losses['w1_' + str(args.w1_num_samples[k]) + 'm'].append(np.mean(np.array(w1s), axis=0))
        losses['w1_' + str(args.w1_num_samples[k]) + 'std'].append(np.std(np.array(w1s), axis=0))

        if args.jf:
            losses['w1j_' + str(args.w1_num_samples[k]) + 'm'].append(np.mean(np.array(w1js), axis=0))
            losses['w1j_' + str(args.w1_num_samples[k]) + 'std'].append(np.std(np.array(w1js), axis=0))

        if args.clabels:
            losses['intra_w1_' + str(args.w1_num_samples[k]) + 'm'].append(np.mean(np.array(intra_w1s), axis=0))
            losses['intra_w1_' + str(args.w1_num_samples[k]) + 'std'].append(np.std(np.array(intra_w1s), axis=0))

            if args.jf:
                losses['intra_w1j_' + str(args.w1_num_samples[k]) + 'm'].append(np.mean(np.array(intra_w1js), axis=0))
                losses['intra_w1j_' + str(args.w1_num_samples[k]) + 'std'].append(np.std(np.array(intra_w1js), axis=0))

    return gen_out


def get_fpnd(args, C, gen_out, mu2, sigma2, fjpnd=False, labels=None):
    logging.info("Evaluating FPND") if not fjpnd else logging.info("Evaluating FJPND")

    gen_out_loaded = DataLoader(TensorDataset(torch.tensor(gen_out)), batch_size=args.fpnd_batch_size)
    if fjpnd: clabels = labels[:len(gen_out), :args.clabels]

    logging.info("Getting ParticleNet Activations")
    C.eval()
    for i, gen_jets in tqdm(enumerate(gen_out_loaded), total=len(gen_out_loaded)):
        gen_jets = gen_jets[0]
        if args.mask:
            mask = gen_jets[:, :, 3:4] >= 0
            gen_jets = (gen_jets * mask)[:, :, :3]

        if(i == 0):
            if not fjpnd: activations = C(gen_jets.to(args.device), ret_activations=True).cpu().detach()
            else: activations = torch.cat((C(gen_jets.to(args.device), ret_activations=True), clabels[i * args.fpnd_batch_size:(i + 1) * args.fpnd_batch_size].to(args.device) * args.fjpnd_alpha), axis=1).cpu().detach()
        else:
            if not fjpnd: activations = torch.cat((C(gen_jets.to(args.device), ret_activations=True).cpu().detach(), activations), axis=0)
            else: activations = torch.cat((torch.cat((C(gen_jets.to(args.device), ret_activations=True), clabels[i * args.fpnd_batch_size:(i + 1) * args.fpnd_batch_size].to(args.device) * args.fjpnd_alpha), axis=1).cpu().detach(), activations), axis=0)

    activations = activations.numpy()

    mu1 = np.mean(activations, axis=0)
    sigma1 = np.cov(activations, rowvar=False)

    fpnd = utils.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    logging.info("FPND: " + str(fpnd)) if not fjpnd else logging.info("FJPND: " + str(fpnd))

    return fpnd



def calc_cov_mmd(args, X, gen_out, losses, labels=None):
    X_rn, mask_real = utils.unnorm_data(args, X.cpu().detach().numpy()[:args.eval_tot_samples], real=True)
    gen_out_rn, mask_gen = utils.unnorm_data(args, gen_out[:args.eval_tot_samples], real=False)

    # converting into EFP format
    X_rn = np.concatenate((np.expand_dims(X_rn[:, :, 2], 2), X_rn[:, :, :2], np.zeros((X_rn.shape[0], X_rn.shape[1], 1))), axis=2)
    gen_out_rn = np.concatenate((np.expand_dims(gen_out_rn[:, :, 2], 2), gen_out_rn[:, :, :2], np.zeros((gen_out_rn.shape[0], gen_out_rn.shape[1], 1))), axis=2)

    if args.clabels == 1: abs_labels = (labels[:args.eval_tot_samples, 0] * args.maxjf[0]).detach().numpy()

    logging.info("Calculating coverage and MMD")
    covs = []
    mmds = []

    if args.clabels == 1:
        intra_covs = []
        intra_mmds = []

    for j in range(args.cov_mmd_num_batches):
        G_rand_sample = rng.choice(args.eval_tot_samples, size=args.cov_mmd_num_samples)
        X_rand_sample = rng.choice(args.eval_tot_samples, size=args.cov_mmd_num_samples)

        Gsample = gen_out_rn[G_rand_sample]
        Xsample = X_rn[X_rand_sample]

        dists = emds(Gsample, Xsample)

        mmds.append(np.mean(np.min(dists, axis=0)))
        covs.append(np.unique(np.argmin(dists, axis=1)).size / args.cov_mmd_num_samples)

        # Intra-W1
        if args.clabels == 1:
            num_regions = len(pt_regions) - 1
            covs_all = []
            mmds_all = []
            for i in range(num_regions):
                cut = (abs_labels > pt_regions[i]) * (abs_labels < pt_regions[i + 1])
                tot_cut = np.sum(cut)

                Gcut = gen_out_rn[cut]
                Xcut = X_rn[cut]

                G_rand_sample = rng.choice(tot_cut, size=args.cov_mmd_num_samples)
                X_rand_sample = rng.choice(tot_cut, size=args.cov_mmd_num_samples)

                Gsample = Gcut[G_rand_sample]
                Xsample = Xcut[X_rand_sample]

                dists = emds(Gsample, Xsample)

                mmds_all += [np.mean(np.min(dists, axis=0))]
                covs_all += [np.unique(np.argmin(dists, axis=1)).size / args.cov_mmd_num_samples]

            intra_covs.append(covs_all)
            intra_mmds.append(mmds_all)

    losses['coverage'].append(np.mean(np.array(covs)))
    losses['mmd'].append(np.mean(np.array(mmds)))

    if args.clabels == 1:
        losses['intra_coverage'].append(np.mean(np.array(intra_covs), axis=0))
        losses['intra_mmd'].append(np.mean(np.array(intra_mmds), axis=0))
