import numpy as np
import utils
from jets_dataset import JetsDataset
from scipy.stats import wasserstein_distance
from energyflow.emd import emds
import evaluation
import torch
from torch.utils.data import DataLoader

num_samples = 50000

samples_dict = {}

for dataset in ['g', 't', 'q']:
    args = utils.objectview({'datasets_path': 'datasets/', 'ttsplit': 0.7, 'node_feat_size': 3, 'num_hits': 30, 'coords': 'polarrel', 'dataset': 'jets', 'clabels': 0, 'jets': dataset, 'norm': 1, 'mask': True, 'real_only': False})
    X = JetsDataset(args, train=False)
    X = X[:][0]
    X_rn, mask_real = utils.unnorm_data(args, X[:num_samples].cpu().detach().numpy(), real=True)
    samples_dict[dataset] = (X_rn, mask_real)

efps = {}
for dataset in samples_dict.keys():
    samples, mask = samples_dict[dataset]
    efps[dataset] = utils.efp(utils.objectview({'mask': True, 'num_hits': 30}), samples, mask, True)

masses = {}
for dataset in samples_dict.keys():
    samples, mask = samples_dict[dataset]
    masses[dataset] = utils.jet_features(samples, mask=mask)[:, 0]


rng = np.random.default_rng()

w1s = {}

for dataset in samples_dict.keys():
    samples, mask = samples_dict[dataset]
    w1sr = []
    for j in range(5):
        X_rand_sample = rng.choice(num_samples, size=10000)
        X_rand_sample2 = rng.choice(num_samples, size=10000)

        Xsample = samples[X_rand_sample]
        Xsample2 = samples[X_rand_sample2]

        mass_sample = masses[dataset][X_rand_sample]
        mass_sample2 = masses[dataset][X_rand_sample2]

        efp_sample = efps[dataset][X_rand_sample]
        efp_sample2 = efps[dataset][X_rand_sample2]

        w1mass = wasserstein_distance(mass_sample, mass_sample2)
        w1jefp = [wasserstein_distance(efp_sample[:, i], efp_sample2[:, i]) for i in range(5)]

        mask_real_sample = mask[X_rand_sample]
        parts_real = Xsample[mask_real_sample]
        mask_real_sample2 = mask[X_rand_sample2]
        parts_real2 = Xsample2[mask_real_sample2]

        w1r = [wasserstein_distance(parts_real[:, i].reshape(-1), parts_real2[:, i].reshape(-1)) for i in range(3)]

        w1sr.append([i for t in (w1r, [w1mass], w1jefp) for i in t])

    w1s[dataset] = (np.mean(np.array(w1sr), axis=0), np.std(np.array(w1sr), axis=0))





w1s



ave_w1s = {}

for dataset in samples_dict.keys():
    dw1s = w1s[dataset][0]
    dw1std = w1s[dataset][1]
    ave_w1s[dataset] = ((np.mean(dw1s[:3]), dw1s[3], np.mean(dw1s[4:])), (np.linalg.norm(dw1std[:3]) / 3, dw1std[3], np.linalg.norm(dw1std[4:]) / 5))






ave_w1s


samples_dict

for dataset in samples_dict.keys():
    print(dataset)
    # converting into EFP format
    X_rn = np.concatenate((np.expand_dims(samples_dict[dataset][0][:, :, 2], 2), samples_dict[dataset][0][:, :, :2], np.zeros((samples_dict[dataset][0].shape[0], samples_dict[dataset][0].shape[1], 1))), axis=2)

    covs = []
    mmds = []

    for j in range(10):
        X_rand_sample = rng.choice(50000, size=100)
        X_rand_sample2 = rng.choice(50000, size=100)

        Xsample = X_rn[X_rand_sample]
        Xsample2 = X_rn[X_rand_sample2]

        dists = emds(Xsample, Xsample2)

        mmds.append(np.mean(np.min(dists, axis=0)))
        covs.append(np.unique(np.argmin(dists, axis=1)).size / 100)

    print(f"mmds: {np.mean(mmds)}")
    print(f"cov: {np.mean(covs)}")


args_txt = {'g': 'args/236_g30_dea_no_pos_diffs_graphcnngang_mpgand.txt', 't': 'args/237_t30_lrx2_dea_no_pos_diffs_graphcnngang_mpgand.txt', 'q': 'args/238_q30_lrx05_dea_no_pos_diffs_graphcnngang_mpgand.txt'}


for dataset in samples_dict.keys():
    print(dataset)
    args = eval(open(args_txt[dataset]).read())
    args['device'] = torch.device('cpu')
    args['datasets_path'] = './datasets/'
    args['evaluation_path'] = './evaluation/'
    args = utils.objectview(args)

    X_test_loaded = DataLoader(X, batch_size=128)
    C, mu2, sigma2 = evaluation.load(args, X_test_loaded)

    print(evaluation.get_fpnd(args, C, samples_dict[dataset][0], mu2, sigma2))
