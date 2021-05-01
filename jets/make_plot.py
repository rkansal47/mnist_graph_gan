import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
from jets_dataset import JetsDataset
from torch.distributions.normal import Normal
import mplhep as hep
from skhep.math.vectors import LorentzVector
from tqdm import tqdm
from scipy.stats import wasserstein_distance
import energyflow as ef
import energyflow.utils as ut
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from model import Graph_GAN
from ext_models import GraphCNNGANG
import importlib
import save_outputs
import math
# plt.switch_backend('macosx')
plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)

import awkward1 as ak
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)

from guppy import hpy
h = hpy()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = 179
epoch = 5
name = str(model) + '_' + str(epoch)
figpath = "figs/" + str(model) + '/' + name

# w1m = np.loadtxt('./losses/82/w1_10000m.txt')
w1m = np.loadtxt('./w1j_10000m.txt')

w1m
(np.argsort(w1m[:, 0])[:20] * 5, np.sort(w1m[:, 0])[:20])

(np.argsort(w1m[:, 1])[:20] * 5, np.sort(w1m[:, 1])[:20])

np.where(np.argsort(w1m[:, 0]) == np.argsort(w1m[:, 1]))

np.argsort(w1m[:, 0])[:20] * 5

s1 = "["
s2 = "["
j = 1
for i in np.argsort(w1m[:, 1])[:20]:
    s1 += str(np.where(np.argsort(w1m[:, 0]) == i)[0][0] + 1) + ", "
    s2 += str(np.where(np.argsort(w1m[:, 0]) == i)[0][0] + 1 + j) + ", "
    j += 1
s1 = s1[:-2] + "]"
s2 = s2[:-2] + "]"
print(s1)
print(s2)

np.argsort(np.linalg.norm(w1m[:, :3], axis=1))[:20] * 5

# realw1m = [0.00584264, 0.00556786, 0.0014096]
# realw1std = [0.00214083, 0.00204827, 0.00051136]

batch_size = 128
normal_dist = Normal(torch.tensor(0.).to(device), torch.tensor(0.2).to(device))
dir = './'
# dir = '/graphganvol/mnist_graph_gan/jets/'

args = utils.objectview({'datasets_path': dir + 'datasets/', 'figs_path': dir + 'figs/' + str(model), 'node_feat_size': 3, 'num_hits': 30, 'coords': 'polarrel', 'latent_node_size': 32, 'clabels': 1, 'jets': 'g', 'norm': 1, 'mask': False, 'mask_manual': False, 'real_only': False, 'mask_feat': False})

args = eval(open("./args/" + "179_t30_graphcnngan_knn_20.txt").read())
args['device'] = device
args['datasets_path'] = dir + 'datasets/'
# args['mask_feat'] = False
# args['mask_learn'] = False
# args['mask_c'] = False
args['figs_path'] = dir + 'figs/' + str(model) + '/' + str(epoch)
args = utils.objectview(args)

args

X = JetsDataset(args)


labels = X[:][1]
# X_loaded = DataLoader(X, shuffle=True, batch_size=32, pin_memory=True)
X = X[:][0]
N = len(X)

rng = np.random.default_rng()

num_samples = 1000

# G = torch.load('./models/' + str(model) + '/G_' + str(epoch) + '.pt', map_location=device)
# G = Graph_GAN(True, args)
G = GraphCNNGANG(args)
G.load_state_dict(torch.load('./models/' + str(model) + '/G_' + str(epoch) + '.pt', map_location=device))

importlib.reload(utils)

G.eval()
gen_out = utils.gen_multi_batch(args, G, num_samples, labels=labels, use_tqdm=True)

labels

# G.eval()
# if args.clabels:
#     gen_out = utils.gen(args, G, num_samples=batch_size, labels=labels[:128]).cpu().detach().numpy()
#     for i in tqdm(range(int(num_samples / batch_size))):
#         gen_out = np.concatenate((gen_out, utils.gen(args, G, dist=normal_dist, num_samples=batch_size, labels=labels[128 * (i + 1):128 * (i + 2)]).cpu().detach().numpy()), 0)
#     gen_out = gen_out[:num_samples]
# else:
#     gen_out = utils.gen(args, G, num_samples=batch_size).cpu().detach().numpy()
#     for i in tqdm(range(int(num_samples / batch_size))):
#         gen_out = np.concatenate((gen_out, utils.gen(args, G, dist=normal_dist, num_samples=batch_size).cpu().detach().numpy()), 0)
#     gen_out = gen_out[:num_samples]
#
# model
# name
# np.save('./models/' + str(model) + '/' + name + "_gen_out", gen_out)

gen_out = np.load('./models/' + str(model) + '/' + name + "_gen_out.npy")

X_rn, mask_real = utils.unnorm_data(args, X[:num_samples].cpu().detach().numpy(), real=True)
gen_out_rn, mask_gen = utils.unnorm_data(args, gen_out, real=False)
#
# gen_out_rn = gen_out[:, :, :3]
# gen_out_rn = gen_out_rn / args.norm
# # gen_out_rn[:, :, 2] += 0.5
# gen_out_rn *= args.maxepp
#
# for i in range(num_samples):
#     for j in range(args.num_hits):
#         if gen_out_rn[i][j][2] < 0:
#             gen_out_rn[i][j][2] = 0

print(X_rn.shape)
print(gen_out_rn.shape)

print(X_rn[0][:10])
print(gen_out_rn[0][:10])


realjf = utils.jet_features(X_rn, args.mask, mask_real)
genjf = utils.jet_features(gen_out_rn, args.mask, mask_gen)

genjf

real_masses = realjf[:, 0]
gen_masses = genjf[:, 0]

gen_masses

len(real_masses)
len(gen_masses)

gen_out_rn

importlib.reload(save_outputs)
%matplotlib inline

plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)
save_outputs.plot_jet_mass_pt(args, realjf, genjf, '_jets_mass_pt', show=True)

importlib.reload(save_outputs)
%matplotlib inline
save_outputs.plot_part_feats_jet_mass(args, X_rn, mask_real, gen_out_rn, mask_gen, realjf, genjf, '', show=True)

h.heap()

importlib.reload(save_outputs)
%matplotlib inline
save_outputs.plot_part_feats(args, X_rn, mask_real, gen_out_rn, mask_gen, 'p', show=True)

h.heap()


importlib.reload(utils)
realefp = utils.efp(args, X_rn, mask=mask_real, real=True)
genefp = utils.efp(args, gen_out_rn, mask=mask_gen, real=False)
save_outputs.plot_jet_feats(args, realjf, genjf, realefp, genefp, 'j', show=True)


w1s = []
w1sr = []
w1js = []
w1jsr = []
for j in range(1000):
    G_rand_sample = rng.choice(num_samples, size=100)
    X_rand_sample = rng.choice(num_samples, size=100)
    X_rand_sample2 = rng.choice(num_samples, size=100)

    Gsample = gen_out_rn[G_rand_sample]
    Xsample = X_rn[X_rand_sample]
    Xsample2 = X_rn[X_rand_sample2]

    realjf_sample = realjf[X_rand_sample]
    realjf_sample2 = realjf[X_rand_sample2]
    genjf_sample = genjf[G_rand_sample]

    realefp_sample = realefp[X_rand_sample]
    realefp_sample2 = realefp[X_rand_sample2]
    genefp_sample = genefp[X_rand_sample]

    w1jf = [wasserstein_distance(realjf_sample[:, i], genjf_sample[:, i]) for i in range(2)]
    w1jefp = [wasserstein_distance(realefp_sample[:, i], genefp_sample[:, i]) for i in range(5)]

    if math.isnan(w1jf[0]):
        print("nan")
        print(genjf_sample)
        break

    w1jfr = [wasserstein_distance(realjf_sample[:, i], realjf_sample2[:, i]) for i in range(2)]
    w1jefpr = [wasserstein_distance(realefp_sample[:, i], realefp_sample2[:, i]) for i in range(5)]

    w1js.append([i for t in (w1jf, w1jefp) for i in t])
    w1jsr.append([i for t in (w1jfr, w1jefpr) for i in t])

    if args.mask:
        mask_gen_sample = mask_gen[G_rand_sample]
        mask_real_sample = mask_real[X_rand_sample]
        parts_real = Xsample[mask_real_sample]
        parts_gen = Gsample[mask_gen_sample]
    else:
        parts_real = Xsample.reshape(-1, 3)
        parts_real2 = Xsample2.reshape(-1, 3)
        parts_gen = Gsample.reshape(-1, 3)

    w1 = [wasserstein_distance(parts_real[:, i].reshape(-1), parts_gen[:, i].reshape(-1)) for i in range(3)]
    w1r = [wasserstein_distance(parts_real[:, i].reshape(-1), parts_real2[:, i].reshape(-1)) for i in range(3)]

    w1s.append(w1)
    w1sr.append(w1r)


w1s
w1js

w1jsr

np.mean(np.array(w1js)[:, 0])

np.mean(np.array(w1js)[:, 2])
np.std(np.array(w1js)[:, 2])
np.mean(np.array(w1jsr)[:, 2])
np.std(np.array(w1jsr)[:, 2])




if args.coords == 'cartesian':
    plabels = ['$p_x$ (GeV)', '$p_y$ (GeV)', '$p_z$ (GeV)']
    bin = np.arange(-500, 500, 10)
    pbins = [bin, bin, bin]
elif args.coords == 'polarrel':
    plabels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T^{rel}$']
    if args.jets == 'g' or args.jets == 'q' or args.jets == 'w' or args.jets == 'z':
        if args.num_hits == 100:
            pbins = [np.linspace(-0.5, 0.5, 101), np.linspace(-0.5, 0.5, 101), np.linspace(0, 0.1, 101)]
        else:
            pbins = [np.linspace(-0.3, 0.3, 101), np.linspace(-0.3, 0.3, 101), np.linspace(0, 0.2, 101)]
    elif args.jets == 't':
        pbins = [np.linspace(-0.4, 0.4, 101), np.linspace(-0.4, 0.4, 101), np.linspace(0, 0.2, 101)]
elif args.coords == 'polarrelabspt':
    plabels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T (GeV)$']
    pbins = [np.arange(-0.5, 0.5, 0.01), np.arange(-0.5, 0.5, 0.01), np.arange(0, 400, 4)]

if args.jets == 'g' or args.jets == 'q': mbins = np.linspace(0, 0.225, 51)
elif args.jets == 't': mbins = np.linspace(0.02, 0.21, 51)
else: mbins = np.linspace(0, 0.12, 51)

if args.mask:
    parts_real = X_rn[mask_real]
    parts_gen = gen_out_rn[mask_gen]
else:
    parts_real = X_rn.reshape(-1, args.node_feat_size)
    parts_gen = gen_out_rn.reshape(-1, args.node_feat_size)

pbin_range = 10
pzmeans = []
pzstds = []

for i in range(3):
    zscores = (np.histogram(parts_real[:, i], pbins[i])[0] - np.histogram(parts_gen[:, i], pbins[i])[0]) / (np.histogram(parts_real[:, i], pbins[i])[0])
    zscores[zscores == -np.inf] = np.nan
    pzmeans.append(np.nanmean(zscores.reshape(-1, bin_range), axis=1))
    pzstds.append(np.nanstd(zscores.reshape(-1, bin_range), axis=1))

mbin_range = 5

zscores = (np.histogram(realjf[:, 0], mbins)[0] - np.histogram(genjf[:, 0], mbins)[0]) / (np.histogram(realjf[:, 0], mbins)[0])
zscores[zscores == -np.inf] = np.nan
mzmeans = np.nanmean(zscores.reshape(-1, mbin_range), axis=1)
mzstds = np.nanstd(zscores.reshape(-1, mbin_range), axis=1)

len(np.mean(mbins[1:].reshape(-1, mbin_range), axis=1))
len(mzmeans)

fig, axs = plt.subplots(2, 4, figsize=(30, 8), gridspec_kw = {'hspace': 0, 'height_ratios': [4, 1]})
# plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)

for i in range(3):
    axs[0, i].ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    _ = axs[0, i].hist(parts_real[:, i], pbins[i], histtype='step', label='Real', color='red')
    _ = axs[0, i].hist(parts_gen[:, i], pbins[i], histtype='step', label='Generated', color='blue')
    axs[0, i].set_ylabel('Number of Particles')
    # if losses is not None: axs[0, i].title('$W_1$ = {:.2e}'.format(losses['w1_' + str(args.w1_num_samples[-1]) + 'm'][-1][i]), fontsize=12)
    axs[0, i].legend(loc=1, prop={'size': 18})

    axs[1, i].errorbar(np.mean(pbins[i][1:].reshape(-1, pbin_range), axis=1), pzmeans[i], yerr=pzstds[i], fmt='x', color='black')
    axs[1, i].set_ylabel('z-score', x = 0.5)
    axs[1, i].set_xlabel('Particle ' + plabels[i])
    # axs[1, i].set_xlim([0, 0.22])
    axs[1, i].set_ylim([-1, 1])
    axs[1, i].grid(which='both', axis='y')

axs[0, 3].ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = axs[0, 3].hist(realjf[:, 0], mbins, histtype='step', label='Real', color='red')
_ = axs[0, 3].hist(genjf[:, 0], mbins, histtype='step', label='Generated', color='blue')
axs[0, 3].set_ylabel('Number of Jets')
# if losses is not None: axs[0, 3].title('$W_1$ = {:.2e}'.format(losses['w1_' + str(args.w1_num_samples[-1]) + 'm'][-1][i]), fontsize=12)
axs[0, 3].legend(loc=1, prop={'size': 18})

axs[1, 3].errorbar(np.mean(mbins[1:].reshape(-1, mbin_range), axis=1), mzmeans, yerr=mzstds, fmt='x', color='black')
axs[1, 3].set_ylabel('z-score', x = 0.5)
axs[1, 3].set_xlabel('Jet $m/p_T$')
# axs[1, 3].set_xlim([0, 0.22])
axs[1, 3].set_ylim([-1, 1])
axs[1, 3].grid(which='both', axis='y')

plt.savefig(figpath + '_feat_mass_ratioplot.pdf', bbox_inches='tight')
plt.show()




















num_pixels = 100
Njets = 100000
img_width = 0.8

average_real_jet_image = np.zeros((num_pixels, num_pixels, 1))
for i in range(Njets):
    average_real_jet_image += ut.pixelate(X_efp_format[i, :, :], npix=num_pixels, img_width=img_width, nb_chan=1, norm=False, charged_counts_only=False)
average_real_jet_image /= Njets

average_gen_jet_image = np.zeros((num_pixels, num_pixels, 1))
for i in range(Njets):
    average_gen_jet_image += ut.pixelate(gen_out_rn_efp_format[i, :, :], npix=num_pixels, img_width=img_width, nb_chan=1, norm=False, charged_counts_only=False)
average_gen_jet_image /= Njets

plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(20, 10))
fig.add_subplot(1, 2, 1)
plt.imshow(average_real_jet_image[:, :, 0], origin='lower', cmap = 'binary', norm=LogNorm(vmin=1e-7))
plt.colorbar()
plt.title("Average of Real Jets", fontsize=25)
plt.xlabel('$i\phi$')
plt.ylabel('$i\eta$')

fig.add_subplot(1, 2, 2)
plt.imshow(average_gen_jet_image[:, :, 0], origin='lower', cmap='binary', norm=LogNorm(vmin=1e-7))
plt.colorbar()
plt.title("Average of Generated Jets", fontsize=25)
plt.xlabel('$i\phi$')
plt.ylabel('$i\eta$')

plt.savefig(figpath + '_jets_ave.pdf', dpi=250, bbox_inches='tight')
plt.show()
plt.rcParams.update({'font.size': 16})


np_gen = np.sum(mask_gen, axis=1)
np_real = np.sum(mask_real, axis=1)

30 * num_samples - np.sum(np_gen)
30 * num_samples - np.sum(np_real)


fig = plt.figure(figsize=(5, 10))
plt.xlabel('# particles')
plt.ylabel('# of jets with N particles')
_ = plt.hist(np_real, np.linspace(25, 30, 6), histtype='step', label='Real', color='red')
_ = plt.hist(np_gen, np.linspace(25, 30, 6), histtype='step', label='Gen', color='blue')
plt.legend(loc=2)
plt.savefig(figpath + '_np.pdf', dpi=250, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(5, 10))
plt.xlabel('# particles')
plt.ylabel('# of jets with N particles')
_ = plt.hist(np_real, np.linspace(25, 30, 6), histtype='step', label='Real', color='red')
_ = plt.hist(np_gen, np.linspace(25, 30, 6), histtype='step', label='Gen', color='blue')
plt.legend(loc=2)
plt.ylim(0, 10000)
plt.savefig(figpath + '_np_10000.pdf', dpi=250, bbox_inches='tight')
plt.show()



efpset2 = ef.EFPSet(('n==', 2), ('d==', 2), ('p==', 1), measure='hadr', beta=1, normed=None, coords='ptyphim')

gen_out_rn_efp2 = efpset2.batch_compute(gen_out_rn_efp_format)
X_efp2 = efpset2.batch_compute(X_efp_format)


fig = plt.figure(figsize=(40, 10))

binsm = np.arange(0, 0.25, 0.0025)
fig.add_subplot(1, 4, 3)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_masses, bins=binsm, histtype='step', label='Real', color='red')
_ = plt.hist(gen_masses, bins=binsm, histtype='step', label='Generated', color='blue')
plt.xlabel('Jet Relative Mass')
plt.ylabel('Jets')
plt.legend(loc=1, prop={'size': 18})

bins = np.arange(0, 0.07, 0.0007)

fig.add_subplot(1, 4, 1)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(X_efp2[:, 0], bins, histtype='step', label='Real', color='red')
_ = plt.hist(gen_out_rn_efp2[:, 0], bins, histtype='step', label='Generated', color='blue')
plt.xlabel('EFP (d=2, n=2)')
plt.ylabel('Jets')
lg = plt.legend(loc=1, prop={'size': 18})

fig.add_subplot(1, 4, 2)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(np.sqrt(X_efp2[:, 0] / 2), binsm, histtype='step', label='Real', color='red')
_ = plt.hist(np.sqrt(gen_out_rn_efp2[:, 0] / 2), binsm, histtype='step', label='Generated', color='blue')
plt.xlabel('$\sqrt{EFP/2}$')
plt.ylabel('Jets')
lg = plt.legend(loc=1, prop={'size': 18})

fig.add_subplot(1, 4, 4)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(np.sqrt(X_efp2[:, 0] / 2), binsm, histtype='step', label='Real EFP', color='red')
_ = plt.hist(np.sqrt(gen_out_rn_efp2[:, 0] / 2), binsm, histtype='step', label='Generated EFP', color='blue')
_ = plt.hist(real_masses, bins=binsm, histtype='step', label='Real Mass', color='orange')
_ = plt.hist(gen_masses, bins=binsm, histtype='step', label='Generated Mass', color='green')
# plt.xlabel('$\sqrt{EFP/2}$')
plt.ylabel('Jets')
lg = plt.legend(loc=1, prop={'size': 18})
plt.savefig(figpath + "_100000_jets_efp_mass_comp.pdf", bbox_inches='tight')
plt.show()


plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(np.sqrt(X_efp2[:, 0] / 2), binsm, histtype='step', label='Real', color='red')
_ = plt.hist(np.sqrt(gen_out_rn_efp2[:, 0] / 2), binsm, histtype='step', label='Generated', color='blue')
plt.xlabel('Jet $m/p_T$')
plt.ylabel('Jets')
lg = plt.legend(loc=1, prop={'size': 18})
plt.savefig(figpath + "_100000_jets_efp_mass.pdf", bbox_inches='tight')
plt.show()

real_efp_masses = np.sqrt(X_efp2[:, 0] / 2)
gen_efp_masses = np.sqrt(gen_out_rn_efp2[:, 0] / 2)

real_mass_diff = real_masses - real_efp_masses
gen_mass_diff = gen_masses - gen_efp_masses

real_mass_hist_diff = np.histogram(real_masses, binsm)[0] - np.histogram(real_efp_masses, binsm)[0]
gen_mass_hist_diff = np.histogram(gen_masses, binsm)[0] - np.histogram(gen_efp_masses, binsm)[0]

binsmd = np.arange(-7e-4, 7e-4, 7e-6)

fig = plt.figure(figsize=(20, 10))
fig.add_subplot(1, 2, 1)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_mass_diff, binsmd, histtype='step', label='Real', color='red')
_ = plt.hist(gen_mass_diff, binsmd, histtype='step', label='Generated', color='blue')
plt.xlabel('Mass Difference (Exact - EFP)', x = 0.5)
plt.ylabel('Jets')
lg = plt.legend(loc=1, prop={'size': 18})

fig.add_subplot(1, 2, 2)
plt.plot(binsm[:-1], real_mass_hist_diff, label='Real', color='red')
plt.plot(binsm[:-1], gen_mass_hist_diff, label='Generated', color='blue')
plt.xlabel('Mass', x = 0.5)
plt.ylabel('# Exact - # EFP Jets')
lg = plt.legend(loc=1, prop={'size': 18})
plt.savefig(figpath + "_100000_jets_mass_diff.pdf", bbox_inches='tight')
plt.show()

Nangle = 1000

real_delta_etaphi = []
gen_delta_etaphi = []

for i in range(Nangle):
    jet = X_rn[i]
    for j in range(30):
        for k in range(j + 1, 30):
            real_delta_etaphi.append([np.abs(jet[j][0] - jet[k][0]), np.abs(jet[j][1] - jet[k][1])])


for i in range(Nangle):
    jet = gen_out_rn[i]
    for j in range(30):
        for k in range(j + 1, 30):
            gen_delta_etaphi.append([np.abs(jet[j][0] - jet[k][0]), np.abs(jet[j][1] - jet[k][1])])

real_delta_etaphi = np.array(real_delta_etaphi)
gen_delta_etaphi = np.array(gen_delta_etaphi)

binseta = np.arange(0, 0.6, 0.006)
binsphi = np.arange(0, 0.4, 0.004)


fig = plt.figure(figsize=(40, 10))
fig.add_subplot(1, 4, 1)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_delta_etaphi[:, 0], binseta, histtype='step', label='Real', color='red')
_ = plt.hist(gen_delta_etaphi[:, 0], binseta, histtype='step', label='Generated', color='blue')
plt.xlabel('$\Delta\eta$')
plt.ylabel('#')
lg = plt.legend(loc=1, prop={'size': 18})

fig.add_subplot(1, 4, 2)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_delta_etaphi[:, 1], binseta, histtype='step', label='Real', color='red')
_ = plt.hist(gen_delta_etaphi[:, 1], binseta, histtype='step', label='Generated', color='blue')
plt.xlabel('$\Delta\phi$')
plt.ylabel('#')
lg = plt.legend(loc=1, prop={'size': 18})

fig.add_subplot(1, 4, 3)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_delta_etaphi[:, 0], binseta, histtype='step', label='Real $\Delta \phi$', color='red')
_ = plt.hist(real_delta_etaphi[:, 1], binseta, histtype='step', label='Real $\Delta \eta$', color='orange')
# plt.xlabel('$\$')
plt.ylabel('#')
lg = plt.legend(loc=1, prop={'size': 18})

fig.add_subplot(1, 4, 4)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(gen_delta_etaphi[:, 0], binseta, histtype='step', label='Generated $\Delta \phi$', color='blue')
_ = plt.hist(gen_delta_etaphi[:, 1], binseta, histtype='step', label='Generated $\Delta \eta$', color='green')
# plt.xlabel('$\$')
plt.ylabel('#')
lg = plt.legend(loc=1, prop={'size': 18})

plt.savefig(figpath + "_" + str(Nangle) + "_jets_delta_eta_phi.pdf", bbox_inches='tight')
plt.show()



N = 100000

mass_diffs = []

for i in range(N):
    jetv = LorentzVector()

    for part in gen_out_rn[i]:
        vec = LorentzVector()
        vec.setptetaphim(part[2], part[0], part[1], 0)
        jetv += vec

    dmass = jetv.mass

    efp = efpset2.compute(gen_out_rn_efp_format[i])[0]

    if(efp < 0):
        efpmass = -10
    else:
        efpmass = np.sqrt(efp / 2)

    mass_diffs.append([dmass, efpmass, dmass - efpmass])

mass_diffs = np.array(mass_diffs)

len(mass_diffs[mass_diffs[:, 1] < 0])

binsm10000 = np.arange(0, 0.25, 0.005)
fig = plt.figure()
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(mass_diffs[:, 0], binsm, histtype='step', log=True, label='Direct', color='blue')
_ = plt.hist(mass_diffs[:, 1], binsm, histtype='step', log=True, label='EFP', color='green')
plt.ylabel('#')
lg = plt.legend(loc=1, prop={'size': 18})
plt.show()

plt.hist(mass_diffs[:, 2], bins=np.arange(-0.002, 0.002, 0.00002), histtype='step', log=True)

gen_mass_hist_diff = np.histogram(mass_diffs[:, 0], binsm)[0] - np.histogram(mass_diffs[:, 1], binsm)[0]

plt.plot(binsm[:-1], gen_mass_hist_diff)

mass_diffs





abs_labels = (labels[:num_samples] * args.maxjf[0]).detach().numpy()

regions = [1045, 1175]
cregions = [(abs_labels < regions[0]).squeeze(), ((abs_labels >= regions[0]) * (abs_labels < regions[1])).squeeze(), (abs_labels >= regions[1]).squeeze()]

len(X_rn[cregions[0]])
len(X_rn[cregions[1]])
len(X_rn[cregions[2]])

bins = [np.arange(-0.3, 0.3, 0.005), np.arange(-0.3, 0.3, 0.005), np.arange(0, 0.2, 0.002)]
binsm = np.arange(0, 0.225, 0.0045)

fig, axs = plt.subplots(3, 4, figsize=(30, 20))

for j in range(3):
    for i in range(3):
        axs[j, i].ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
        _ = axs[j, i].hist(X_rn[cregions[j]][:, :, i].reshape(-1), bins[i], histtype='step', label='Real', color='red')
        _ = axs[j, i].hist(gen_out_rn[cregions[j]][:, :, i].reshape(-1), bins[i], histtype='step', label='Generated', color='blue')
        axs[j, i].set_xlabel('Particle ' + plabels[i])
        if i == 0:
            axs[j, i].set_ylabel('Region ' + str(j + 1) + ' \t \t Particles')
        else:
            axs[j, i].set_ylabel('Particles')
        lg = axs[j, i].legend(loc=1, prop={'size': 18})

    axs[j, 3].ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    _ = axs[j, 3].hist(np.array(real_masses)[cregions[j]], bins=binsm, histtype='step', label='Real', color='red')
    _ = axs[j, 3].hist(np.array(gen_masses)[cregions[j]], bins=binsm, histtype='step', label='Generated', color='blue')
    axs[j, 3].set_xlabel('Jet $m/p_{T}$')
    axs[j, 3].set_ylabel('Jets')
    axs[j, 3].legend(loc=1, prop={'size': 18})

plt.tight_layout(pad=2.0)
plt.savefig(figpath + "_clabels.pdf", bbox_inches='tight')
plt.show()


abs_real_masses = real_masses[:num_samples] * abs_labels.squeeze()[:num_samples]
abs_gen_masses = gen_masses[:num_samples] * abs_labels.squeeze()[:num_samples]

binsabsm = np.arange(0, 300, 6)

fig = plt.figure(figsize=(30, 10))

for i in range(3):
    fig.add_subplot(1, 3, i + 1)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(abs_real_masses[cregions[i]], binsabsm, histtype='step', label='Real', color='red')
    _ = plt.hist(abs_gen_masses[cregions[i]], binsabsm, histtype='step', label='Generated', color='blue')
    plt.xlabel('Jet mass (GeV)')
    plt.ylabel('Jets')
    plt.title('Region ' + str(i + 1))
    lg = plt.legend(loc=1, prop={'size': 18})

plt.tight_layout(pad=2.0)
plt.savefig(figpath + "_cregions_abs_jet_mass.pdf", bbox_inches='tight')
plt.show()


x = 4

binss = np.array([0])
binss = np.append(binss, np.arange(0.02, 0.175, 0.00317))
binss = np.append(binss, 0.225)
binss.shape
binss
bin_range = 5
zmeans = []
zstds = []

for i in range(3):
    zscores = (np.histogram(np.array(real_masses)[cregions[i]], binss)[0] - np.histogram(np.array(gen_masses)[cregions[i]], binss)[0]) / np.histogram(np.array(real_masses)[cregions[i]], binss)[0]
    zmeans.append(np.mean(zscores.reshape(-1, bin_range), axis=1))
    zstds.append(np.std(zscores.reshape(-1, bin_range), axis=1))

xbins = np.mean(binss[1:].reshape(-1, 5), axis=1)

colours = ['red', 'blue', 'green']

fig = plt.figure()
# plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)

for i in range(3):
    plt.errorbar(xbins + (i * 0.0015), zmeans[i], yerr=zstds[i], fmt='x', label='Region ' + str(i), color=colours[i])

plt.ylabel('Mass Difference (Real-Generated)/Real', x = 0.5)
plt.xlabel('Bins')
plt.xlim([0, 0.22])
plt.ylim([-1, 1])
lg = plt.legend(loc=0, prop={'size': 18})
plt.tight_layout(pad=2.0)
plt.savefig(figpath + "_binned_mass_comp.pdf", bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(24, 8))
# plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.errorbar(xbins + (i * 0.0015), zmeans[i], yerr=zstds[i], fmt='x', color='black')
    plt.ylabel('Mass Difference (Real-Generated)/Real', x = 0.5)
    plt.xlabel('Bins')
    plt.xlim([0, 0.22])
    plt.ylim([-1, 1])
    plt.title('Region ' + str(i + 1))

# lg = plt.legend(loc=0, prop={'size': 18})
plt.tight_layout(pad=2.0)
plt.savefig(figpath + "_binned_mass_comp_2.pdf", bbox_inches='tight')
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(30, 12), gridspec_kw = {'hspace': 0, 'height_ratios': [4, 1]})
# plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)

for i in range(3):
    axs[0, i].ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    _ = axs[0, i].hist(np.array(real_masses)[cregions[i]], bins=binsm, histtype='step', label='Real', color='red')
    _ = axs[0, i].hist(np.array(gen_masses)[cregions[i]], bins=binsm, histtype='step', label='Generated', color='blue')
    # axs[0, i].set_xlabel('Jet $m/p_{T}$')
    axs[0, i].set_ylabel('Jets')
    axs[0, i].legend(loc=1, prop={'size': 18})
    axs[0, i].set_title('Region ' + str(i + 1))

for i in range(3):
    axs[1, i].errorbar(xbins + (i * 0.0015), zmeans[i], yerr=zstds[i], fmt='x', color='black')
    axs[1, i].set_ylabel('z-score', x = 0.5)
    axs[1, i].set_xlabel('Jet $m/p_{T}$')
    axs[1, i].set_xlim([0, 0.22])
    axs[1, i].set_ylim([-1, 1])
    axs[1, i].grid(which='both', axis='y')
    # axs[1, i].title('Region ' + str(i + 1))

# lg = plt.legend(loc=0, prop={'size': 18})
plt.tight_layout(pad=2.0)
plt.savefig(figpath + "_binned_mass_comp_3.pdf", bbox_inches='tight')
plt.show()


len(labels)
colours = ['gold', 'orange', 'tomato']

plt.figure()
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_, bins, patches = plt.hist(abs_labels[:num_samples], bins=120)
bregions = [np.sum(bins < regions[0]), np.sum(bins < regions[1])]
for i in range(bregions[0]):
    patches[i].set_facecolor(colours[0])
for i in range(bregions[0], bregions[1]):
    patches[i].set_facecolor(colours[1])
for i in range(bregions[1], len(bins) - 1):
    patches[i].set_facecolor(colours[2])

from matplotlib.patches import Rectangle
handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colours]
labels = ["Region " + str(i) for i in range(1, 4)]
plt.legend(handles, labels)
plt.xlim([0, 2500])
plt.ylabel('# jets', x = 0.5)
plt.xlabel('Jet $p_T$ (MeV)')
plt.savefig(figpath + "_jet_pt_regions.pdf", bbox_inches='tight')
plt.show()


N = 100000
gen_out_rn_efp_format = np.concatenate((np.expand_dims(gen_out_rn[:, :, 2], 2), gen_out_rn[:, :, :2], np.zeros((gen_out_rn.shape[0], gen_out_rn.shape[1], 1))), axis=2)
X_efp_format = np.concatenate((np.expand_dims(X_rn[:N, :, 2], 2), X_rn[:N, :, :2], np.zeros((N, 30, 1))), axis=2)

cgen_efps = []
cX_efps = []

efpset = ef.EFPSet(('n==', 4), ('d==', 4), ('p==', 1), measure='hadr', beta=1, normed=None, coords='ptyphim')
for i in range(3):
    cgen_efps.append(efpset.batch_compute(gen_out_rn_efp_format[cregions[i]]))
    cX_efps.append(efpset.batch_compute(X_efp_format[cregions[i]]))

cgen_efps[2].shape
cX_efps[2].shape

fig = plt.figure(figsize=(30, 18))
bins0 = np.arange(0, 0.0013, step=0.000013)
bins1 = np.arange(0, 0.0004, step=0.000004)

bins = [bins0, bins1, bins1, bins1, bins1]

for j in range(3):
    for i in range(5):
        fig.add_subplot(3, 5, j * 5 + i + 1)
        plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
        plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
        _ = plt.hist(cX_efps[j][:, i], bins[i], histtype='step', label='Real', color='red')
        _ = plt.hist(cgen_efps[j][:, i], bins[i], histtype='step', label='Generated', color='blue')
        plt.xlabel('EFP ' + str(i + 1), x = 0.7)
        if i == 0:
            plt.ylabel('Region ' + str(j + 1) + ' \t \t Jets')
        else:
            plt.ylabel('Jets')
        lg = plt.legend(loc=1, prop={'size': 18})

plt.tight_layout(pad=0.5)
plt.savefig(figpath + "_cregions_efps.pdf", bbox_inches='tight')
plt.show()


gen_out_rn_efp.shape
X_efp.shape
