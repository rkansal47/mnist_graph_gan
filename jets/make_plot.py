import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
from jets_dataset import JetsDataset
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
import mplhep as hep
from skhep.math.vectors import LorentzVector
from tqdm import tqdm
from scipy.stats import wasserstein_distance
import energyflow as ef
import energyflow.utils as ut
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = 39
epoch = 3630
name = str(model) + '_' + str(epoch)
figpath = "figs/" + str(model) + '/' + name

w1m = np.loadtxt('./losses/39/w1_10000m.txt')

len(w1m)
w1std = np.loadtxt('./losses/7/w1_100std.txt')
w1m[int(795/5)]
w1std[int(795/5)]

w1m
(np.argsort(w1m[:, 0])[:20] * 5, np.sort(w1m[:, 0])[:20])

(np.argsort(w1m[:, 1])[:20] * 5, np.sort(w1m[:, 1])[:20])

np.argsort(w1m[:, 2])[:20] * 5


np.argsort(np.linalg.norm(w1m[:, :2], axis=1))[:20] * 5


np.argsort(np.linalg.norm(w1m[:, :3], axis=1))[:20] * 5

# realw1m = [0.00584264, 0.00556786, 0.0014096]
# realw1std = [0.00214083, 0.00204827, 0.00051136]

batch_size = 128

normal_dist = Normal(torch.tensor(0.).to(device), torch.tensor(0.2).to(device))

dir = './'
# dir = '/graphganvol/mnist_graph_gan/jets/'

args = utils.objectview({'dataset_path': dir + 'datasets/', 'num_hits': 30, 'coords': 'polarrel', 'latent_node_size': 32, 'clabels': 0, 'jets': 't', 'norm': 1, 'mask': False, 'mask_manual': False, 'real_only': False})
X = JetsDataset(args)

labels = X[:][1]
# X_loaded = DataLoader(X, shuffle=True, batch_size=32, pin_memory=True)
X = X[:][0]
N = len(X)

rng = np.random.default_rng()

num_samples = 100000

G = torch.load('./models/' + str(model) + '/G_' + str(epoch) + '.pt', map_location=device)

if args.clabels:
    gen_out = utils.gen(args, G, dist=normal_dist, num_samples=batch_size, labels=labels[:128]).cpu().detach().numpy()
    for i in tqdm(range(int(num_samples / batch_size))):
        gen_out = np.concatenate((gen_out, utils.gen(args, G, dist=normal_dist, num_samples=batch_size, labels=labels[128 * (i + 1):128 * (i + 2)]).cpu().detach().numpy()), 0)
    gen_out = gen_out[:num_samples]
else:
    gen_out = utils.gen(args, G, dist=normal_dist, num_samples=batch_size).cpu().detach().numpy()
    for i in tqdm(range(int(num_samples / batch_size))):
        gen_out = np.concatenate((gen_out, utils.gen(args, G, dist=normal_dist, num_samples=batch_size).cpu().detach().numpy()), 0)
    gen_out = gen_out[:num_samples]

model
name
np.save('./models/' + str(model) + '/' + name + "_gen_out", gen_out)
# gen_out = np.load('./models/' + str(model) + '/' + name + "_gen_out.npy")

# gen_out /= maxepp

# maxepp = [1.4130558967590332, 0.520724892616272, 0.8537549376487732]
Xplot = X[:num_samples, :, :].cpu().detach().numpy()
Xplot = Xplot / args.norm
Xplot[:, :, 2] += 0.5
Xplot *= args.maxepp

gen_out = gen_out / args.norm
gen_out[:, :, 2] += 0.5
gen_out *= args.maxepp

print(Xplot.shape)
print(gen_out.shape)

print(Xplot[0][:10])
print(gen_out[0][:10])

len(gen_out[gen_out[:, :, 2] < 0])

for i in range(num_samples):
    for j in range(30):
        if gen_out[i][j][2] < 0:
            gen_out[i][j][2] = 0

len(gen_out[gen_out[:, :, 2] < 0])
# # num_samples = 100000
#
# plt.hist(Xplot[:, :, 2].reshape(-1), np.arange())
#
# plt.hist(Xplot[:, :, 2].reshape(-1), np.arange(0, 0.0002, 0.000002), histtype='step', label='Real', color='red', log=True)
#
# np.unique(Xplot[:, :, 2])
#
# # pT < 9e-5 means 0 for g100
# # pT < 1.3e-4 for g30; 10^4 zero-padded
# # ggp
#
# plt.hist(gen_out[:, :, 2].reshape(-1), np.arange(0, 0.0002, 0.000002), histtype='step', label='Real', color='red', log=True)
#
# len(gen_out[gen_out[:, :, 2] < 0.0001])
#
# len(Xplot[Xplot[:, :, 2] < 0.00012])
#
# plt.hist(gen_out[gen_out[:, :, 2] < 0.0001][:, 0], bins[0], histtype='step', label='Real', color='red')

real_masses = []
real_pt = []
gen_masses = []
gen_pt = []

for i in tqdm(range(num_samples)):
    jetv = LorentzVector()

    for part in Xplot[i]:
        vec = LorentzVector()
        vec.setptetaphim(part[2], part[0], part[1], 0)
        jetv += vec

    real_masses.append(jetv.mass)
    real_pt.append(jetv.pt)

for i in tqdm(range(num_samples)):
    jetv = LorentzVector()

    for part in gen_out[i]:
        vec = LorentzVector()
        if part[2] >= 0:
            vec.setptetaphim(part[2], part[0], part[1], 0)
        else:
            vec.setptetaphim(0, part[0], part[1], 0)
        jetv += vec

    gen_masses.append(jetv.mass)
    gen_pt.append(jetv.pt)

len(real_masses)
len(gen_masses)

binsm = np.arange(0, 0.3, 0.003)
binspt = np.arange(0.5, 1.2, 0.007)

fig = plt.figure(figsize=(16, 8))

fig.add_subplot(1, 2, 1)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
# plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_masses, bins=binsm, histtype='step', label='Real', color='red')
_ = plt.hist(gen_masses, bins=binsm, histtype='step', label='Generated', color='blue')
plt.xlabel('Jet Relative Mass')
plt.ylabel('Jets')
plt.legend(loc=1, prop={'size': 18})

fig.add_subplot(1, 2, 2)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_pt, bins=binspt, histtype='step', label='Real', color='red')
_ = plt.hist(gen_pt, bins=binspt, histtype='step', label='Generated', color='blue')
plt.xlabel('Jet Relative Pt')
plt.ylabel('Jets')
plt.legend(loc=1, prop={'size': 18})

plt.tight_layout(pad=2.0)
plt.savefig(figpath + "_100000_jets_rel_mass_pt_cut.pdf", bbox_inches='tight')
plt.show()

plabels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T^{rel}$']
# sf = [3, 2, 3]
# rnd = [0, 1, 0]
# castings = [int, float, int]

if args.jets == 'g':
    bins = [np.arange(-0.3, 0.3, 0.005), np.arange(-0.3, 0.3, 0.005), np.arange(0, 0.2, 0.002)]
elif args.jets == 't':
    bins = [np.arange(-0.5, 0.5, 0.005), np.arange(-0.5, 0.5, 0.005), np.arange(0, 0.2, 0.002)]

fig = plt.figure(figsize=(30, 8))

for i in range(3):
    fig.add_subplot(1, 4, i + 1)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(Xplot[:, :, i].reshape(-1), bins[i], histtype='step', label='Real', color='red')
    _ = plt.hist(gen_out[:, :, i].reshape(-1), bins[i], histtype='step', label='Generated', color='blue')
    plt.xlabel('Particle ' + plabels[i])
    plt.ylabel('Particles')
    # plt.title('$W_1$ = (' + str(castings[i](round(w1m[idx][i] * int(10 ** sf[i]), rnd[i]))) + ' ± ' + str(castings[i](round(w1std[idx][i] * int(10 ** sf[i]), rnd[i]))) + ') $\\times 10^{-' + str(sf[i]) + '}$')
    # title = '$W_1$ = (' + str(castings[i](round(w1m[idx][i] * int(10 ** sf[i]), rnd[i]))) + ' ± ' + str(castings[i](round(w1std[idx][i] * int(10 ** sf[i]), rnd[i]))) + ') $\\times 10^{-' + str(sf[i]) + '}$'
    lg = plt.legend(loc=1, prop={'size': 18})
    # lg.set_title(title)
    # lg.get_title().set_fontsize(13)

binsm = np.arange(0, 0.225, 0.00225)

fig.add_subplot(1, 4, 4)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
# plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_masses, bins=binsm, histtype='step', label='Real', color='red')
_ = plt.hist(gen_masses, bins=binsm, histtype='step', label='Generated', color='blue')
plt.xlabel('Jet $m/p_{T}$')
plt.ylabel('Jets')
plt.legend(loc=1, prop={'size': 18})

plt.tight_layout(pad=2.0)
plt.savefig(figpath + ".pdf", bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(30, 10))

for i in range(3):
    fig.add_subplot(1, 3, i + 1)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(Xplot[:, :, i].reshape(-1), bins[i], histtype='step', label='Real', color='red')
    _ = plt.hist(gen_out[:, :, i].reshape(-1), bins[i], histtype='step', label='Generated', color='blue')
    plt.xlabel('Particle ' + plabels[i])
    plt.ylabel('Particles')
    lg = plt.legend(loc=1, prop={'size': 18})

plt.tight_layout(pad=2.0)
plt.savefig(figpath + "_particle_level.pdf", bbox_inches='tight')
plt.show()


rng.choice(100000, size=10)

np.array(gen_masses)[rng.choice(100000, size=10)]

gen_masses = np.array(gen_masses)
real_masses = np.array(real_masses)


num_samples = np.array([100, 1000, 10000])
num_batches = np.array(100000 / num_samples, dtype=int)

real_means = []
real_stds = []
gen_means = []
gen_stds = []

N = 100000

for k in range(len(num_samples)):
    print("Num Samples: " + str(num_samples[k]))
    gen_w1s = []
    real_w1s = []
    for j in tqdm(range(num_batches[k])):
        gen_sample = gen_masses[rng.choice(N, size=num_samples[k])]
        sample = real_masses[rng.choice(N, size=num_samples[k])]

        gen_w1s.append(wasserstein_distance(gen_sample, sample))

        # sample1 = real_masses[rng.choice(N, size=num_samples[k])]
        # sample2 = real_masses[rng.choice(N, size=num_samples[k])]
        #
        # real_w1s.append(wasserstein_distance(sample1, sample2))

    # real_means.append(np.mean(np.array(real_w1s), axis=0))
    # real_stds.append(np.std(np.array(real_w1s), axis=0))
    gen_means.append(np.mean(np.array(gen_w1s), axis=0))
    gen_stds.append(np.std(np.array(gen_w1s), axis=0))


real_means
real_stds
gen_means
gen_stds


# Get all prime EFPs with n=4, d=4
# Specify EFPs set
efpset = ef.EFPSet(('n==', 4), ('d==', 4), ('p==', 1), measure='hadr', beta=1, normed=None, coords='ptyphim')

N = 100000
gen_out_efp_format = np.concatenate((np.expand_dims(gen_out[:, :, 2], 2), gen_out[:, :, :2], np.zeros((gen_out.shape[0], gen_out.shape[1], 1))), axis=2)
X_efp_format = np.concatenate((np.expand_dims(Xplot[:, :, 2], 2), Xplot[:, :, :2], np.zeros((N, 30, 1))), axis=2)

gen_out_efp = efpset.batch_compute(gen_out_efp_format)
X_efp = efpset.batch_compute(X_efp_format)

gen_out_efp.shape
X_efp.shape

fig = plt.figure(figsize=(20, 12))

if args.jets == 'g':
    bins0 = np.arange(0, 0.0013, step=0.000013)
    bins1 = np.arange(0, 0.0004, step=0.000004)
    bins = [bins0, bins1, bins1, bins1, bins1]
elif args.jets == 't':
    binranges = [0.0045, 0.0035, 0.004, 0.002, 0.003]
    bins = [np.arange(0, binr, step=binr / 100) for binr in binranges]

for i in range(5):
    fig.add_subplot(2, 3, i + 1)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(X_efp[:, i], bins[i], histtype='step', label='Real', color='red')
    _ = plt.hist(gen_out_efp[:, i], bins[i], histtype='step', label='Generated', color='blue')
    plt.xlabel('EFP ' + str(i + 1), x = 0.7)
    plt.ylabel('Jets')
    lg = plt.legend(loc=1, prop={'size': 18})

plt.tight_layout(pad=0.5)
plt.savefig(figpath + "_100000_jets_efp.pdf", bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(20, 12))

for i in range(5):
    fig.add_subplot(2, 3, i + 2)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(X_efp[:, i], bins[i], histtype='step', label='Real', color='red')
    _ = plt.hist(gen_out_efp[:, i], bins[i], histtype='step', label='Generated', color='blue')
    plt.xlabel('EFP ' + str(i + 1), x = 0.7)
    plt.ylabel('Jets')
    lg = plt.legend(loc=1, prop={'size': 18})

fig.add_subplot(2, 3, 1)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_masses, bins=binsm, histtype='step', label='Real', color='red')
_ = plt.hist(gen_masses, bins=binsm, histtype='step', label='Generated', color='blue')
plt.xlabel('Jet $m/p_{T}$')
plt.ylabel('Jets')
plt.legend(loc=1, prop={'size': 18})

plt.tight_layout(pad=0.5)
plt.savefig(figpath + "_100000_jets_efp_mass.pdf", bbox_inches='tight')
plt.show()

efpset2 = ef.EFPSet(('n==', 2), ('d==', 2), ('p==', 1), measure='hadr', beta=1, normed=None, coords='ptyphim')

gen_out_efp2 = efpset2.batch_compute(gen_out_efp_format)
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
_ = plt.hist(gen_out_efp2[:, 0], bins, histtype='step', label='Generated', color='blue')
plt.xlabel('EFP (d=2, n=2)')
plt.ylabel('Jets')
lg = plt.legend(loc=1, prop={'size': 18})

fig.add_subplot(1, 4, 2)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(np.sqrt(X_efp2[:, 0] / 2), binsm, histtype='step', label='Real', color='red')
_ = plt.hist(np.sqrt(gen_out_efp2[:, 0] / 2), binsm, histtype='step', label='Generated', color='blue')
plt.xlabel('$\sqrt{EFP/2}$')
plt.ylabel('Jets')
lg = plt.legend(loc=1, prop={'size': 18})

fig.add_subplot(1, 4, 4)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(np.sqrt(X_efp2[:, 0] / 2), binsm, histtype='step', label='Real EFP', color='red')
_ = plt.hist(np.sqrt(gen_out_efp2[:, 0] / 2), binsm, histtype='step', label='Generated EFP', color='blue')
_ = plt.hist(real_masses, bins=binsm, histtype='step', label='Real Mass', color='orange')
_ = plt.hist(gen_masses, bins=binsm, histtype='step', label='Generated Mass', color='green')
# plt.xlabel('$\sqrt{EFP/2}$')
plt.ylabel('Jets')
lg = plt.legend(loc=1, prop={'size': 18})
plt.savefig(figpath + "_100000_jets_efp_mass_comp.pdf", bbox_inches='tight')
plt.show()


plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(np.sqrt(X_efp2[:, 0] / 2), binsm, histtype='step', label='Real', color='red')
_ = plt.hist(np.sqrt(gen_out_efp2[:, 0] / 2), binsm, histtype='step', label='Generated', color='blue')
plt.xlabel('Jet $m/p_T$')
plt.ylabel('Jets')
lg = plt.legend(loc=1, prop={'size': 18})
plt.savefig(figpath + "_100000_jets_efp_mass.pdf", bbox_inches='tight')
plt.show()

real_efp_masses = np.sqrt(X_efp2[:, 0] / 2)
gen_efp_masses = np.sqrt(gen_out_efp2[:, 0] / 2)

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


num_pixels = 100
Njets = 100000
img_width = 0.8

average_real_jet_image = np.zeros((num_pixels, num_pixels, 1))
for i in range(Njets):
    average_real_jet_image += ut.pixelate(X_efp_format[i, :, :], npix=num_pixels, img_width=img_width, nb_chan=1, norm=False, charged_counts_only=False)
average_real_jet_image /= Njets

average_gen_jet_image = np.zeros((num_pixels, num_pixels, 1))
for i in range(Njets):
    average_gen_jet_image += ut.pixelate(gen_out_efp_format[i, :, :], npix=num_pixels, img_width=img_width, nb_chan=1, norm=False, charged_counts_only=False)
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


Nangle = 1000

real_delta_etaphi = []
gen_delta_etaphi = []

for i in range(Nangle):
    jet = Xplot[i]
    for j in range(30):
        for k in range(j + 1, 30):
            real_delta_etaphi.append([np.abs(jet[j][0] - jet[k][0]), np.abs(jet[j][1] - jet[k][1])])


for i in range(Nangle):
    jet = gen_out[i]
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

    for part in gen_out[i]:
        vec = LorentzVector()
        vec.setptetaphim(part[2], part[0], part[1], 0)
        jetv += vec

    dmass = jetv.mass

    efp = efpset2.compute(gen_out_efp_format[i])[0]

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

len(Xplot[cregions[0]])
len(Xplot[cregions[1]])
len(Xplot[cregions[2]])

bins = [np.arange(-0.3, 0.3, 0.005), np.arange(-0.3, 0.3, 0.005), np.arange(0, 0.2, 0.002)]
binsm = np.arange(0, 0.225, 0.0045)

fig, axs = plt.subplots(3, 4, figsize=(30, 20))

for j in range(3):
    for i in range(3):
        axs[j, i].ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
        _ = axs[j, i].hist(Xplot[cregions[j]][:, :, i].reshape(-1), bins[i], histtype='step', label='Real', color='red')
        _ = axs[j, i].hist(gen_out[cregions[j]][:, :, i].reshape(-1), bins[i], histtype='step', label='Generated', color='blue')
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
    plt.xlabel('Jet mass')
    plt.ylabel('Jets')
    lg = plt.legend(loc=1, prop={'size': 18})

plt.tight_layout(pad=2.0)
# plt.savefig(figpath + "_particle_level.pdf", bbox_inches='tight')
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
gen_out_efp_format = np.concatenate((np.expand_dims(gen_out[:, :, 2], 2), gen_out[:, :, :2], np.zeros((gen_out.shape[0], gen_out.shape[1], 1))), axis=2)
X_efp_format = np.concatenate((np.expand_dims(Xplot[:N, :, 2], 2), Xplot[:N, :, :2], np.zeros((N, 30, 1))), axis=2)

cgen_efps = []
cX_efps = []

efpset = ef.EFPSet(('n==', 4), ('d==', 4), ('p==', 1), measure='hadr', beta=1, normed=None, coords='ptyphim')
for i in range(3):
    cgen_efps.append(efpset.batch_compute(gen_out_efp_format[cregions[i]]))
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


gen_out_efp.shape
X_efp.shape
