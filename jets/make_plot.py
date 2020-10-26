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
plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epoch = 790

G = torch.load('./models/7_batch_size_128_coords_polarrel/G_' + str(epoch) + '.pt', map_location=device)
w1m = np.loadtxt('./losses/7/w1_100m.txt')
w1std = np.loadtxt('./losses/7/w1_100std.txt')

realw1m = [0.00584264, 0.00556786, 0.0014096]
realw1std = [0.00214083, 0.00204827, 0.00051136]

batch_size = 128

normal_dist = Normal(torch.tensor(0.).to(device), torch.tensor(0.2).to(device))

dir = './'
# dir = '/graphganvol/mnist_graph_gan/jets/'

args = {'dataset_path': dir + 'datasets/', 'num_hits': 30, 'coords': 'polarrel', 'latent_node_size': 32}
X = JetsDataset(utils.objectview(args))

N = len(X)

rng = np.random.default_rng()

num_samples = 100000

# gen_out = utils.gen(utils.objectview(args), G, dist=normal_dist, num_samples=batch_size).cpu().detach().numpy()
# for i in range(int(num_samples / batch_size)):
#     gen_out = np.concatenate((gen_out, utils.gen(utils.objectview(args), G, dist=normal_dist, num_samples=batch_size).cpu().detach().numpy()), 0)
# gen_out = gen_out[:num_samples]
#
# np.save("790_gen_out", gen_out)
gen_out = np.load("790_gen_out.npy")

# fig.suptitle("Particle Feature Distributions")

labels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T^{rel}$']

maxepp = [1.4130558967590332, 0.520724892616272, 0.8537549376487732]
Xplot = X[:num_samples, :, :].cpu().detach().numpy() * maxepp
gen_out *= maxepp

print(Xplot.shape)
print(gen_out.shape)

print(Xplot[0][:10])
print(gen_out[0][:10])

num_samples = 100000

real_masses = []
real_pt = []
gen_masses = []
gen_pt = []
real_jets = []
gen_jets = []

for i in range(num_samples):
    jetv = LorentzVector()

    for part in Xplot[i]:
        vec = LorentzVector()
        vec.setptetaphim(part[2], part[0], part[1], 0)
        # rel_mass += vec.mass / (vec.pt + 1e-12)
        jetv += vec

    real_masses.append(jetv.mass)
    real_pt.append(jetv.pt)
    real_jets.append([jetv.pt, jetv.eta, jetv.phi])

for i in range(num_samples):
    jetv = LorentzVector()

    rel_mass = 0

    for part in gen_out[i]:
        vec = LorentzVector()
        vec.setptetaphim(part[2], part[0], part[1], 0)
        # rel_mass += vec.mass / (vec.pt + 1e-12)
        jetv += vec

    gen_masses.append(jetv.mass)
    gen_pt.append(jetv.pt)
    gen_jets.append([jetv.pt, jetv.eta, jetv.phi])

len(real_masses)
len(gen_masses)

plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)

# bins = np.arange(-2e-7, 2e-7, step=0.1e-7)
binsm = np.arange(0, 0.3, 0.003)
binspt = np.arange(0.5, 1.2, 0.007)

# bins

fig = plt.figure(figsize=(16, 8))

fig.add_subplot(1, 2, 1)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_masses, bins=binsm, histtype='step', label='real', color='blue')
_ = plt.hist(gen_masses, bins=binsm, histtype='step', label='generated', color='red')
plt.xlabel('Jet Relative Mass')
plt.ylabel('Jets')
plt.legend(loc=1, prop={'size': 18})

fig.add_subplot(1, 2, 2)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_pt, bins=binspt, histtype='step', label='real', color='blue')
_ = plt.hist(gen_pt, bins=binspt, histtype='step', label='generated', color='red')
plt.xlabel('Jet Relative Pt')
plt.ylabel('Jets')
plt.legend(loc=1, prop={'size': 18})

plt.tight_layout(2.0)
plt.savefig("7_" + str(epoch) + "_100000_jets_rel_mass_pt.pdf", bbox_inches='tight')
plt.show()


plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)

sf = [3, 2, 3]
rnd = [0, 1, 0]
castings = [int, float, int]

idx = int(epoch / 5 - 1)

bins = [np.arange(-0.3, 0.3, 0.005), np.arange(-0.3, 0.3, 0.005), np.arange(0, 0.2, 0.002)]

fig = plt.figure(figsize=(30, 8))

for i in range(3):
    fig.add_subplot(1, 4, i + 1)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(Xplot[:, :, i].reshape(-1), bins[i], histtype='step', label='Real', color='red')
    _ = plt.hist(gen_out[:, :, i].reshape(-1), bins[i], histtype='step', label='Generated', color='blue')
    plt.xlabel('Particle ' + labels[i])
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

# name = args.name + "/" + str(epoch)

plt.tight_layout(2.0)
plt.savefig("7_" + str(epoch) + ".pdf", bbox_inches='tight')
plt.show()


gen_masses

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

        sample1 = real_masses[rng.choice(N, size=num_samples[k])]
        sample2 = real_masses[rng.choice(N, size=num_samples[k])]

        real_w1s.append(wasserstein_distance(sample1, sample2))

    real_means.append(np.mean(np.array(real_w1s), axis=0))
    real_stds.append(np.std(np.array(real_w1s), axis=0))
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
plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)

bins0 = np.arange(0, 0.0013, step=0.000013)
bins1 = np.arange(0, 0.0004, step=0.000004)

bins = [bins0, bins1, bins1, bins1, bins1]

for i in range(5):
    fig.add_subplot(2, 3, i + 1)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(X_efp[:, i], bins[i], histtype='step', label='Real', color='red')
    _ = plt.hist(gen_out_efp[:, i], bins[i], histtype='step', label='Generated', color='blue')
    plt.xlabel('EFP ' + str(i + 1), x = 0.7)
    plt.ylabel('Jets')
    # plt.title('$W_1$ = (' + str(castings[i](round(w1m[idx][i] * int(10 ** sf[i]), rnd[i]))) + ' ± ' + str(castings[i](round(w1std[idx][i] * int(10 ** sf[i]), rnd[i]))) + ') $\\times 10^{-' + str(sf[i]) + '}$')
    # title = '$W_1$ = (' + str(castings[i](round(w1m[idx][i] * int(10 ** sf[i]), rnd[i]))) + ' ± ' + str(castings[i](round(w1std[idx][i] * int(10 ** sf[i]), rnd[i]))) + ') $\\times 10^{-' + str(sf[i]) + '}$'
    lg = plt.legend(loc=1, prop={'size': 18})
    # lg.set_title(title)
    # lg.get_title().set_fontsize(13)

plt.tight_layout(0.5)
plt.savefig("7_" + str(epoch) + "_100000_jets_efp_fix.pdf", bbox_inches='tight')
plt.show()


efpset2 = ef.EFPSet(('n==', 2), ('d==', 2), ('p==', 1), measure='hadr', beta=1, normed=None, coords='ptyphim')

gen_out_efp2 = efpset2.batch_compute(gen_out_efp_format)
X_efp2 = efpset2.batch_compute(X_efp_format)

X_efp2.shape


fig = plt.figure(figsize=(30, 10))

binsm = np.arange(0, 0.25, 0.0025)
fig.add_subplot(1, 3, 3)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_masses, bins=binsm, histtype='step', label='Real', color='red')
_ = plt.hist(gen_masses, bins=binsm, histtype='step', label='Generated', color='blue')
plt.xlabel('Jet Relative Mass')
plt.ylabel('Jets')
plt.legend(loc=1, prop={'size': 18})

bins = np.arange(0, 0.07, 0.0007)

fig.add_subplot(1, 3, 1)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(X_efp2[:, 0], bins, histtype='step', label='Real', color='red')
_ = plt.hist(gen_out_efp2[:, 0], bins, histtype='step', label='Generated', color='blue')
plt.xlabel('EFP (d=2, n=2)')
plt.ylabel('Jets')
lg = plt.legend(loc=1, prop={'size': 18})

fig.add_subplot(1, 3, 2)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(np.sqrt(X_efp2[:, 0] / 2), binsm, histtype='step', label='Real', color='red')
_ = plt.hist(np.sqrt(gen_out_efp2[:, 0] / 2), binsm, histtype='step', label='Generated', color='blue')
plt.xlabel('$\sqrt{EFP/2}$')
plt.ylabel('Jets')
lg = plt.legend(loc=1, prop={'size': 18})

plt.tight_layout(0.5)
plt.savefig("7_" + str(epoch) + "_100000_jets_efp_mass_2.pdf", bbox_inches='tight')
plt.show()

plt.figure()
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(np.sqrt(X_efp2[:, 0] / 2), binsm, histtype='step', label='Real EFP', color='red')
_ = plt.hist(np.sqrt(gen_out_efp2[:, 0] / 2), binsm, histtype='step', label='Generated EFP', color='blue')
_ = plt.hist(real_masses, bins=binsm, histtype='step', label='Real Mass', color='orange')
_ = plt.hist(gen_masses, bins=binsm, histtype='step', label='Generated Mass', color='green')
# plt.xlabel('$\sqrt{EFP/2}$')
plt.ylabel('Jets')
lg = plt.legend(loc=1, prop={'size': 18})
plt.savefig("7_" + str(epoch) + "_100000_jets_efp_mass_comp_2.pdf", bbox_inches='tight')
plt.show()


plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(np.sqrt(X_efp2[:, 0] / 2), binsm, histtype='step', label='Real', color='red')
_ = plt.hist(np.sqrt(gen_out_efp2[:, 0] / 2), binsm, histtype='step', label='Generated', color='blue')
plt.xlabel('Jet $m/p_T$')
plt.ylabel('Jets')
lg = plt.legend(loc=1, prop={'size': 18})
plt.savefig("7_" + str(epoch) + "_100000_jets_efp_mass.pdf", bbox_inches='tight')
plt.show()


def average_jet_image_plot(ptyphi_jets, num_pixels, n_jets, jet_image_name):  # ptyphi_jets shape (jets, N-particles, M-features)
    num_particles = ptyphi_jets.shape[1]
    # initialize average image array with zeros
    average_jet_image = np.zeros((num_pixels, num_pixels, 1))
    # Compute average jet image
    for i in range(len(ptyphi_jets)):
        average_jet_image += ut.pixelate(ptyphi_jets[i, :, :], npix=num_pixels, nb_chan=1, norm=False, charged_counts_only=False)
    average_jet_image /= n_jets
    # Plot average jet image
    # Set colormap of your choice
    plt.imshow(average_jet_image[:, :, 0], origin='lower', cmap = 'binary', norm=LogNorm(vmin=0.01))
    plt.colorbar()
    plt.title(jet_image_name, fontsize=18)
    plt.xlabel('$i\phi$')
    plt.ylabel('$i\eta$')
    plt.savefig('7_790_' + jet_image_name + str(num_particles) + 'p.pdf', dpi=250, bbox_inches='tight')
    plt.clf()


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
plt.style.use(hep.style.CMS)
fig = plt.figure(figsize=(20, 10))
fig.add_subplot(1, 2, 1)
plt.imshow(average_real_jet_image[:, :, 0], origin='lower', cmap = 'binary', norm=LogNorm(vmin=1e-7))
plt.colorbar()
plt.title("Average of Real Jets", fontsize=25)
plt.xlabel('$i\phi$')
plt.ylabel('$i\eta$')
# plt.savefig('7_790_real_jets_ave.pdf', dpi=250, bbox_inches='tight')
# plt.show()

fig.add_subplot(1, 2, 2)
plt.imshow(average_gen_jet_image[:, :, 0], origin='lower', cmap='binary', norm=LogNorm(vmin=1e-7))
plt.colorbar()
plt.title("Average of Generated Jets", fontsize=25)
plt.xlabel('$i\phi$')
plt.ylabel('$i\eta$')

plt.savefig('7_790_jets_ave.pdf', dpi=250, bbox_inches='tight')
plt.show()


real_efp_masses = np.sqrt(X_efp2[:, 0] / 2)
gen_efp_masses = np.sqrt(gen_out_efp2[:, 0] / 2)

real_mass_diff = real_masses - real_efp_masses
gen_mass_diff = gen_masses - gen_efp_masses

real_mass_hist_diff = np.histogram(real_masses, binsm)[0] - np.histogram(real_efp_masses, binsm)[0]
gen_mass_hist_diff = np.histogram(gen_masses, binsm)[0] - np.histogram(gen_efp_masses, binsm)[0]

plt.plot(binsm[:-1], real_mass_hist_diff)

bins = np.arange(-7e-4, 7e-4, 7e-6)

fig = plt.figure(figsize=(20, 10))
fig.add_subplot(1, 2, 1)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_mass_diff, bins, histtype='step', label='Real', color='red')
_ = plt.hist(gen_mass_diff, bins, histtype='step', label='Generated', color='blue')
plt.xlabel('Mass Difference (Exact - EFP)', x = 0.5)
plt.ylabel('Jets')
lg = plt.legend(loc=1, prop={'size': 18})

fig.add_subplot(1, 2, 2)
plt.plot(binsm[:-1], real_mass_hist_diff, label='Real', color='red')
plt.plot(binsm[:-1], gen_mass_hist_diff, label='Generated', color='blue')
plt.xlabel('Mass', x = 0.5)
plt.ylabel('# Exact - # EFP Jets')
lg = plt.legend(loc=1, prop={'size': 18})
plt.savefig("7_" + str(epoch) + "_100000_jets_mass_diff.pdf", bbox_inches='tight')
plt.show()




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

len(real_delta_etaphi)
len(gen_delta_etaphi)

max(gen_delta_etaphi[:, 0])
max(gen_delta_etaphi[:, 1])

real_delta_etaphi[:, 0]

plt.hist(real_delta_etaphi[:, 0], np.arange(0, 0.8, 0.008), histtype='step', label='Real', color='red')
plt.hist(gen_delta_etaphi[:, 0], np.arange(0, 0.8, 0.008), histtype='step', label='Real', color='red')

binseta = np.arange(0, 0.6, 0.006)
binsphi = np.arange(0, 0.4, 0.004)

fig = plt.figure(figsize=(20, 8))
fig.add_subplot(1, 2, 1)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_delta_etaphi[:, 0], binseta, histtype='step', label='Real', color='red')
_ = plt.hist(gen_delta_etaphi[:, 0], binseta, histtype='step', label='Generated', color='blue')
plt.xlabel('$\Delta\eta$')
plt.ylabel('#')
lg = plt.legend(loc=1, prop={'size': 18})

fig.add_subplot(1, 2, 2)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_delta_etaphi[:, 1], binseta, histtype='step', label='Real', color='red')
_ = plt.hist(gen_delta_etaphi[:, 1], binseta, histtype='step', label='Generated', color='blue')
plt.xlabel('$\Delta\phi$')
plt.ylabel('#')
lg = plt.legend(loc=1, prop={'size': 18})

# plt.savefig("7_" + str(epoch) + "_10000_jets_delta_eta_phi.pdf", bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(20, 10))
fig.add_subplot(1, 2, 1)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(real_delta_etaphi[:, 0], binseta, histtype='step', label='Real $\Delta \phi$', color='red')
_ = plt.hist(real_delta_etaphi[:, 1], binseta, histtype='step', label='Real $\Delta \eta$', color='orange')
# plt.xlabel('$\$')
plt.ylabel('#')
lg = plt.legend(loc=1, prop={'size': 18})

fig.add_subplot(1, 2, 2)
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
_ = plt.hist(gen_delta_etaphi[:, 0], binseta, histtype='step', label='Generated $\Delta \phi$', color='blue')
_ = plt.hist(gen_delta_etaphi[:, 1], binseta, histtype='step', label='Generated $\Delta \eta$', color='green')
# plt.xlabel('$\$')
plt.ylabel('#')
lg = plt.legend(loc=1, prop={'size': 18})

plt.savefig("7_" + str(epoch) + "_1000_jets_delta_eta_phi_real_gen.pdf", bbox_inches='tight')
plt.show()
