import torch
import numpy as np
import matplotlib.pyplot as plt

import mplhep as hep
plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)

jet_types = ['g', 't', 'z', 'w', 'q']

datasets = {}
masks = {}

for t in jet_types:
    datasets[t] = torch.load('datasets/all_' + t + '_jets_100p_polarrel.pt')
    masks[t] = torch.gt(torch.norm(datasets[t], dim=2), 0).float()
    datasets[t] = torch.cat((datasets[t].float(), (masks[t] - 0.5).unsqueeze(2)), dim=2)

for t in jet_types:
    torch.save(datasets[t], './datasets/all_' + t + '_jets_100p_polarrel_mask.pt')

for t in jet_types:
    print("{} {} jets total, ave {:.1f} non-zero padded particles per jet".format(len(datasets[t]), t, torch.sum(masks[t]) / len(datasets[t])))

num_ps = {}

for t in jet_types:
    num_ps[t] = torch.sum(masks[t], dim=1)

num_ps

colors = ['red', 'brown', 'orange', 'green', 'blue']
plt.figure(figsize=(10, 8))
i = 0
for t in jet_types:
    plt.hist(num_ps[t].detach().numpy(), bins=np.linspace(0, 100, 101), histtype='step', label=t, color=colors[i], density=True)
    i += 1
plt.xlabel("# particles")
plt.ylabel("fraction of jets with N particles")
plt.legend(loc=2)
plt.savefig("figs/num_particles.pdf", bbox_inches='tight')
plt.show()


datasets30 = {}
nps30 = {}

for t in jet_types:
    datasets30[t] = datasets[t][:, :30, :]
    nps30[t] = torch.sum(datasets30[t][:, :, 3] + 0.5, dim=1)

parts = {t: datasets30[t][(datasets30[t][:, :, 3] + 0.5).bool()] for t in jet_types}

30 * len(nps30['g']) - torch.sum(nps30['g'])

30 * len(nps30['t']) - torch.sum(nps30['t'])

datasets30['t'][:, :, 3] + 0.5

len(datasets30['t'][(datasets30['t'][:, :, 3] + 0.5).bool()])
torch.mean(nps30['t']) * len(nps30['t'])

plabels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T^{rel}$']
bins = [np.linspace(-0.5, 0.5, 201), np.linspace(-0.5, 0.5, 201), np.linspace(0, 0.2, 101)]

fig = plt.figure(figsize=(22, 8))

for i in range(3):
    fig.add_subplot(1, 3, i + 1)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(datasets30['t'][:, :, i].detach().numpy().reshape(-1), bins[i], label='With 0P', histtype='step', color='blue')
    _ = plt.hist(parts['t'][:, i].detach().numpy().reshape(-1), bins[i], label='Without 0P', histtype='step', color='red')
    plt.xlabel('Particle ' + plabels[i])
    plt.ylabel('Particles')
    plt.legend(prop={'size': 18})

plt.tight_layout(pad=2.0)
plt.savefig("figs/all_t30_0pcomp.pdf", bbox_inches='tight')
plt.show()


bins = [np.linspace(-0.3, 0.3, 201), np.linspace(-0.3, 0.3, 201), np.linspace(0, 0.2, 101)]
fig = plt.figure(figsize=(22, 8))

for i in range(3):
    fig.add_subplot(1, 3, i + 1)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(datasets30['g'][:, :, i].detach().numpy().reshape(-1), bins[i], label='With 0P', histtype='step', color='blue')
    _ = plt.hist(parts['g'][:, i].detach().numpy().reshape(-1), bins[i], label='Without 0P', histtype='step', color='red')
    plt.xlabel('Particle ' + plabels[i])
    plt.ylabel('Particles')

plt.tight_layout(pad=2.0)
plt.savefig("figs/all_g30_0pcomp.pdf", bbox_inches='tight')
plt.show()



plt.hist(datasets30['t'][:, :, 0].detach().numpy().reshape(-1), bins=np.arange(0, 0.2, 0.002), histtype='step')

plt.hist(datasets30['t'][:, :, 2].detach().numpy().reshape(-1), bins=np.arange(0, 0.2, 0.002), histtype='step')

dataset30 = torch.load('datasets_backup/all_g_jets_30p_polarrel.pt')

datasetg = datasets['g'][:, :30, :3]

torch.sum(masks['g'][:122922, :30])

dataset30


len(dataset30)


mask30 = torch.gt(torch.norm(dataset30, dim=2), 0).float()
torch.sum(mask30)
