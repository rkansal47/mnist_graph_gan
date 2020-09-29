import torch
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

dataset = np.array(torch.load('datasets/all_g_jets_30p_polarrel.pt'))

N = dataset.size(0)

N

batch_size = 10000

rng = np.random.default_rng()

bins = [np.arange(-1, 1, 0.02), np.arange(-1, 1, 0.02), np.arange(0, 1, 0.01)]

jsds2 = []

for j in range(100):
    print(j)
    jsd = []
    sample1 = dataset[rng.choice(N, size=batch_size, replace=False)]
    sample2 = dataset[rng.choice(N, size=batch_size, replace=False)]
    for i in range(3):
        hist1 = np.histogram(sample1, bins=bins[i], density=True)[0]
        hist2 = np.histogram(sample2, bins=bins[i], density=True)[0]
        jsd.append(jensenshannon(hist1, hist2))
    jsds2.append(jsd)

jsds


jsds2

np.mean(np.array(jsds), axis=0)
np.std(np.array(jsds), axis=0)

np.mean(np.array(jsds2), axis=0)
np.std(np.array(jsds2), axis=0)

labels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T^{rel}$']

plt.rcParams.update({'font.size': 15})
fig = plt.figure(figsize=(22, 7))

# fig.suptitle("Particle Feature Distributions")

for i in range(3):
    fig.add_subplot(1, 3, i + 1)

    Xplot = dataset[:batch_size, :, :]

    _ = plt.hist(Xplot[:, :, i].reshape(-1), bins[i], histtype='step', label='real', color='red')
    plt.xlabel('Particle ' + labels[i])
    plt.ylabel('Number of Particles')
    plt.title('JSD = ')
    plt.legend(loc=1, prop={'size': 11})

fig.tight_layout(pad=2.5)

plt.show()
