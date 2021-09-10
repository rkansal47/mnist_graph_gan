from model import Graph_GAN
import torch
import numpy as np
import utils
from torch.distributions.normal import Normal
from jets_dataset import JetsDataset
from tqdm import tqdm
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128


normal_dist = Normal(torch.tensor(0.).to(device), torch.tensor(0.2).to(device))

# dir = './'
dir = '/graphganvol/mnist_graph_gan/jets/'

args = {'dataset_path': dir + 'datasets/', 'num_hits': 30, 'coords': 'polarrel', 'latent_node_size': 32}
X = JetsDataset(utils.objectview(args))

N = len(X)

name = '7_batch_size_128_coords_polarrel'

full_path = dir + 'models/' + name + '/'

rng = np.random.default_rng()

num_samples = np.array([100, 1000, 10000])
num_batches = np.array(100000 / num_samples, dtype=int)

num_batches

epochs = 980

losses = {}
losses['w1_10000m'] = []
losses['w1_1000m'] = []
losses['w1_100m'] = []
losses['w1_10000std'] = []
losses['w1_1000std'] = []
losses['w1_100std'] = []

for i in range(0, epochs + 1, 5):
    print(i)
    # if i != 535: continue  # COMMENT OUT
    G = torch.load(full_path + 'G_' + str(i) + '.pt', map_location=device)
    for k in range(len(num_samples)):
        print("Num Samples: " + str(num_samples[k]))
        w1s = []
        for j in tqdm(range(num_batches[k])):
            gen_out = utils.gen(utils.objectview(args), G, dist=normal_dist, num_samples=batch_size).cpu().detach().numpy()
            for i in range(int(num_samples[k] / batch_size)):
                gen_out = np.concatenate((gen_out, utils.gen(utils.objectview(args), G, dist=normal_dist, num_samples=batch_size).cpu().detach().numpy()), 0)
            gen_out = gen_out[:num_samples[k]]

            sample = X[rng.choice(N, size=num_samples[k])].cpu().detach().numpy()
            w1 = []

            for i in range(3):
                w1.append(wasserstein_distance(sample[:, :, i].reshape(-1), gen_out[:, :, i].reshape(-1)))

            w1s.append(w1)

        losses['w1_' + str(num_samples[k]) + 'm'].append(np.mean(np.array(w1s), axis=0))
        losses['w1_' + str(num_samples[k]) + 'std'].append(np.std(np.array(w1s), axis=0))

for key in losses:
    np.savetxt(dir + 'losses/' + name + '/' + key + '.txt', losses[key])

labels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T^{rel}$']

5 * (np.argmin(losses['w1_10000m'][:, 1]) + 1)
5 * (np.argmin(losses['w1_10000m'][:, 0]) + 1)
5 * (np.argmin(losses['w1_10000m'][:, 2]) + 1)

for k in range(3):
    losses['w1_' + str(num_samples[k]) + 'm'] = np.loadtxt('./losses/7/w1_' + str(num_samples[k]) + 'm.txt')
    losses['w1_' + str(num_samples[k]) + 'std'] = np.loadtxt('./losses/7/w1_' + str(num_samples[k]) + 'std.txt')

len(losses['w1_1000m'])

790/5 - 1

losses['w1_100m'][157]
losses['w1_100std'][157]
losses['w1_1000m'][157]
losses['w1_1000std'][157]
losses['w1_10000m'][157]
losses['w1_10000std'][157]


x = np.arange(5, epochs + 1, step=5)

realw1m = [[0.00584264, 0.00556786, 0.0014096], [0.00179309, 0.00170772, 0.00046562], [0.00050421, 0.00046688, 0.00010837]]
realw1std = [[0.00214083, 0.00204827, 0.00051136], [1.06719727e-04, 1.15946909e-04, 1.63954948e-05], [1.06719727e-04, 1.15946909e-04, 1.63954948e-05]]

plt.rcParams.update({'font.size': 12})
colors = ['blue', 'green', 'orange']


fig = plt.figure(figsize=(30, 7))

for i in range(3):
    fig.add_subplot(1, 3, i + 1)
    for k in range(len(num_samples)):
        plt.plot(x, np.log10(np.array(losses['w1_' + str(num_samples[k]) + 'm'])[:, i]), label=str(num_samples[k]) + ' Jet Samples', color=colors[k])
        # plt.fill_between(x, np.log10(np.array(losses['w1_' + str(num_samples[k]) + 'm'])[:, i] - np.array(losses['w1_' + str(num_samples[k]) + 'std'])[:, i]), np.log10(np.array(losses['w1_' + str(num_samples[k]) + 'm'])[:, i] + np.array(losses['w1_' + str(num_samples[k]) + 'std'])[:, i]), color=colors[k], alpha=0.2)
        plt.plot(x, np.ones(len(x)) * np.log10(realw1m[k][i]), '--', label=str(num_samples[k]) + ' Real W1', color=colors[k])
        plt.fill_between(x, np.log10(np.ones(len(x)) * (realw1m[k][i] - realw1std[k][i])), np.log10(np.ones(len(x)) * (realw1m[k][i] + realw1std[k][i])), color=colors[k], alpha=0.2)
    # plt.ylim((0, 5))
    plt.legend(loc=2, prop={'size': 11})
    plt.xlabel('Epoch')
    plt.ylabel('Particle ' + labels[i] + ' LogW1')
# plt.legend()
plt.savefig(dir + 'losses/7/logw1.pdf', bbox_inches='tight')
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
