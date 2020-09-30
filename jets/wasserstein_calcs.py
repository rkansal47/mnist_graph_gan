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

x = np.arange(epochs + 1, step=5)

plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(22, 5))

for i in range(3):
    fig.add_subplot(1, 3, i + 1)
    for k in range(len(num_samples)):
        plt.plot(x, np.log10(np.array(losses['w1_' + str(num_samples[k]) + 'm'])[:, i]), label=str(num_samples[k]) + ' Jet Samples')
    # plt.ylim((0, 5))
    plt.legend(loc=1, prop={'size': 11})
    plt.xlabel('Epoch')
    plt.ylabel('Particle ' + labels[i] + ' LogJSD')
# plt.legend()
plt.savefig(dir + 'losses/' + name + "/w1.pdf", bbox_inches='tight')
plt.close()
