import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
from jets_dataset import JetsDataset
from torch.distributions.normal import Normal
import mplhep as hep

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

num_samples = 10000

gen_out = utils.gen(utils.objectview(args), G, dist=normal_dist, num_samples=batch_size).cpu().detach().numpy()
for i in range(int(num_samples / batch_size)):
    gen_out = np.concatenate((gen_out, utils.gen(utils.objectview(args), G, dist=normal_dist, num_samples=batch_size).cpu().detach().numpy()), 0)
gen_out = gen_out[:num_samples]


# fig.suptitle("Particle Feature Distributions")

labels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T^{rel}$']

maxepp = [1.4130558967590332, 0.520724892616272, 0.8537549376487732]
Xplot = X[:num_samples, :, :].cpu().detach().numpy() * maxepp
gen_out *= maxepp

print(Xplot.shape)
print(gen_out.shape)

print(Xplot[0][:10])
print(gen_out[0][:10])

plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)

sf = [3, 2, 3]
rnd = [0, 1, 0]
castings = [int, float, int]

idx = int(epoch / 5 - 1)

bins = [np.arange(-0.3, 0.3, 0.005), np.arange(-0.3, 0.3, 0.005), np.arange(0, 0.2, 0.002)]

fig = plt.figure(figsize=(22, 8))

for i in range(3):
    fig.add_subplot(1, 3, i + 1)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(Xplot[:, :, i].reshape(-1), bins[i], histtype='step', label='Real', color='red')
    _ = plt.hist(gen_out[:, :, i].reshape(-1), bins[i], histtype='step', label='Generated', color='blue')
    plt.xlabel('Particle ' + labels[i])
    plt.ylabel('Particles')
    # plt.title('$W_1$ = (' + str(castings[i](round(w1m[idx][i] * int(10 ** sf[i]), rnd[i]))) + ' ± ' + str(castings[i](round(w1std[idx][i] * int(10 ** sf[i]), rnd[i]))) + ') $\\times 10^{-' + str(sf[i]) + '}$')
    title = '$W_1$ = (' + str(castings[i](round(w1m[idx][i] * int(10 ** sf[i]), rnd[i]))) + ' ± ' + str(castings[i](round(w1std[idx][i] * int(10 ** sf[i]), rnd[i]))) + ') $\\times 10^{-' + str(sf[i]) + '}$'
    lg = plt.legend(loc=1, prop={'size': 18})
    # lg.set_title(title)
    # lg.get_title().set_fontsize(13)

# name = args.name + "/" + str(epoch)

plt.tight_layout(2.0)
plt.savefig("7_" + str(epoch) + ".pdf", bbox_inches='tight')
plt.show()
