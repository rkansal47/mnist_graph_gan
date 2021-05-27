import numpy as np
import matplotlib.pyplot as plt
import utils
import os
from jets_dataset import JetsDataset
import mplhep as hep
from skhep.math.vectors import LorentzVector
from tqdm import tqdm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import save_outputs
# plt.switch_backend('macosx')
plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)


dirs = os.listdir('final_models')

num_samples = 50000

samples_dict = {'g': {}, 't': {}, 'q': {}}

for dir in dirs:
    print(dir)
    if dir == '.DS_Store': continue

    model_name = dir.split('_')[0]

    if not (model_name == 'fcpnet' or model_name == 'graphcnnpnet' or model_name == 'mp' or model_name == 'mppnet'):
        continue

    if not (model_name == 'mppnet'):
        continue


    samples = np.load('final_models/' + dir + '/samples.npy')[:num_samples]

    path = 'final_models/' + dir + '/'
    files = os.listdir(path)
    for file in files:
        if file[-4:] == ".txt": args_file = file

    args = eval(open(path + args_file).read())
    args['datasets_path'] = 'datasets/'
    args = utils.objectview(args)

    X = JetsDataset(args)

    gen_out_rn, mask_gen = utils.unnorm_data(args, samples, real=False)

    dataset = dir.split('_')[1]

    if model_name == 'fcpnet':
        samples_dict[dataset]['FC'] = (gen_out_rn, mask_gen)
    elif model_name == 'graphcnnpnet':
        samples_dict[dataset]['GraphCNN'] = (gen_out_rn, mask_gen)
    elif model_name == 'mp':
        samples_dict[dataset]['MP'] = (gen_out_rn, mask_gen)
    elif model_name == 'mppnet':
        samples_dict[dataset]['MPPNET'] = (gen_out_rn, mask_gen)

for dataset in samples_dict.keys():
    args = utils.objectview({'datasets_path': 'datasets/', 'ttsplit': 0.7, 'node_feat_size': 3, 'num_hits': 30, 'coords': 'polarrel', 'dataset': 'jets', 'clabels': 0, 'jets': dataset, 'norm': 1, 'mask': True, 'real_only': False})
    X = JetsDataset(args, train=False)
    X = X[:][0]
    X_rn, mask_real = utils.unnorm_data(args, X[:num_samples].cpu().detach().numpy(), real=True)
    samples_dict[dataset]['Real'] = (X_rn, mask_real)

samples_dict

efps = {}
for dataset in samples_dict.keys():
    efps[dataset] = {}
    for key in line_opts.keys():
        samples, mask = samples_dict[dataset][key]
        efps[dataset][key] = utils.efp(utils.objectview({'mask': key == 'Real' or key == 'MP', 'num_hits': 30}), samples, mask, key == 'Real')[:, 0]



%matplotlib inline

plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)

line_opts = {'Real': {'color': 'red', 'linewidth': 2, 'linestyle': 'solid'},
                'FC': {'color': 'green', 'linewidth': 2, 'linestyle': 'dashdot'},
                'GraphCNN': {'color': 'orange', 'linewidth': 2, 'linestyle': 'dotted'},
                'MP': {'color': 'blue', 'linewidth': 2, 'linestyle': 'dashed'},
                # 'MPPNET': {'color': 'purple', 'linewidth': 2, 'linestyle': (0, (5, 10))},
            }

fig = plt.figure(figsize=(36, 24))
i = 0
for dataset in samples_dict.keys():
    if dataset == 'g':
        efpbins = np.linspace(0, 0.0013, 51)
        pbins = [np.linspace(-0.3, 0.3, 101), np.linspace(0, 0.1, 101)]
        ylims = [1.3e5, 0.7e5, 0, 1.75e4]
    elif dataset == 't':
        efpbins = np.linspace(0, 0.0045, 51)
        pbins = [np.arange(-0.5, 0.5, 0.005), np.arange(0, 0.1, 0.001)]
        ylims = [0.35e5, 0.7e5, 0, 0.35e4]
    else:
        efpbins = np.linspace(0, 0.002, 51)
        pbins = [np.linspace(-0.3, 0.3, 101), np.linspace(0, 0.2, 101)]
        ylims = [1.5e5, 1.8e5, 0, 2e4]

    mbins = np.linspace(0, 0.225, 51)

    fig.add_subplot(3, 4, i * 4 + 1)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Particle $\eta^{rel}$')
    plt.ylabel('Number of Particles')

    for key in line_opts.keys():
        samples, mask = samples_dict[dataset][key]
        if key == 'MP' or key == 'Real':
            parts = samples[mask]
        else:
            parts = samples.reshape(-1, 3)

        _ = plt.hist(parts[:, 0], pbins[0], histtype='step', label=key, **line_opts[key])

    plt.legend(loc=1, prop={'size': 18}, fancybox=True)
    plt.ylim(0, ylims[0])

    fig.add_subplot(3, 4, i * 4 + 2)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Particle $p_T^{rel}$')
    plt.ylabel('Number of Particles')

    for key in line_opts.keys():
        samples, mask = samples_dict[dataset][key]
        if key == 'MP' or key == 'Real':
            parts = samples[mask]
        else:
            parts = samples.reshape(-1, 3)

        _ = plt.hist(parts[:, 2], pbins[1], histtype='step', label=key, **line_opts[key])

    plt.legend(loc=1, prop={'size': 18}, fancybox=True)
    plt.ylim(0, ylims[1])

    fig.add_subplot(3, 4, i * 4 + 3)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Relative Jet Mass')
    plt.ylabel('Number of Jets')

    for key in line_opts.keys():
        samples, mask = samples_dict[dataset][key]
        masses = utils.jet_features(samples, mask=mask_real)[:, 0]

        _ = plt.hist(masses, mbins, histtype='step', label=key, **line_opts[key])

    plt.legend(loc=1, prop={'size': 18}, fancybox=True)
    # plt.ylim(0, ylims[1])


    fig.add_subplot(3, 4, i * 4 + 4)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Jet EFP', x = 0.7)
    plt.ylabel('Number of Jets')

    for key in line_opts.keys():
        _ = plt.hist(efps[dataset][key], efpbins, histtype='step', label=key, **line_opts[key])

    plt.legend(loc=1, prop={'size': 18}, fancybox=True)
    plt.ylim(0, ylims[3])

    i += 1

plt.tight_layout(pad=0.5)
plt.savefig('final_figure.pdf', bbox_inches='tight')
plt.show()
