import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
from os import remove
import mplhep as hep

import logging

from guppy import hpy
h = hpy()

plt.switch_backend('agg')
plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)


# split regions for cGAN - make sure to change in evaluation as well
pt_regions = [0, 1045, 1175, 3000]
num_regions = len(pt_regions) - 1


def plot_part_feats(args, X_rn, mask_real, gen_out, mask_gen, name, losses=None, show=False):
    if args.coords == 'cartesian':
        plabels = ['$p_x$ (GeV)', '$p_y$ (GeV)', '$p_z$ (GeV)']
        bin = np.arange(-500, 500, 10)
        pbins = [bin, bin, bin]
    elif args.coords == 'polarrel':
        if args.dataset == 'jets':
            plabels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T^{rel}$']
            if args.jets == 'g' or args.jets == 'q' or args.jets == 'w' or args.jets == 'z':
                if args.num_hits == 100:
                    pbins = [np.arange(-0.5, 0.5, 0.005), np.arange(-0.5, 0.5, 0.005), np.arange(0, 0.1, 0.001)]
                else:
                    pbins = [np.linspace(-0.3, 0.3, 101), np.linspace(-0.3, 0.3, 101), np.linspace(0, 0.2, 101)]
                    ylims = [3e5, 3e5, 3e5]
            elif args.jets == 't':
                pbins = [np.linspace(-0.5, 0.5, 101), np.linspace(-0.5, 0.5, 101), np.linspace(0, 0.2, 101)]
        elif args.dataset == 'jets-lagan':
            plabels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T^{rel}$']
            pbins = [np.linspace(-1.25, 1.25, 25 + 1), np.linspace(-1.25, 1.25, 25 + 1), np.linspace(0, 1, 51)]


    elif args.coords == 'polarrelabspt':
        plabels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T (GeV)$']
        pbins = [np.arange(-0.5, 0.5, 0.01), np.arange(-0.5, 0.5, 0.01), np.arange(0, 400, 4)]

    if args.mask:
        parts_real = X_rn[mask_real]
        parts_gen = gen_out[mask_gen]
    else:
        parts_real = X_rn.reshape(-1, args.node_feat_size)
        parts_gen = gen_out.reshape(-1, args.node_feat_size)

    fig = plt.figure(figsize=(22, 8))

    for i in range(3):
        fig.add_subplot(1, 3, i + 1)
        plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
        _ = plt.hist(parts_real[:, i], pbins[i], histtype='step', label='Real', color='red')
        _ = plt.hist(parts_gen[:, i], pbins[i], histtype='step', label='Generated', color='blue')
        plt.xlabel('Particle ' + plabels[i])
        plt.ylabel('Number of Particles')
        if hasattr(args, 'const_ylim') and args.const_ylim: plt.ylim(0, ylims[i])
        if losses is not None: plt.title('$W_1$ = {:.2e}'.format(losses['w1_' + str(args.w1_num_samples[-1]) + 'm'][-1][i]), fontsize=12)
        plt.legend(loc=1, prop={'size': 18})

    plt.tight_layout(pad=2.0)
    plt.savefig(args.figs_path + name + ".pdf", bbox_inches='tight')
    if show: plt.show()
    else: plt.close()


def plot_part_feats_jet_mass(args, X_rn, mask_real, gen_out, mask_gen, realjf, genjf, name, losses=None, cregions=None, show=False):
    if args.coords == 'cartesian':
        plabels = ['$p_x$ (GeV)', '$p_y$ (GeV)', '$p_z$ (GeV)']
        bin = np.arange(-500, 500, 10)
        pbins = [bin, bin, bin]
    elif args.coords == 'polarrel':
        plabels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T^{rel}$']
        if args.jets == 'g' or args.jets == 'q' or args.jets == 'w' or args.jets == 'z':
            if args.num_hits == 100:
                pbins = [np.arange(-0.5, 0.5, 0.005), np.arange(-0.5, 0.5, 0.005), np.arange(0, 0.1, 0.001)]
            else:
                pbins = [np.linspace(-0.3, 0.3, 100), np.linspace(-0.3, 0.3, 100), np.linspace(0, 0.2, 100)]
        elif args.jets == 't':
            pbins = [np.linspace(-0.5, 0.5, 100), np.linspace(-0.5, 0.5, 100), np.linspace(0, 0.2, 100)]
    elif args.coords == 'polarrelabspt':
        plabels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T (GeV)$']
        pbins = [np.arange(-0.5, 0.5, 0.01), np.arange(-0.5, 0.5, 0.01), np.arange(0, 400, 4)]

    if args.jets == 'g' or args.jets == 'q' or args.jets == 't': mbins = np.linspace(0, 0.225, 51)
    else: mbins = np.linspace(0, 0.12, 51)

    if args.mask:
        parts_real = X_rn[mask_real]
        parts_gen = gen_out[mask_gen]
    else:
        parts_real = X_rn.reshape(-1, args.node_feat_size)
        parts_gen = gen_out.reshape(-1, args.node_feat_size)

    fig = plt.figure(figsize=(30, 8))

    for i in range(3):
        fig.add_subplot(1, 4, i + 1)
        plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
        _ = plt.hist(parts_real[:, i], pbins[i], histtype='step', label='Real', color='red')
        _ = plt.hist(parts_gen[:, i], pbins[i], histtype='step', label='Generated', color='blue')
        plt.xlabel('Particle ' + plabels[i])
        plt.ylabel('Number of Particles')
        if losses is not None: plt.title('$W_1$ = {:.2e}'.format(losses['w1_' + str(args.w1_num_samples[-1]) + 'm'][-1][i]), fontsize=12)
        plt.legend(loc=1, prop={'size': 18})

    fig.add_subplot(1, 4, 4)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(realjf[:, 0], bins=mbins, histtype='step', label='Real', color='red')
    _ = plt.hist(genjf[:, 0], bins=mbins, histtype='step', label='Generated', color='blue')
    plt.xlabel('Jet $m/p_{T}$')
    plt.ylabel('Jets')
    plt.legend(loc=1, prop={'size': 18})
    if losses is not None: plt.title('$W_1$ = {:.2e}'.format(losses['w1j_' + str(args.w1_num_samples[-1]) + 'm'][-1][0]), fontsize=16)

    plt.tight_layout(pad=2.0)
    plt.savefig(args.figs_path + name + ".pdf", bbox_inches='tight')
    if show: plt.show()
    else: plt.close()


def plot_part_feats_jet_mass_cregions(args, X_rn, mask_real, gen_out, mask_gen, realjf, genjf, name, cregions, losses=None, show=False):
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
            pbins = [np.linspace(-0.5, 0.5, 101), np.linspace(-0.5, 0.5, 101), np.linspace(0, 0.2, 101)]
    elif args.coords == 'polarrelabspt':
        plabels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T (GeV)$']
        pbins = [np.arange(-0.5, 0.5, 0.01), np.arange(-0.5, 0.5, 0.01), np.arange(0, 400, 4)]

    if args.jets == 'g' or args.jets == 'q' or args.jets == 't': mbins = np.linspace(0, 0.225, 51)
    else: mbins = np.linspace(0, 0.12, 51)

    parts_real = []
    parts_gen = []

    for i in range(num_regions):
        if args.mask:
            parts_real.append(X_rn[cregions[i]][mask_real[cregions[i]]])
            parts_gen.append(gen_out[cregions[i]][mask_gen[cregions[i]]])
        else:
            parts_real.append(X_rn[cregions[i]].reshape(-1, args.node_feat_size))
            parts_gen.append(gen_out[cregions[i]].reshape(-1, args.node_feat_size))

    fig, axs = plt.subplots(num_regions, 4, figsize=(30, num_regions * 7))

    for j in range(num_regions):
        for i in range(3):
            axs[j, i].ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
            _ = axs[j, i].hist(parts_real[j][:, i], pbins[i], histtype='step', label='Real', color='red')
            _ = axs[j, i].hist(parts_gen[j][:, i], pbins[i], histtype='step', label='Generated', color='blue')
            axs[j, i].set_xlabel('Particle ' + plabels[i])
            if i == 0: axs[j, i].set_ylabel(f'{pt_regions[j]} $\leq p_T < $ {pt_regions[j + 1]} GeV' + ' \t Particles')
            else: axs[j, i].set_ylabel('Particles')
            axs[j, i].legend(loc=1, prop={'size': 18})

        axs[j, 3].ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
        _ = axs[j, 3].hist(realjf[cregions[i]][:, 0], bins=mbins, histtype='step', label='Real', color='red')
        _ = axs[j, 3].hist(genjf[cregions[i]][:, 0], bins=mbins, histtype='step', label='Generated', color='blue')
        axs[j, 3].set_xlabel('Jet $m/p_{T}$')
        axs[j, 3].set_ylabel('Jets')
        axs[j, 3].legend(loc=1, prop={'size': 18})

    plt.tight_layout(pad=2.0)
    plt.savefig(args.figs_path + name + ".pdf", bbox_inches='tight')
    if show: plt.show()
    else: plt.close()


def plot_jet_feats(args, realjf, genjf, realefp, genefp, name, losses=None, show=False):
    if args.jets == 'g':
        binranges = [0.0013, 0.0004, 0.0004, 0.0004, 0.0004]
    elif args.jets == 'q':
        binranges = [0.002, 0.001, 0.001, 0.0005, 0.0005]
    else:
        binranges = [0.0045, 0.0035, 0.004, 0.002, 0.003]

    bins = [np.linspace(0, binr, 101) for binr in binranges]

    if args.jets == 'g' or args.jets == 'q' or args.jets == 't': mbins = np.linspace(0, 0.225, 51)
    else: mbins = np.linspace(0, 0.12, 51)

    fig = plt.figure(figsize=(20, 12))

    for i in range(5):
        fig.add_subplot(2, 3, i + 2)
        plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
        plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
        _ = plt.hist(realefp[:, i], bins[i], histtype='step', label='Real', color='red')
        _ = plt.hist(genefp[:, i], bins[i], histtype='step', label='Generated', color='blue')
        plt.xlabel('EFP ' + str(i + 1), x = 0.7)
        plt.ylabel('Jets')
        plt.legend(loc=1, prop={'size': 18})
        if losses is not None: plt.title('$W_1$ = {:.2e}'.format(losses['w1j_' + str(args.w1_num_samples[-1]) + 'm'][-1][i + 2]), fontsize=16)

    fig.add_subplot(2, 3, 1)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(realjf[:, 0], bins=mbins, histtype='step', label='Real', color='red')
    _ = plt.hist(genjf[:, 0], bins=mbins, histtype='step', label='Generated', color='blue')
    plt.xlabel('Jet $m/p_{T}$')
    plt.ylabel('Jets')
    plt.legend(loc=1, prop={'size': 18})
    if losses is not None: plt.title('$W_1$ = {:.2e}'.format(losses['w1j_' + str(args.w1_num_samples[-1]) + 'm'][-1][0]), fontsize=16)

    plt.tight_layout(pad=0.5)
    plt.savefig(args.figs_path + name + ".pdf", bbox_inches='tight')
    if show: plt.show()
    else: plt.close()


def plot_jet_mass_pt(args, realjf, genjf, name, show=False):
    if args.dataset == 'jets':
        jlabels = ['Jet Relative Mass', 'Jet Relative $p_T$']
        binsm = np.linspace(0, 0.225, 101)
        binspt = np.linspace(0.5, 1.2, 101)
    elif args.dataset == 'jets-lagan':
        jlabels = ['Jet Mass (GeV)', 'Jet $p_T$ (GeV)']
        binsm = np.linspace(40, 120, 51)
        binspt = np.linspace(220, 340, 51)

    fig = plt.figure(figsize=(16, 8))

    fig.add_subplot(1, 2, 1)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    # plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(realjf[:, 0], bins=binsm, histtype='step', label='Real', color='red')
    _ = plt.hist(genjf[:, 0], bins=binsm, histtype='step', label='Generated', color='blue')
    plt.xlabel(jlabels[0])
    plt.ylabel('Jets')
    plt.legend(loc=1, prop={'size': 18})

    fig.add_subplot(1, 2, 2)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(realjf[:, 1], bins=binspt, histtype='step', label='Real', color='red')
    _ = plt.hist(genjf[:, 1], bins=binspt, histtype='step', label='Generated', color='blue')
    plt.xlabel(jlabels[1])
    plt.ylabel('Jets')
    plt.legend(loc=1, prop={'size': 18})

    plt.tight_layout(pad=2)
    plt.savefig(args.figs_path + name + ".pdf", bbox_inches='tight')
    if show: plt.show()
    else: plt.close()


def plot_losses(args, losses, name, show=False):
    plt.figure()

    if(args.loss == "og" or args.loss == "ls"):
        plt.plot(losses['Dr'], label='Discriminitive real loss')
        plt.plot(losses['Df'], label='Discriminitive fake loss')
        plt.plot(losses['G'], label='Generative loss')
    elif(args.loss == 'w'):
        plt.plot(losses['D'], label='Critic loss')
    elif(args.loss == 'hinge'):
        plt.plot(losses['Dr'], label='Discriminitive real loss')
        plt.plot(losses['Df'], label='Discriminitive fake loss')
        plt.plot(losses['G'], label='Generative loss')

    if(args.gp): plt.plot(losses['gp'], label='Gradient penalty')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(args.losses_path + name + ".pdf", bbox_inches='tight')
    if show: plt.show()
    else: plt.close()


def plot_eval(args, losses, name, epoch, show=False):
    if args.coords == 'cartesian': plabels = ['$p_x$ (GeV)', '$p_y$ (GeV)', '$p_z$ (GeV)']
    elif args.coords == 'polarrel': plabels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T^{rel}$']
    elif args.coords == 'polarrelabspt': plabels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T (GeV)$']
    jlabels = ['Relative Mass', 'Relative $p_T$', 'EFP']
    colors = ['blue', 'green', 'orange', 'red', 'yellow']

    x = np.arange(0, epoch + 1, args.save_epochs)[-len(losses['w1_' + str(args.w1_num_samples[0]) + 'm']):]

    if args.jf: fig = plt.figure(figsize=(30, 24))
    else: fig = plt.figure(figsize=(30, 16))

    for i in range(3):
        if args.jf: fig.add_subplot(3, 3, i + 1)
        else: fig.add_subplot(2, 3, i + 1)

        if not args.clabels:
            for k in range(len(args.w1_num_samples)):
                plt.plot(x, np.log10(np.array(losses['w1_' + str(args.w1_num_samples[k]) + 'm'])[:, i]), label=str(args.w1_num_samples[k]) + ' Jet Samples', color=colors[k])
                # plt.fill_between(x, np.log10(np.array(losses['w1_' + str(args.num_samples[k]) + 'm'])[:, i] - np.array(losses['w1_' + str(args.num_samples[k]) + 'std'])[:, i]), np.log10(np.array(losses['w1_' + str(args.num_samples[k]) + 'm'])[:, i] + np.array(losses['w1_' + str(args.num_samples[k]) + 'std'])[:, i]), color=colors[k], alpha=0.2)
                # plt.plot(x, np.ones(len(x)) * np.log10(realw1m[k][i]), '--', label=str(args.num_samples[k]) + ' Real $W_1$', color=colors[k])
                # plt.fill_between(x, np.log10(np.ones(len(x)) * (realw1m[k][i] - realw1std[k][i])), np.log10(np.ones(len(x)) * (realw1m[k][i] + realw1std[k][i])), color=colors[k], alpha=0.2)
        else:
            for k in range(num_regions):
                plt.plot(x, np.log10(np.array(losses[f'intra_w1_{args.w1_num_samples[0]}m'])[:, i + (k * 3)]), label=f'Region {k + 1}', color=colors[k], linestyle='dashed')

            plt.plot(x, np.log10(np.array(losses[f'w1_{args.w1_num_samples[0]}m'])[:, i]), label='All Regions', color=colors[num_regions])

        plt.legend(loc=1)
        plt.xlabel('Epoch')
        plt.ylabel('Particle ' + plabels[i] + ' Log$W_1$')

    if args.jf:
        x = np.arange(0, epoch + 1, args.save_epochs)[-len(losses['w1j_' + str(args.w1_num_samples[0]) + 'm']):]

        for i in range(2):
            fig.add_subplot(3, 3, i + 4)

            if not args.clabels:
                for k in range(len(args.w1_num_samples)):
                    plt.plot(x, np.log10(np.array(losses['w1j_' + str(args.w1_num_samples[k]) + 'm'])[:, i]), label=str(args.w1_num_samples[k]) + ' Jet Samples', color=colors[k])
            else:
                for k in range(num_regions):
                    plt.plot(x, np.log10(np.array(losses[f'intra_w1j_{args.w1_num_samples[0]}m'])[:, i + (k * 2)]), label=f'Region {k + 1}', color=colors[k], linestyle='dashed')

                plt.plot(x, np.log10(np.array(losses[f'w1j_{args.w1_num_samples[0]}m'])[:, i]), label='All Regions', color=colors[num_regions])

            plt.legend(loc=1)
            plt.xlabel('Epoch')
            plt.ylabel(jlabels[i] + ' Log$W_1$')

        if args.efp:
            fig.add_subplot(3, 3, 6)
            for i in range(5):
                plt.plot(x, np.log10(np.array(losses['w1j_' + str(args.w1_num_samples[-1]) + 'm'])[:, i + 2]), label='EFP ' + str(i + 1), color=colors[i])
            plt.legend(loc=1)
            plt.xlabel('Epoch')
            plt.ylabel('Jet EFPs Log$W_1$')

    x = x[-len(losses['mmd']):]
    paxis = 3 if args.jf else 2
    metrics = {'mmd': (1, 'LogMMD'), 'coverage': (2, 'Coverage')}
    for key, (i, label) in metrics.items():
        fig.add_subplot(paxis, 3, i + (paxis - 1) * 3)

        if not args.clabels:
            if key == 'coverage': plt.plot(x, np.array(losses[key]))
            else: plt.plot(x, np.log10(np.array(losses[key])))
        else:
            for k in range(num_regions):
                plot_thing = np.array(losses['intra_' + key])[:, k] if key == 'coverage' else np.log10(np.array(losses['intra_' + key])[:, k])
                plt.plot(x, plot_thing, label=f'Region {k + 1}', color=colors[k], linestyle='dashed')

            plot_thing = np.array(losses[key]) if key == 'coverage' else np.log10(np.array(losses[key]))
            plt.plot(x, plot_thing, label='All Regions', color=colors[num_regions])
            plt.legend(loc=1)

        plt.xlabel('Epoch')
        plt.ylabel(label)

    fig.add_subplot(paxis, 3, (paxis) * 3)

    if not args.clabels:
        plt.plot(x, np.log10(np.array(losses['fpnd'])))
    else:
        plt.plot(x, np.log10(np.array(losses['fpnd'])), label='FPND', color=colors[0])
        plt.plot(x, np.log10(np.array(losses['fjpnd'])), label='FJPND', color=colors[1])
        plt.legend(loc=1)

    plt.xlabel('Epoch')
    plt.ylabel('LogFPND')

    plt.savefig(args.losses_path + name + ".pdf", bbox_inches='tight')
    if show: plt.show()
    else: plt.close()


def save_sample_outputs(args, D, G, X, epoch, losses, labels=None, gen_out=None):
    logging.info("drawing figs")

    # save loss arrays
    for key in losses: np.savetxt(args.losses_path + args.name + "/" + key + '.txt', losses[key])

    # Generating data
    G.eval()
    if gen_out is None:
        logging.info("gen out none")
        gen_out = utils.gen_multi_batch(args, G, args.num_samples, labels=labels)
    elif args.eval_tot_samples < args.num_samples:
        logging.info("gen out not large enough: size {}".format(len(gen_out)))
        gen_out = np.concatenate((gen_out, utils.gen_multi_batch(args, G, args.num_samples - args.eval_tot_samples, labels=labels)), 0)

    X_rn, mask_real = utils.unnorm_data(args, X.cpu().detach().numpy()[:args.num_samples], real=True)
    gen_out, mask_gen = utils.unnorm_data(args, gen_out[:args.num_samples], real=False)

    if args.clabels == 1:
        abs_labels = (labels[:args.eval_tot_samples, 0] * args.maxjf[0]).detach().numpy()
        cregions = [((abs_labels >= pt_regions[i]) * (abs_labels < pt_regions[i + 1])).squeeze() for i in range(len(pt_regions) - 1)]

    logging.info("real, gen outputs: \n {} \n {} \n {} \n {}".format(X_rn.shape, gen_out.shape, X_rn[0][:10], gen_out[0][:10]))

    name = args.name + "/" + str(epoch)

    logging.info(f"pre-plotting {h.heap()}")

    plot_part_feats(args, X_rn, mask_real, gen_out, mask_gen, name + 'p', losses)

    logging.info(f"plotted particle feats {h.heap()}")

    if args.jf:
        realjf = utils.jet_features(X_rn, mask=mask_real)
        genjf = utils.jet_features(gen_out, mask=mask_gen)

        if args.efp:
            realefp = utils.efp(args, X_rn, mask=mask_real, real=True)
            genefp = utils.efp(args, gen_out, mask=mask_gen, real=False)
            plot_jet_feats(args, realjf, genjf, realefp, genefp, name + 'j', losses)

            logging.info(f"plotted jet feats feats {h.heap()}")

        plot_jet_mass_pt(args, realjf, genjf, name + 'mpt')

        logging.info(f"plotted jet mass pt {h.heap()}")

        if args.clabels:
            plot_part_feats_jet_mass_cregions(args, X_rn, mask_real, gen_out, mask_gen, realjf, genjf, name + 'c', cregions, losses=losses)
            logging.info(f"plotted cregions {h.heap()}")

    if len(losses['G']) > 1:
        plot_losses(args, losses, name)
        logging.info(f"plotted losses {h.heap()}")
    # if args.fid: plot_fid(args, losses, name)

    # UNCOMMENT!!
    if args.eval and len(losses['w1_' + str(args.w1_num_samples[-1]) + 'm']) > 1:
        plot_eval(args, losses, name + '_eval', epoch)
        logging.info(f"plotted eval {h.heap()}")

    # remove previous plots
    try: remove(args.losses_path + args.name + "/" + str(epoch - args.save_epochs) + ".pdf")
    except: logging.info("couldn't remove loss file")

    try: remove(args.losses_path + args.name + "/" + str(epoch - args.save_epochs) + "_eval.pdf")
    except: logging.info("couldn't remove loss file")

    logging.info("saved figs")


def save_models(args, D, G, optimizers, name, epoch):
    if args.multi_gpu:
        torch.save(D.module.state_dict(), args.models_path + args.name + "/D_" + str(epoch) + ".pt")
        torch.save(G.module.state_dict(), args.models_path + args.name + "/G_" + str(epoch) + ".pt")
    else:
        torch.save(D.state_dict(), args.models_path + args.name + "/D_" + str(epoch) + ".pt")
        torch.save(G.state_dict(), args.models_path + args.name + "/G_" + str(epoch) + ".pt")

    torch.save(optimizers[0].state_dict(), args.models_path + args.name + "/D_optim_" + str(epoch) + ".pt")
    torch.save(optimizers[1].state_dict(), args.models_path + args.name + "/G_optim_" + str(epoch) + ".pt")
