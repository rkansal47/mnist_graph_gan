import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
from os import remove
import mplhep as hep

import logging

plt.switch_backend('agg')


def save_sample_outputs(args, D, G, X, name, epoch, losses, X_loaded=None, gen_out=None):
    logging.info("drawing figs")
    plt.rcParams.update({'font.size': 16})
    plt.style.use(hep.style.CMS)

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

    jlabels = ['Relative Mass', 'Relative $p_T']
    mbins = np.arange(0, 0.225, 0.0045)

    # Generating data

    G.eval()
    if gen_out is None:
        gen_out = utils.gen(args, G, num_samples=args.batch_size, X_loaded=X_loaded).cpu().detach().numpy()
        for i in range(int(args.num_samples / args.batch_size)):
            gen_out = np.concatenate((gen_out, utils.gen(args, G, num_samples=args.batch_size, X_loaded=X_loaded).cpu().detach().numpy()), 0)
        gen_out = gen_out[:args.num_samples]
    elif args.w1_tot_samples < args.num_samples:
        for i in range(int((args.num_samples - args.w1_tot_samples) / args.batch_size) + 1):
            gen_out = np.concatenate((gen_out, utils.gen(args, G, num_samples=args.batch_size, X_loaded=X_loaded).cpu().detach().numpy()), 0)
        gen_out = gen_out[:args.num_samples]

    X_rn, mask_real = utils.unnorm_data(args, X.cpu().detach().numpy()[:args.num_samples], real=True)
    gen_out, mask_gen = utils.unnorm_data(args, gen_out[:args.num_samples], real=False)

    if args.jf:
        realjf = utils.jet_features(X_rn, mask=mask_real)
        genjf = utils.jet_features(gen_out, mask=mask_gen)

    if args.mask:
        parts_real = X_rn[mask_real]
        parts_gen = gen_out[mask_gen]
    else:
        parts_real = X_rn.reshape(-1, args.node_feat_size)
        parts_gen = gen_out.reshape(-1, args.node_feat_size)

    logging.info("real, gen outputs: ")
    logging.info(X_rn.shape)
    logging.info(gen_out.shape)

    logging.info(X_rn[0][:10])
    logging.info(gen_out[0][:10])

    # Plot particle features + jet mass

    fig = plt.figure(figsize=(30, 8))

    for i in range(3):
        fig.add_subplot(1, 4, i + 1)
        plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
        _ = plt.hist(parts_real[:, i], pbins[i], histtype='step', label='Real', color='red')
        _ = plt.hist(parts_gen[:, i], pbins[i], histtype='step', label='Generated', color='blue')
        plt.xlabel('Particle ' + plabels[i])
        plt.ylabel('Number of Particles')
        # plt.title('JSD = ' + str(round(losses['jsdm'][-1][i], 3)) + ' ± ' + str(round(losses['jsdstd'][-1][i], 3)))
        plt.legend(loc=1, prop={'size': 18})

    fig.add_subplot(1, 4, 4)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    # plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(realjf[:, 0], bins=mbins, histtype='step', label='Real', color='red')
    _ = plt.hist(genjf[:, 0], bins=mbins, histtype='step', label='Generated', color='blue')
    plt.xlabel('Jet $m/p_{T}$')
    plt.ylabel('Jets')
    plt.legend(loc=1, prop={'size': 18})

    name = args.name + "/" + str(epoch)
    plt.tight_layout(2.0)
    plt.savefig(args.figs_path + name + ".pdf", bbox_inches='tight')
    plt.close()

    # Plot loss

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
    plt.close()

    # Plot fid

    if args.fid:
        fid_5 = losses['fid'][::5]
        x = np.arange(len(losses['fid']), step=5)

        plt.figure()
        plt.plot(x, np.log10(fid_5))
        # plt.ylim((0, 5))
        plt.xlabel('Epoch')
        plt.ylabel('Log10FID')
        # plt.legend()
        plt.savefig(args.losses_path + name + "_fid.pdf", bbox_inches='tight')
        plt.close()

    # Plot W1

    if args.w1 and epoch >= 5:
        x = np.arange(0, epoch + 1, 5)[-len(losses['w1_' + str(args.w1_num_samples[0]) + 'm']):]

        plt.rcParams.update({'font.size': 12})
        colors = ['blue', 'green', 'orange']

        fig = plt.figure(figsize=(30, 7))

        logging.info(x.shape)
        logging.info(np.array(losses['w1_' + str(args.w1_num_samples[0]) + 'm']).shape)

        for i in range(3):
            fig.add_subplot(1, 3, i + 1)
            for k in range(len(args.w1_num_samples)):
                plt.plot(x, np.log10(np.array(losses['w1_' + str(args.w1_num_samples[k]) + 'm'])[:, i]), label=str(args.w1_num_samples[k]) + ' Jet Samples', color=colors[k])
                # plt.fill_between(x, np.log10(np.array(losses['w1_' + str(args.num_samples[k]) + 'm'])[:, i] - np.array(losses['w1_' + str(args.num_samples[k]) + 'std'])[:, i]), np.log10(np.array(losses['w1_' + str(args.num_samples[k]) + 'm'])[:, i] + np.array(losses['w1_' + str(args.num_samples[k]) + 'std'])[:, i]), color=colors[k], alpha=0.2)
                # plt.plot(x, np.ones(len(x)) * np.log10(realw1m[k][i]), '--', label=str(args.num_samples[k]) + ' Real W1', color=colors[k])
                # plt.fill_between(x, np.log10(np.ones(len(x)) * (realw1m[k][i] - realw1std[k][i])), np.log10(np.ones(len(x)) * (realw1m[k][i] + realw1std[k][i])), color=colors[k], alpha=0.2)
            plt.legend(loc=2, prop={'size': 11})
            plt.xlabel('Epoch')
            plt.ylabel('Particle ' + plabels[i] + ' LogW1')

        plt.savefig(args.losses_path + name + "_w1.pdf", bbox_inches='tight')
        plt.close()

        if args.jf:
            x = np.arange(0, epoch + 1, 5)[-len(losses['w1j_' + str(args.w1_num_samples[0]) + 'm']):]
            fig = plt.figure(figsize=(20, 7))

            for i in range(2):
                fig.add_subplot(1, 2, i + 1)
                for k in range(len(args.w1_num_samples)):
                    plt.plot(x, np.log10(np.array(losses['w1j_' + str(args.w1_num_samples[k]) + 'm'])[:, i]), label=str(args.w1_num_samples[k]) + ' Jet Samples', color=colors[k])
                plt.legend(loc=2, prop={'size': 11})
                plt.xlabel('Epoch')
                plt.ylabel('Particle ' + jlabels[i] + ' LogW1')

            plt.savefig(args.losses_path + name + "_w1j.pdf", bbox_inches='tight')
            plt.close()

    # save losses and remove earlier ones

    for key in losses: np.savetxt(args.losses_path + args.name + "/" + key + '.txt', losses[key])

    try:
        remove(args.losses_path + args.name + "/" + str(epoch - args.save_epochs) + ".pdf")
        remove(args.losses_path + args.name + "/" + str(epoch - args.save_epochs) + "_w1.pdf")
        remove(args.losses_path + args.name + "/" + str(epoch - args.save_epochs) + "_w1j.pdf")
        # remove(args.losses_path + args.name + "/" + str(epoch - args.save_epochs) + "_fid.pdf")
    except:
        logging.info("couldn't remove loss file")

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
