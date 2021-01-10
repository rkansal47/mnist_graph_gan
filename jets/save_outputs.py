import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
from os import remove
import mplhep as hep
from skhep.math.vectors import LorentzVector

plt.switch_backend('agg')


def save_sample_outputs(args, D, G, X, dist, name, epoch, losses, X_loaded=None):
    print("drawing figs")
    plt.rcParams.update({'font.size': 16})
    plt.style.use(hep.style.CMS)
    # if(args.fid): plt.suptitle("FID: " + str(losses['fid'][-1]))
    # noise = torch.load(args.noise_path + args.noise_file_name).to(args.device)

    G.eval()
    gen_out = utils.gen(args, G, dist=dist, num_samples=args.batch_size, X_loaded=X_loaded).cpu().detach().numpy()
    for i in range(int(args.num_samples / args.batch_size)):
        gen_out = np.concatenate((gen_out, utils.gen(args, G, dist=dist, num_samples=args.batch_size, X_loaded=X_loaded).cpu().detach().numpy()), 0)
    gen_out = gen_out[:args.num_samples]

    # print(gen_out.shape)

    if args.coords == 'cartesian':
        labels = ['$p_x$ (GeV)', '$p_y$ (GeV)', '$p_z$ (GeV)']
        bin = np.arange(-500, 500, 10)
        bins = [bin, bin, bin]
    elif args.coords == 'polarrel':
        labels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T^{rel}$']
        if args.jets == 'g':
            if args.num_hits == 100:
                bins = [np.arange(-0.5, 0.5, 0.005), np.arange(-0.5, 0.5, 0.005), np.arange(0, 0.1, 0.001)]
            else:
                bins = [np.linspace(-0.3, 0.3, 100), np.linspace(-0.3, 0.3, 100), np.linspace(0, 0.2, 100)]
        elif args.jets == 't':
            bins = [np.linspace(-0.5, 0.5, 100), np.linspace(-0.5, 0.5, 100), np.linspace(0, 0.2, 100)]
    elif args.coords == 'polarrelabspt':
        labels = ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T (GeV)$']
        bins = [np.arange(-0.5, 0.5, 0.01), np.arange(-0.5, 0.5, 0.01), np.arange(0, 400, 4)]

    labelsj = ['Relative Mass', 'Relative $p_T']

    # print(X)
    # print(X.shape)

    if args.coords == 'cartesian':
        Xplot = X.cpu().detach().numpy() * args.maxp / args.norm
        gen_out = gen_out * args.maxp / args.norm
    else:
        if args.mask_manual:
            mask_real = (X.cpu().detach().numpy()[:, :, 3] + 0.5) >= 1
            mask_gen = (gen_out[:, :, 3] + 0.5) >= 1

        Xplot = X.cpu().detach().numpy()[:, :, :3]
        Xplot = Xplot / args.norm
        Xplot[:, :, 2] += 0.5
        Xplot *= args.maxepp

        gen_out = gen_out[:, :, :3] / args.norm
        gen_out[:, :, 2] += 0.5
        gen_out *= args.maxepp

    if args.mask_manual:
        print(mask_real)
        print(mask_real.shape)
        parts_real = Xplot[mask_real]
        parts_gen = gen_out[mask_gen]
    else:
        for i in range(args.num_samples):
            for j in range(args.num_hits):
                if gen_out[i][j][2] < 0:
                    gen_out[i][j][2] = 0

        if args.mask:
            parts_real = Xplot[Xplot[:, :, args.node_feat_size - 1] > 0]
            parts_gen = gen_out[gen_out[:, :, args.node_feat_size - 1] > 0]
        else:
            parts_real = Xplot.reshape(-1, args.node_feat_size)
            parts_gen = gen_out.reshape(-1, args.node_feat_size)

    print(Xplot.shape)
    print(gen_out.shape)

    print(Xplot[0][:10])
    print(gen_out[0][:10])

    real_masses = []
    gen_masses = []

    for i in range(args.num_samples):
        jetv = LorentzVector()

        for j in range(args.num_hits):
            part = Xplot[i][j]
            if (not args.mask or part[3] > 0) and (not args.mask_manual or mask_real[i][j]):
                vec = LorentzVector()
                vec.setptetaphim(part[2], part[0], part[1], 0)
                jetv += vec

        real_masses.append(jetv.mass)

    for i in range(args.num_samples):
        jetv = LorentzVector()

        for j in range(args.num_hits):
            part = gen_out[i][j]
            if (not args.mask or part[3] > 0) and (not args.mask_manual or mask_gen[i][j]):
                vec = LorentzVector()
                vec.setptetaphim(part[2], part[0], part[1], 0)
                jetv += vec

        gen_masses.append(jetv.mass)

    fig = plt.figure(figsize=(30, 8))

    for i in range(3):
        fig.add_subplot(1, 4, i + 1)
        plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
        _ = plt.hist(parts_real[:, i], bins[i], histtype='step', label='Real', color='red')
        _ = plt.hist(parts_gen[:, i], bins[i], histtype='step', label='Generated', color='blue')
        plt.xlabel('Particle ' + labels[i])
        plt.ylabel('Number of Particles')
        # plt.title('JSD = ' + str(round(losses['jsdm'][-1][i], 3)) + ' ± ' + str(round(losses['jsdstd'][-1][i], 3)))
        plt.legend(loc=1, prop={'size': 18})

    binsm = np.arange(0, 0.225, 0.0045)

    fig.add_subplot(1, 4, 4)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    # plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
    _ = plt.hist(real_masses, bins=binsm, histtype='step', label='Real', color='red')
    _ = plt.hist(gen_masses, bins=binsm, histtype='step', label='Generated', color='blue')
    plt.xlabel('Jet $m/p_{T}$')
    plt.ylabel('Jets')
    plt.legend(loc=1, prop={'size': 18})

    name = args.name + "/" + str(epoch)
    plt.tight_layout(2.0)
    plt.savefig(args.figs_path + name + ".pdf", bbox_inches='tight')
    plt.close()

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

    # if args.jf:
    #     real_masses = []
    #     real_pts = []
    #
    #     gen_masses = []
    #     gen_pts = []
    #
    #     for i in range(args.num_samples):
    #         jetv = LorentzVector()
    #
    #         for part in Xplot[i]:
    #             vec = LorentzVector()
    #             vec.setptetaphim(part[2], part[0], part[1], 0)
    #             jetv += vec
    #
    #         real_masses.append(jetv.mass)
    #         real_pts.append(jetv.pt)
    #
    #     for i in range(args.num_samples):
    #         jetv = LorentzVector()
    #
    #         for part in gen_out[i]:
    #             vec = LorentzVector()
    #             vec.setptetaphim(part[2], part[0], part[1], 0)
    #             jetv += vec
    #
    #         gen_masses.append(jetv.mass)
    #         gen_pts.append(jetv.pt)
    #
    #     mass_bins = np.arange(0, 400, 4)
    #     pt_bins = np.arange(0, 3000, 30)
    #
    #     fig = plt.figure(figsize=(16, 8))
    #
    #     fig.add_subplot(1, 2, 1)
    #     plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    #     _ = plt.hist(real_masses, bins=mass_bins, histtype='step', label='real', color='red')
    #     _ = plt.hist(gen_masses, bins=mass_bins, histtype='step', label='real', color='blue')
    #     plt.xlabel('Jet Mass (GeV)')
    #     plt.ylabel('Jets')
    #     plt.legend(loc=1, prop={'size': 18})
    #
    #     fig.add_subplot(1, 2, 2)
    #     plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    #     _ = plt.hist(real_pts, bins=pt_bins, histtype='step', label='real', color='red')
    #     _ = plt.hist(gen_pts, bins=pt_bins, histtype='step', label='real', color='blue')
    #     plt.xlabel('Jet $p_T$ (GeV)')
    #     plt.ylabel('Jets')
    #     plt.legend(loc=1, prop={'size': 18})
    #
    #     plt.savefig(args.figs_path + name + "_mass_pt.pdf", bbox_inches='tight')
    #     plt.close()

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

    x = np.arange(epoch + 1, step=args.save_epochs)

    # plt.rcParams.update({'font.size': 12})
    # fig = plt.figure(figsize=(22, 5))

    # for i in range(3):
    #     fig.add_subplot(1, 3, i + 1)
    #     plt.plot(x, np.log10(np.array(losses['jsdm'])[:, i]))
    #     # plt.ylim((0, 5))
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Particle ' + labels[i] + ' LogJSD')
    # # plt.legend()
    # plt.savefig(args.losses_path + name + "_jsd.pdf", bbox_inches='tight')
    # plt.close()

    if args.w1 and epoch >= 5:
        x = np.arange(5, epoch + 1, 5)[-len(losses['w1_' + str(args.w1_num_samples[0]) + 'm']):]

        plt.rcParams.update({'font.size': 12})
        colors = ['blue', 'green', 'orange']

        fig = plt.figure(figsize=(30, 7))

        for i in range(3):
            fig.add_subplot(1, 3, i + 1)
            for k in range(len(args.w1_num_samples)):
                plt.plot(x, np.log10(np.array(losses['w1_' + str(args.w1_num_samples[k]) + 'm'])[:, i]), label=str(args.w1_num_samples[k]) + ' Jet Samples', color=colors[k])
                # plt.fill_between(x, np.log10(np.array(losses['w1_' + str(args.num_samples[k]) + 'm'])[:, i] - np.array(losses['w1_' + str(args.num_samples[k]) + 'std'])[:, i]), np.log10(np.array(losses['w1_' + str(args.num_samples[k]) + 'm'])[:, i] + np.array(losses['w1_' + str(args.num_samples[k]) + 'std'])[:, i]), color=colors[k], alpha=0.2)
                # plt.plot(x, np.ones(len(x)) * np.log10(realw1m[k][i]), '--', label=str(args.num_samples[k]) + ' Real W1', color=colors[k])
                # plt.fill_between(x, np.log10(np.ones(len(x)) * (realw1m[k][i] - realw1std[k][i])), np.log10(np.ones(len(x)) * (realw1m[k][i] + realw1std[k][i])), color=colors[k], alpha=0.2)
            plt.legend(loc=2, prop={'size': 11})
            plt.xlabel('Epoch')
            plt.ylabel('Particle ' + labels[i] + ' LogW1')

        plt.savefig(args.losses_path + name + "_w1.pdf", bbox_inches='tight')
        plt.close()

        if args.jf:
            x = np.arange(5, epoch + 1, 5)[-len(losses['w1j_' + str(args.w1_num_samples[0]) + 'm']):]
            fig = plt.figure(figsize=(20, 7))

            for i in range(2):
                fig.add_subplot(1, 2, i + 1)
                for k in range(len(args.w1_num_samples)):
                    plt.plot(x, np.log10(np.array(losses['w1j_' + str(args.w1_num_samples[k]) + 'm'])[:, i]), label=str(args.w1_num_samples[k]) + ' Jet Samples', color=colors[k])
                plt.legend(loc=2, prop={'size': 11})
                plt.xlabel('Epoch')
                plt.ylabel('Particle ' + labelsj[i] + ' LogW1')

            plt.savefig(args.losses_path + name + "_w1j.pdf", bbox_inches='tight')
            plt.close()

    for key in losses:
        np.savetxt(args.losses_path + args.name + "/" + key + '.txt', losses[key])

    try:
        remove(args.losses_path + args.name + "/" + str(epoch - args.save_epochs) + ".pdf")
        remove(args.losses_path + args.name + "/" + str(epoch - args.save_epochs) + "_w1.pdf")
        remove(args.losses_path + args.name + "/" + str(epoch - args.save_epochs) + "_w1j.pdf")
        # remove(args.losses_path + args.name + "/" + str(epoch - args.save_epochs) + "_fid.pdf")
    except:
        print("couldn't remove loss file")

    print("saved figs")


def save_models(args, D, G, optimizers, name, epoch):
    torch.save(D.state_dict(), args.model_path + args.name + "/D_" + str(epoch) + ".pt")
    torch.save(G.state_dict(), args.model_path + args.name + "/G_" + str(epoch) + ".pt")

    torch.save(optimizers[0].state_dict(), args.model_path + args.name + "/D_optim_" + str(epoch) + ".pt")
    torch.save(optimizers[1].state_dict(), args.model_path + args.name + "/G_optim_" + str(epoch) + ".pt")
