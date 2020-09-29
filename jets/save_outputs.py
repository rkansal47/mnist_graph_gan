import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
from os import remove
from scipy.spatial.distance import jensenshannon

plt.switch_backend('agg')


def save_sample_outputs(args, D, G, X, dist, name, epoch, losses):
    print("drawing figs")
    plt.rcParams.update({'font.size': 15})
    # if(args.fid): plt.suptitle("FID: " + str(losses['fid'][-1]))

    # noise = torch.load(args.noise_path + args.noise_file_name).to(args.device)

    G.eval()
    gen_out = utils.gen(args, G, dist=dist, num_samples=args.batch_size).cpu().detach().numpy()
    for i in range(int(args.num_samples / args.batch_size)):
        gen_out = np.concatenate((gen_out, utils.gen(args, G, dist=dist, num_samples=args.batch_size).cpu().detach().numpy()), 0)
    gen_out = gen_out[:args.num_samples]

    labels = ['$p_x$ (GeV)', '$p_y$ (GeV)', '$p_z$ (GeV)'] if args.coords == 'cartesian' else ['$\eta^{rel}$', '$\phi^{rel}$', '$p_T^{rel}$']
    # if args.coords

    if(args.coords == 'cartesian'):
        bin = np.arange(-500, 500, 10)
        bins = [bin, bin, bin]
    else:
        bins = [np.arange(-0.5, 0.5, 0.01), np.arange(-0.5, 0.5, 0.01), np.arange(0, 0.5, 0.005)]

    fig = plt.figure(figsize=(22, 7))
    # fig.suptitle("Particle Feature Distributions")

    if args.coords == 'cartesian':
        Xplot = X[:args.num_samples, :, :].cpu().detach().numpy() * args.maxp
        gen_out = gen_out * args.maxp
    else:
        Xplot = X[:args.num_samples, :, :].cpu().detach().numpy() * args.maxepp
        gen_out *= args.maxepp

    print(Xplot.shape)
    print(gen_out.shape)

    print(Xplot[0][:10])
    print(gen_out[0][:10])

    for i in range(3):
        fig.add_subplot(1, 3, i + 1)
        _ = plt.hist(Xplot[:, :, i].reshape(-1), bins[i], histtype='step', label='Real', color='red')
        _ = plt.hist(gen_out[:, :, i].reshape(-1), bins[i], histtype='step', label='Generated', color='blue')
        plt.xlabel('Particle ' + labels[i])
        plt.ylabel('Number of Particles')
        plt.title('JSD = ' + str(round(losses['jsdm'][-1][i], 3)) + ' ± ' + str(round(losses['jsdstd'][-1][i], 3)))
        plt.legend(loc=1, prop={'size': 11})

    name = args.name + "/" + str(epoch)

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
        # plt.plot(losses['D'], label='Disciriminative total loss')

    if(args.gp): plt.plot(losses['gp'], label='Gradient penalty')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(args.losses_path + name + ".pdf", bbox_inches='tight')
    plt.close()

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

    fig = plt.figure(figsize=(20, 5))

    for i in range(3):
        fig.add_subplot(1, 3, i + 1)
        plt.plot(x, np.log10(np.array(losses['jsdm'])[:, i]))
        # plt.ylim((0, 5))
        plt.xlabel('Epoch')
        plt.ylabel('Particle ' + labels[i] + ' LogJSD')
    # plt.legend()
    plt.savefig(args.losses_path + name + "_jsd.pdf", bbox_inches='tight')
    plt.close()

    if(args.gp): np.savetxt(args.losses_path + args.name + "/" + "gp.txt", losses['gp'])
    np.savetxt(args.losses_path + args.name + "/" + "D.txt", losses['D'])
    np.savetxt(args.losses_path + args.name + "/" + "G.txt", losses['G'])
    np.savetxt(args.losses_path + args.name + "/" + "Dr.txt", losses['Dr'])
    np.savetxt(args.losses_path + args.name + "/" + "Df.txt", losses['Df'])
    np.savetxt(args.losses_path + args.name + "/" + "jsdm.txt", np.array(losses['jsdm']))
    np.savetxt(args.losses_path + args.name + "/" + "jsdstd.txt", np.array(losses['jsdstd']))
    if args.fid: np.savetxt(args.losses_path + args.name + "/" + "fid.txt", losses['fid'])

    try:
        remove(args.losses_path + args.name + "/" + str(epoch - args.save_epochs) + ".pdf")
        remove(args.losses_path + args.name + "/" + str(epoch - args.save_epochs) + "_jsd.pdf")
        remove(args.losses_path + args.name + "/" + str(epoch - args.save_epochs) + "_fid.pdf")
    except:
        print("couldn't remove loss file")

    print("saved figs")


def save_models(args, D, G, optimizers, name, epoch):
    torch.save(D, args.model_path + args.name + "/D_" + str(epoch) + ".pt")
    torch.save(G, args.model_path + args.name + "/G_" + str(epoch) + ".pt")

    torch.save(optimizers[0].state_dict(), args.model_path + args.name + "/D_optim_" + str(epoch) + ".pt")
    torch.save(optimizers[1].state_dict(), args.model_path + args.name + "/G_optim_" + str(epoch) + ".pt")
