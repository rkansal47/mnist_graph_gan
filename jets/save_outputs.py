import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
from os import remove
import sys

plt.switch_backend('agg')


def save_sample_outputs(args, D, G, X, dist, name, epoch, losses):
    print("drawing figs")
    fig = plt.figure(figsize=(10, 10))
    if(args.fid): plt.suptitle("FID: " + str(losses['fid'][-1]))

    # noise = torch.load(args.noise_path + args.noise_file_name).to(args.device)

    gen_out = utils.gen(args, G, dist=dist, num_samples=args.batch_size).cpu().detach().numpy()
    for i in range(int(args.num_samples / args.batch_size)):
        gen_out = np.concatenate((gen_out, utils.gen(args, G, dist=dist, num_samples=args.batch_size).cpu().detach().numpy()), 0)
    gen_out = gen_out[:args.num_samples]

    labels = ['$p_x$', '$p_y$', '$p_z$'] if args.coords == 'cartesian' else ['$\eta$', '$\phi$', '$p_T$']

    fig = plt.figure(figsize=(20, 5))

    print(gen_out.shape)
    print(X[:args.num_samples].shape)

    print(gen_out[0, :, 0])
    print(X[0, :, 0])

    bins = np.arange(-0.2, 0.2, 0.002)

    for i in range(3):
        fig.add_subplot(1, 3, i + 1)
        _ = plt.hist(X[:args.num_samples, :, i].reshape(-1).cpu().detach().numpy(), bins, histtype='step', label='real', color='red')
        _ = plt.hist(gen_out[:, :, i].reshape(-1), bins, histtype='step', label='generated', color='blue')
        plt.xlabel('particle ' + labels[i])
        plt.legend(loc=1, prop={'size': 7})

    name = args.name + "/" + str(epoch)

    plt.savefig(args.figs_path + name + ".png")
    plt.close()

    sys.exit()

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
    plt.savefig(args.losses_path + name + ".png")
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
        plt.savefig(args.losses_path + name + "_fid.png")
        plt.close()

    if(args.gp): np.savetxt(args.losses_path + args.name + "/" + "gp.txt", losses['gp'])
    np.savetxt(args.losses_path + args.name + "/" + "D.txt", losses['D'])
    np.savetxt(args.losses_path + args.name + "/" + "G.txt", losses['G'])
    np.savetxt(args.losses_path + args.name + "/" + "Dr.txt", losses['Dr'])
    np.savetxt(args.losses_path + args.name + "/" + "Df.txt", losses['Df'])
    if args.fid: np.savetxt(args.losses_path + args.name + "/" + "fid.txt", losses['fid'])

    try:
        remove(args.losses_path + args.name + "/" + str(epoch - 5) + ".png")
        remove(args.losses_path + args.name + "/" + str(epoch - 5) + "_fid.png")
    except:
        print("couldn't remove loss file")

    print("saved figs")


def save_models(args, D, G, optimizers, name, epoch, k=-1, j=-1):
    g_only = "_g_only_" + str(k) + "_" + str(j) if j > -1 else ""
    torch.save(D, args.model_path + args.name + "/D_" + str(epoch) + g_only + ".pt")
    torch.save(G, args.model_path + args.name + "/G_" + str(epoch) + g_only + ".pt")

    torch.save(optimizers[0].state_dict(), args.model_path + args.name + "/D_optim_" + str(epoch) + g_only + ".pt")
    torch.save(optimizers[1].state_dict(), args.model_path + args.name + "/G_optim_" + str(epoch) + g_only + ".pt")
