import numpy as np
import torch
from skimage.draw import draw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import utils
from os import remove

plt.switch_backend('agg')


def draw_graph(graph, node_r, im_px):
    imd = im_px + node_r
    img = np.zeros((imd, imd), dtype=np.float)

    circles = []
    for node in graph:
        circles.append((draw.circle_perimeter(int(node[1]), int(node[0]), node_r), draw.circle(int(node[1]), int(node[0]), node_r), node[2]))

    for circle in circles:
        img[circle[1]] = circle[2]

    return img


def save_sample_outputs(args, D, G, dist, name, epoch, losses, k=-1, j=-1):
    print("drawing figs")
    fig = plt.figure(figsize=(10, 10))

    num_ims = 100

    gen_out = utils.gen(args, G, dist, args.batch_size, disp=True).cpu().detach().numpy()

    for i in range(int(num_ims / args.batch_size)):
        gen_out = np.concatenate((gen_out, utils.gen(args, G, dist, args.batch_size, disp=True).cpu().detach().numpy()), 0)

    gen_out = gen_out[:num_ims]

    # print(gen_out)

    if(args.sparse_mnist):
        gen_out = gen_out * [28, 28, 1] + [14, 14, 1]

        for i in range(1, num_ims + 1):
            fig.add_subplot(10, 10, i)
            im_disp = np.zeros((28, 28)) - 0.5

            im_disp += np.min(gen_out[i - 1])

            for x in gen_out[i - 1]:
                x0 = int(round(x[0])) if x[0] < 27 else 27
                x0 = x0 if x0 > 0 else 0

                x1 = int(round(x[1])) if x[1] < 27 else 27
                x1 = x1 if x1 > 0 else 0

                im_disp[x1, x0] = x[2]

            plt.imshow(im_disp, cmap=cm.gray_r, interpolation='nearest')
            plt.axis('off')
    else:
        node_r = 30
        im_px = 1000

        gen_out[gen_out > 0.47] = 0.47
        gen_out[gen_out < -0.5] = -0.5

        gen_out = gen_out * [im_px, im_px, 1] + [(im_px + node_r) / 2, (im_px + node_r) / 2, 0.55]

        for i in range(1, num_ims + 1):
            fig.add_subplot(10, 10, i)
            im_disp = draw_graph(gen_out[i - 1], node_r, im_px)
            plt.imshow(im_disp, cmap=cm.gray_r, interpolation='nearest')
            plt.axis('off')

    g_only = "_g_only_" + str(k) + "_" + str(j) if j > -1 else ""
    name = args.name + "/" + str(epoch) + g_only

    plt.savefig(args.figs_path + name + ".png")
    plt.close()

    plt.figure()

    if(args.loss == "og" or args.loss == "ls"):
        plt.plot(losses['Dr'], label='Discriminitive real loss')
        plt.plot(losses['Df'], label='Discriminitive fake loss')
        if not args.optimizer == 'acgd': plt.plot(losses['G'], label='Generative loss')
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

    fid_5 = losses['fid'][::5]
    x = np.arange(len(losses['fid']), step=5)

    plt.figure()
    plt.plot(x, np.log(fid_5))
    # plt.ylim((0, 5))
    plt.xlabel('Epoch')
    plt.ylabel('LogFID')
    # plt.legend()
    plt.savefig(args.losses_path + name + "_fid.png")
    plt.close()

    if(args.gp): np.savetxt(args.losses_path + args.name + "/" + "gp.txt", losses['gp'])
    np.savetxt(args.losses_path + args.name + "/" + "D.txt", losses['D'])
    np.savetxt(args.losses_path + args.name + "/" + "G.txt", losses['G'])
    np.savetxt(args.losses_path + args.name + "/" + "Dr.txt", losses['Dr'])
    np.savetxt(args.losses_path + args.name + "/" + "Df.txt", losses['Df'])
    np.savetxt(args.losses_path + args.name + "/" + "fid.txt", losses['fid'])

    try:
        if(j == -1):
            remove(args.losses_path + args.name + "/" + str(epoch - 5) + ".png")
            remove(args.losses_path + args.name + "/" + str(epoch - 5) + "_fid.png")
        else: remove(args.losses_path + args.name + "/" + str(epoch) + "_g_only_" + str(k) + "_" + str(j - 5) + ".png")
    except:
        print("couldn't remove loss file")

    print("saved figs")


def save_models(args, D, G, optimizers, name, epoch, k=-1, j=-1):
    g_only = "_g_only_" + str(k) + "_" + str(j) if j > -1 else ""
    torch.save(D, args.model_path + args.name + "/D_" + str(epoch) + g_only + ".pt")
    torch.save(G, args.model_path + args.name + "/G_" + str(epoch) + g_only + ".pt")
    if(args.optimizer == 'acgd'):
        torch.save(optimizers.state_dict(), args.model_path + args.name + "/optim_" + str(epoch) + g_only + ".pt")
    else:
        torch.save(optimizers[0].state_dict(), args.model_path + args.name + "/D_optim_" + str(epoch) + g_only + ".pt")
        torch.save(optimizers[1].state_dict(), args.model_path + args.name + "/G_optim_" + str(epoch) + g_only + ".pt")
