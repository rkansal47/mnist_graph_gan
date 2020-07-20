# import setGPU

import torch
from model import Graph_GAN, MoNet, GaussianGenerator  # , Graph_Generator, Graph_Discriminator, Gaussian_Discriminator
from superpixels_dataset import SuperpixelsDataset
from graph_dataset_mnist import MNISTGraphDataset
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from skimage.draw import draw

import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

from os import listdir, mkdir, remove
from os.path import exists, dirname, realpath

import sys
from copy import deepcopy

from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader as tgDataLoader
from torch_geometric.data import Batch, Data

plt.switch_backend('agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


def main(args):
    # global D
    torch.manual_seed(4)
    torch.autograd.set_detect_anomaly(True)

    name = [args.name]
    if args.wgan:
        name.append('wgan')
    if args.gru:
        name.append('gru')
    if args.gcnn:
        name.append('gcnn')
    if args.gom:
        name.append('g_only_mode')
    if args.sparse_mnist:
        name.append('sparse_mnist')

    # name.append('num_iters_{}'.format(args.num_iters))
    # name.append('num_critic_{}'.format(args.num_critic))
    args.name = '_'.join(name)

    args.model_path = args.dir_path + '/models/'
    args.losses_path = args.dir_path + '/losses/'
    args.args_path = args.dir_path + '/args/'
    args.figs_path = args.dir_path + '/figs/'
    args.dataset_path = args.dir_path + '/raw/' if not args.sparse_mnist else args.dir_path + '/mnist_dataset/'
    args.err_path = args.dir_path + '/err/'

    if(not exists(args.model_path)):
        mkdir(args.model_path)
    if(not exists(args.losses_path)):
        mkdir(args.losses_path)
    if(not exists(args.args_path)):
        mkdir(args.args_path)
    if(not exists(args.figs_path)):
        mkdir(args.figs_path)
    if(not exists(args.err_path)):
        mkdir(args.err_path)
    if(not exists(args.dataset_path)):
        mkdir(args.dataset_path)
        print("Downloading dataset")
        if(not args.sparse_mnist):
            import tarfile, urllib
            url = 'http://ls7-www.cs.uni-dortmund.de/cvpr_geometric_dl/mnist_superpixels.tar.gz'
            try:
                # python2
                file_tmp = urllib.urlretrieve(url)[0]
            except:
                # python3
                file_tmp = urllib.request.urlretrieve(url)[0]

            tar = tarfile.open(file_tmp)
            tar.extractall(args.dataset_path)
        else:
            import requests
            r = requests.get('https://pjreddie.com/media/files/mnist_train.csv', allow_redirects=True)
            open(args.dataset_path + 'mnist_train.csv', 'wb').write(r.content)
            r = requests.get('https://pjreddie.com/media/files/mnist_test.csv', allow_redirects=True)
            open(args.dataset_path + 'mnist_test.csv', 'wb').write(r.content)

        print("Downloaded dataset")

    prev_models = [f[:-4] for f in listdir(args.args_path)]  # removing .txt

    if (args.name in prev_models):
        print("name already used")
        # if(not args.load_model):
        #    sys.exit()
    else:
        mkdir(args.losses_path + args.name)
        mkdir(args.model_path + args.name)
        mkdir(args.figs_path + args.name)

    if(not args.load_model):
        f = open(args.args_path + args.name + ".txt", "w+")
        f.write(str(vars(args)))
        f.close()
    else:
        temp = args.start_epoch, args.num_epochs
        f = open(args.args_path + args.name + ".txt", "r")
        args_dict = vars(args)
        load_args_dict = eval(f.read())
        for key in load_args_dict:
            args_dict[key] = load_args_dict[key]

        args = objectview(args_dict)
        f.close()
        args.load_model = True
        args.start_epoch, args.num_epochs = temp

    def pf(data):
        return data.y == args.num

    pre_filter = pf if args.num != -1 else None

    print("loading")

    if(args.sparse_mnist):
        print("train")
        print(args.train)
        X = MNISTGraphDataset(args.dataset_path, args.num_hits, train=args.train, num=args.num)
        X_loaded = DataLoader(X, shuffle=True, batch_size=args.batch_size, pin_memory=True)
    else:
        if(args.gcnn):
            X = MNISTSuperpixels(args.dir_path, train=args.train, pre_transform=T.Cartesian(), pre_filter=pre_filter)
            X_loaded = tgDataLoader(X, shuffle=True, batch_size=args.batch_size)
        else:
            X = SuperpixelsDataset(args.dataset_path, args.num_hits, train=args.train, num=args.num)
            X_loaded = DataLoader(X, shuffle=True, batch_size=args.batch_size, pin_memory=True)

    print("loaded")

    if(args.load_model):
        G = torch.load(args.model_path + args.name + "/G_" + str(args.start_epoch) + ".pt")
        D = torch.load(args.model_path + args.name + "/D_" + str(args.start_epoch) + ".pt")
    else:
        # G = Graph_Generator(args.node_feat_size, args.fe_hidden_size, args.fe_out_size, args.fn_hidden_size, args.fn_num_layers, args.mp_iters_gen, args.num_hits, args.gen_dropout, args.leaky_relu_alpha, hidden_node_size=args.hidden_node_size, int_diffs=args.int_diffs, pos_diffs=args.pos_diffs, gru=args.gru, batch_norm=args.batch_norm, device=device).to(device)
        if(args.gcnn):
            G = GaussianGenerator(args=deepcopy(args)).to(device)
            D = MoNet(args=deepcopy(args)).to(device)
            # D = Gaussian_Discriminator(args.node_feat_size, args.fe_hidden_size, args.fe_out_size, args.mp_hidden_size, args.mp_num_layers, args.num_iters, args.num_hits, args.dropout, args.leaky_relu_alpha, kernel_size=args.kernel_size, hidden_node_size=args.hidden_node_size, int_diffs=args.int_diffs, gru=GRU, batch_norm=args.batch_norm, device=device).to(device)
        else:
            # D = Graph_Discriminator(args.node_feat_size, args.fe_hidden_size, args.fe_out_size, args.fn_hidden_size, args.fn_num_layers, args.mp_iters_disc, args.num_hits, args.disc_dropout, args.leaky_relu_alpha, hidden_node_size=args.hidden_node_size, wgan=args.wgan, int_diffs=args.int_diffs, pos_diffs=args.pos_diffs, gru=args.gru, batch_norm=args.batch_norm, device=device).to(device)
            print("Generator")
            G = Graph_GAN(gen=True, args=deepcopy(args)).to(device)
            print("Discriminator")
            D = Graph_GAN(gen=False, args=deepcopy(args)).to(device)

    print("Models loaded")

    if(args.optimizer == 'rmsprop'):
        G_optimizer = optim.RMSprop(G.parameters(), lr=args.lr_gen)
        D_optimizer = optim.RMSprop(D.parameters(), lr=args.lr_disc)
    elif(args.optimizer == 'adadelta'):
        G_optimizer = optim.Adadelta(G.parameters(), lr=args.lr_gen)
        D_optimizer = optim.Adadelta(D.parameters(), lr=args.lr_disc)
    else:
        G_optimizer = optim.Adam(G.parameters(), lr=args.lr_gen, weight_decay=5e-4, betas=(args.beta1, args.beta2))
        D_optimizer = optim.Adam(D.parameters(), lr=args.lr_disc, weight_decay=5e-4, betas=(args.beta1, args.beta2))

    print("optimizers loaded")

    normal_dist = Normal(torch.tensor(0.).to(device), torch.tensor(args.sd).to(device))

    if(not args.wgan):
        Y_real = torch.ones(args.batch_size, 1).to(device)
        Y_fake = torch.zeros(args.batch_size, 1).to(device)

    def wasserstein_loss(y_out, y_true):
        return -torch.mean(y_out * y_true)

    if(args.wgan):
        criterion = wasserstein_loss
    else:
        criterion = torch.nn.MSELoss()
        # if(LSGAN):
        #     criterion = torch.nn.MSELoss()
        # else:
        #     criterion = torch.nn.BCELoss()

    # print(criterion(torch.tensor([1.0]),torch.tensor([-1.0])))

    def gen(num_samples, noise=0, disp=False):
        if(noise == 0):
            if(args.gcnn):
                rand = normal_dist.sample((num_samples * 5, 2 + args.channels[0]))
                noise = Data(pos=rand[:, :2], x=rand[:, 2:])
            else:
                noise = normal_dist.sample((num_samples, args.num_hits, args.hidden_node_size))

        gen_data = G(noise)

        if args.gcnn and disp: return torch.cat((gen_data.pos, gen_data.x), 1).view(num_samples, 75, 3)
        return gen_data

    # transform my format to torch_geometric's
    def tg_transform(X):
        batch_size = X.size(0)

        pos = X[:, :, :2]

        x1 = pos.repeat(1, 1, 75).reshape(batch_size, 75 * 75, 2)
        x2 = pos.repeat(1, 75, 1)

        diff_norms = torch.norm(x2 - x1 + 1e-12, dim=2)

        # diff = x2-x1
        # diff = diff[diff_norms < args.cutoff]

        norms = diff_norms.reshape(batch_size, 75, 75)
        neighborhood = torch.nonzero(norms < args.cutoff, as_tuple=False)
        # diff = diff[neighborhood[:, 1] != neighborhood[:, 2]]

        neighborhood = neighborhood[neighborhood[:, 1] != neighborhood[:, 2]]  # remove self-loops
        unique, counts = torch.unique(neighborhood[:, 0], return_counts=True)
        # edge_slices = torch.cat((torch.tensor([0]).to(device), counts.cumsum(0)))
        edge_index = (neighborhood[:, 1:] + (neighborhood[:, 0] * 75).view(-1, 1)).transpose(0, 1)

        # normalizing edge attributes
        # edge_attr_list = list()
        # for i in range(batch_size):
        #     start_index = edge_slices[i]
        #     end_index = edge_slices[i + 1]
        #     temp = diff[start_index:end_index]
        #     max = torch.max(temp)
        #     temp = temp/(2 * max + 1e-12) + 0.5
        #     edge_attr_list.append(temp)
        #
        # edge_attr = torch.cat(edge_attr_list)

        # edge_attr = diff/(2 * args.cutoff) + 0.5

        x = X[:, :, 2].reshape(batch_size * 75, 1) + 0.5
        pos = 28 * pos.reshape(batch_size * 75, 2) + 14

        row, col = edge_index
        edge_attr = (pos[col] - pos[row]) / (2 * 28 * args.cutoff) + 0.5

        zeros = torch.zeros(batch_size * 75, dtype=int).to(device)
        zeros[torch.arange(batch_size) * 75] = 1
        batch = torch.cumsum(zeros, 0) - 1

        return Batch(batch=batch, x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr, y=None, pos=pos)

    def draw_graph(graph, node_r, im_px):
        imd = im_px + node_r
        img = np.zeros((imd, imd), dtype=np.float)

        circles = []
        for node in graph:
            circles.append((draw.circle_perimeter(int(node[1]), int(node[0]), node_r), draw.circle(int(node[1]), int(node[0]), node_r), node[2]))

        for circle in circles:
            img[circle[1]] = circle[2]

        return img

    def save_sample_outputs(name, epoch, drlosses, dflosses, glosses, k=-1, j=-1):
        print("drawing figs")
        fig = plt.figure(figsize=(10, 10))

        num_ims = 100

        gen_out = gen(args.batch_size, disp=True).cpu().detach().numpy()

        for i in range(int(num_ims / args.batch_size)):
            gen_out = np.concatenate((gen_out, gen(args.batch_size, disp=True).cpu().detach().numpy()), 0)

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
        plt.plot(drlosses, label='Discriminitive real loss')
        plt.plot(dflosses, label='Discriminitive fake loss')
        plt.plot(glosses, label='Generative loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(args.losses_path + name + ".png")
        plt.close()

        np.savetxt(args.losses_path + args.name + "/" + "G.txt", glosses)
        np.savetxt(args.losses_path + args.name + "/" + "Dr.txt", drlosses)
        np.savetxt(args.losses_path + args.name + "/" + "Df.txt", dflosses)

        try:
            if(j == -1): remove(args.losses_path + args.name + "/" + str(epoch - 5) + ".png")
            else: remove(args.losses_path + args.name + "/" + str(epoch) + "_g_only_" + str(k) + "_" + str(j - 5) + ".png")
        except:
            print("couldn't remove loss file")

        print("saved figs")

    def save_models(name, epoch, k=-1, j=-1):
        g_only = "_g_only_" + str(k) + "_" + str(j) if j > -1 else ""
        torch.save(G, args.model_path + args.name + "/G_" + str(epoch) + g_only + ".pt")
        torch.save(D, args.model_path + args.name + "/D_" + str(epoch) + g_only + ".pt")

    # from https://github.com/EmilienDupont/wgan-gp
    def gradient_penalty(real_data, generated_data, batch_size):
        # Calculate interpolation
        if(not args.gcnn):
            alpha = torch.rand(batch_size, 1, 1).to(device)
            alpha = alpha.expand_as(real_data)
            interpolated = alpha * real_data + (1 - alpha) * generated_data
            interpolated = Variable(interpolated, requires_grad=True).to(device)
        else:
            alpha = torch.rand(batch_size, 1, 1).to(device)
            alpha_x = alpha.expand((batch_size, 75, 1))
            interpolated_x = alpha_x * real_data.x.reshape(batch_size, 75, 1) + (1 - alpha_x) * generated_data.x.reshape(batch_size, 75, 1)
            alpha_pos = alpha.expand((batch_size, 75, 2))
            interpolated_pos = alpha_pos * real_data.pos.reshape(batch_size, 75, 2) + (1 - alpha_pos) * generated_data.pos.reshape(batch_size, 75, 2)
            interpolated_X = Variable(torch.cat(((interpolated_pos - 14) / 28, interpolated_x - 0.5), dim=2), requires_grad=True)
            interpolated = tg_transform(interpolated_X)

        del alpha
        torch.cuda.empty_cache()

        # Calculate probability of interpolated examples
        prob_interpolated = D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        if(not args.gcnn):
            gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones(prob_interpolated.size()).to(device), create_graph=True, retain_graph=True, allow_unused=True)[0].to(device)
        if(args.gcnn):
            gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated_X, grad_outputs=torch.ones(prob_interpolated.size()).to(device), create_graph=True, retain_graph=True, allow_unused=True)[0].to(device)

        gradients = gradients.contiguous()

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        gp = args.gp_weight * ((gradients_norm - 1) ** 2).mean()
        # print("gradient penalty")
        # print(gp)
        return gp

    def convert_to_batch(data, batch_size):
        zeros = torch.zeros(batch_size * 75, dtype=int).to(device)
        zeros[torch.arange(batch_size) * 75] = 1
        batch = torch.cumsum(zeros, 0) - 1

        return Batch(batch=batch, x=data.x, pos=data.pos, edge_index=data.edge_index, edge_attr=data.edge_attr)

    def train_D(data, gen_data=None, unrolled=False):
        # print("dtrain")
        D.train()
        D_optimizer.zero_grad()

        run_batch_size = data.shape[0] if not args.gcnn else data.y.shape[0]

        # print("real")
        # print(D_real_output)

        if gen_data is None:
            gen_data = gen(run_batch_size)
            if(args.gcnn): gen_data = convert_to_batch(gen_data, run_batch_size)

        # print("fake")
        # print(D_fake_output)

        if args.label_smoothing:
            Y_real = torch.empty(run_batch_size).uniform_(0.7, 1.2).to(device)
            Y_fake = torch.empty(run_batch_size).uniform_(0.0, 0.3).to(device)
        else:
            Y_real = torch.ones(args.batch_size, 1).to(device)
            Y_fake = torch.zeros(args.batch_size, 1).to(device)

        # randomly flipping labels for D
        Y_real[torch.rand(run_batch_size) < args.label_noise] = 0
        Y_fake[torch.rand(run_batch_size) < args.label_noise] = 1

        if(args.wgan):
            # want reals > 0 and fakes < 0
            D_real_output = D(data.clone())
            D_real_loss = -D_real_output.mean()
            D_real_loss.backward(create_graph=unrolled)

            D_fake_output = D(gen_data)
            D_fake_loss = D_fake_output.mean()
            D_fake_loss.backward(create_graph=unrolled)
            # D_loss = D_real_loss + D_fake_loss + gradient_penalty(data, gen_data, run_batch_size)
            gp = gradient_penalty(data, gen_data, run_batch_size)
            gp.backward(create_graph=unrolled)
        else:
            D_real_output = D(data.clone())
            D_real_loss = criterion(D_real_output, Y_real[:run_batch_size])

            D_fake_output = D(gen_data)
            D_fake_loss = criterion(D_fake_output, Y_fake[:run_batch_size])

            D_loss = D_real_loss + D_fake_loss

            D_loss.backward(create_graph=unrolled)

        # print("real outputs")
        # print(D_real_output[:5])
        #
        # print("fake outputs")
        # print(D_fake_output[:5])

        D_optimizer.step()

        return (D_real_loss.item(), D_fake_loss.item())

    def train_G(data):
        # global D
        # print("gtrain")
        G.train()
        G_optimizer.zero_grad()

        gen_data = gen(args.batch_size)
        if(args.gcnn): gen_data = convert_to_batch(gen_data, args.batch_size)

        if(args.unrolled_steps > 0):
            D_backup = deepcopy(D)
            for i in range(args.unrolled_steps - 1):
                train_D(data, gen_data=gen_data, unrolled=True)

        D_fake_output = D(gen_data)

        # print(D_fake_output)

        if(args.wgan):
            G_loss = -D_fake_output.mean()
        else:
            G_loss = criterion(D_fake_output, Y_real)

        G_loss.backward()
        G_optimizer.step()

        # D.assigntest(5)
        # print("in unrolled D")
        # D.printtest()

        if(args.unrolled_steps > 0):
            D.load(D_backup)

        return G_loss.item()

    if(args.load_model):
        G_losses = np.loadtxt(args.losses_path + args.name + "/" + "G.txt").tolist()[:args.start_epoch]
        # backwards compatibility
        try:
            Dr_losses = np.loadtxt(args.losses_path + args.name + "/" + "Dr.txt").tolist()[:args.start_epoch]
            Df_losses = np.loadtxt(args.losses_path + args.name + "/" + "Df.txt").tolist()[:args.start_epoch]
        except:
            D_losses = np.loadtxt(args.losses_path + args.name + "/" + "D.txt").tolist()[:args.start_epoch]
            Dr_losses, Df_losses = D_losses / 2
    else:
        Dr_losses = []
        Df_losses = []
        G_losses = []

    if(args.save_zero):
        save_sample_outputs(args.name, 0, Dr_losses, Df_losses, G_losses)

    def train():
        k = 0
        temp_ng = args.num_gen
        for i in range(args.start_epoch, args.num_epochs):
            print("Epoch %d %s" % ((i + 1), args.name))
            Dr_loss = 0
            Df_loss = 0
            G_loss = 0
            lenX = len(X_loaded)
            for batch_ndx, data in tqdm(enumerate(X_loaded), total=lenX):
                data = data.to(device)
                if(args.gcnn):
                    data.pos = (data.pos - 14) / 28
                    row, col = data.edge_index
                    data.edge_attr = (data.pos[col] - data.pos[row]) / (2 * args.cutoff) + 0.5

                if(args.num_critic > 1):
                    D_loss = train_D(data)
                    Dr_loss += D_loss[0]
                    Df_loss += D_loss[1]

                    if((batch_ndx - 1) % args.num_critic == 0):
                        G_loss += train_G(data)
                else:
                    if(batch_ndx == 0 or (batch_ndx - 1) % args.num_gen == 0):
                        D_loss = train_D(data)
                        Dr_loss += D_loss[0]
                        Df_loss += D_loss[1]

                    G_loss += train_G(data)

                # if(batch_ndx == 10):
                #     return

            if(args.num_critic > 1):
                Dr_losses.append(Dr_loss / lenX)
                Df_losses.append(Df_loss / lenX)
                G_losses.append(G_loss / (lenX / args.num_critic))
            else:
                Dr_losses.append((Dr_loss) / (lenX / args.num_gen))
                Df_losses.append((Df_loss) / (lenX / args.num_gen))
                G_losses.append(G_loss / lenX)

            print("g loss: " + str(G_losses[-1]))
            print("dr loss: " + str(Dr_losses[-1]))
            print("df loss: " + str(Df_losses[-1]))

            gloss = G_losses[-1]
            drloss = Dr_losses[-1]
            dfloss = Df_losses[-1]
            dloss = (drloss + dfloss) / 2
            bag = 0.1

            if(args.bgm):
                if(i > 20 and gloss > dloss + bag):
                    print("num gen upping to 10")
                    args.num_gen = 10
                else:
                    print("num gen normal")
                    args.num_gen = temp_ng
            elif(args.gom):
                if(i > 20 and gloss > dloss + bag):
                    print("G loss too high - training G only")
                    j = 0
                    print("starting g loss: " + str(gloss))
                    print("starting d loss: " + str(dloss))

                    while(gloss > dloss + bag * 0.5):
                        print(j)
                        gloss = 0
                        for l in tqdm(range(lenX)):
                            gloss += train_G()

                        gloss /= lenX
                        print("g loss: " + str(gloss))
                        print("d loss: " + str(dloss))

                        G_losses.append(gloss)
                        Dr_losses.append(drloss)
                        Df_losses.append(dfloss)

                        if(j % 5 == 0):
                            save_sample_outputs(args.name, i + 1, Dr_losses, Df_losses, G_losses, k, j)

                        j += 1

                    k += 1
            elif(args.rd):
                if(i > 20 and gloss > dloss + bag):
                    print("gloss too high, resetting D params")
                    D.reset_params()

            if((i + 1) % 5 == 0):
                save_sample_outputs(args.name, i + 1, Dr_losses, Df_losses, G_losses)

            if((i + 1) % 5 == 0):
                save_models(args.name, i + 1)

    train()


def add_bool_arg(parser, name, help, default=False):
    varname = '_'.join(name.split('-'))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=varname, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=varname, action='store_false', help="don't " + help)
    parser.set_defaults(**{varname: default})


def parse_args():
    import argparse

    dir_path = dirname(realpath(__file__))

    parser = argparse.ArgumentParser()

    add_bool_arg(parser, "load-model", "load a pretrained model", default=False)
    add_bool_arg(parser, "save-zero", "save the initial figure", default=False)
    add_bool_arg(parser, "wgan", "use wgan (instead of LSGAN)", default=False)
    add_bool_arg(parser, "gcnn", "use gcnn", default=False)
    add_bool_arg(parser, "gru", "use GRUs", default=False)
    add_bool_arg(parser, "gom", "use gen only mode", default=False)
    add_bool_arg(parser, "bgm", "use boost g mode", default=False)
    add_bool_arg(parser, "rd", "use restart d mode", default=False)

    add_bool_arg(parser, "dea", "use early averaging discriminator", default=False)

    add_bool_arg(parser, "label-smoothing", "use label smotthing with discriminator", default=False)
    add_bool_arg(parser, "sparse-mnist", "use sparse mnist dataset (as opposed to superpixels)", default=False)

    add_bool_arg(parser, "n", "run on nautilus cluster", default=False)

    parser.add_argument("--start-epoch", type=int, default=0, help="which epoch to start training on (only makes sense if loading a model)")

    parser.add_argument("--dir-path", type=str, default=dir_path, help="path where dataset and output will be stored")

    parser.add_argument("--node-feat-size", type=int, default=3, help="node feature size")
    parser.add_argument("--hidden-node-size", type=int, default=32, help="latent vector size of each node (incl node feature size)")

    # parser.add_argument("--fe-hidden-size", type=int, default=128, help="edge network hidden layer size")
    # parser.add_argument("--fe-out-size", type=int, default=256, help="edge network out size")
    #
    # parser.add_argument("--fn-hidden-size", type=int, default=256, help="message passing hidden layers sizes")
    # parser.add_argument("--fn-num-layers", type=int, default=2, help="message passing number of layers in generator")

    parser.add_argument("--fn", type=int, nargs='*', default=[256, 256], help="hidden fn layers e.g. 256 256")
    parser.add_argument("--fe", type=int, nargs='+', default=[96, 128, 192], help="hidden and output fe layers e.g. 64 128")
    parser.add_argument("--fnd", type=int, nargs='*', default=[128, 256], help="hidden disc output layers e.g. 256 128")

    parser.add_argument("--disc-dropout", type=float, default=0.5, help="fraction of discriminator dropout")
    parser.add_argument("--gen-dropout", type=float, default=0, help="fraction of generator dropout")

    parser.add_argument("--mp-iters-gen", type=int, default=2, help="number of message passing iterations in the generator")
    parser.add_argument("--mp-iters-disc", type=int, default=2, help="number of message passing iterations in the discriminator (if applicable)")

    parser.add_argument("--leaky-relu-alpha", type=float, default=0.2, help="leaky relu alpha")
    parser.add_argument("--num-hits", type=int, default=75, help="number of hits")
    parser.add_argument("--num-epochs", type=int, default=2000, help="number of epochs to train")

    parser.add_argument("--lr-disc", type=float, default=1e-4, help="learning rate discriminator")
    parser.add_argument("--lr-gen", type=float, default=1e-4, help="learning rate generator")

    parser.add_argument("--num-critic", type=int, default=1, help="number of critic updates for each generator update")
    parser.add_argument("--num-gen", type=int, default=1, help="number of generator updates for each critic update (num-critic must be 1 for this to apply)")

    parser.add_argument("--kernel-size", type=int, default=25, help="graph convolutional layer kernel size")
    parser.add_argument("--num", type=int, default=3, help="number to train on")
    parser.add_argument("--sd", type=float, default=0.2, help="standard deviation of noise")

    parser.add_argument("--label-noise", type=float, default=0, help="discriminator label noise (between 0 and 1)")
    parser.add_argument("--unrolled-steps", type=int, default=0, help="number of unrolled D steps for G training")

    parser.add_argument("--batch-size", type=int, default=10, help="batch size")
    parser.add_argument("--gp-weight", type=float, default=10, help="WGAN generator penalty weight")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam optimizer beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam optimizer beta2")
    parser.add_argument("--name", type=str, default="test", help="name or tag for model; will be appended with other info")

    parser.add_argument("--cutoff", type=float, default=0.32178, help="cutoff edge distance")  # found empirically to match closest to Superpixels

    parser.add_argument("--optimizer", type=str, default="None", help="optimizer - options are adam, rmsprop, adadelta")  # found empirically to match closest to Superpixels

    add_bool_arg(parser, "int-diffs", "use int diffs", default=False)
    add_bool_arg(parser, "pos-diffs", "use pos diffs", default=True)

    add_bool_arg(parser, "batch-norm", "use batch normalization", default=False)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--train', dest='train', action='store_true', help=help)
    group.add_argument('--test', dest='train', action='store_false', help=help)
    parser.set_defaults(**{'train': True})

    args = parser.parse_args()

    if(args.int_diffs and not args.pos_diffs):
        print("int_diffs = true and pos_diffs = false not supported yet")
        sys.exit()

    if(args.gru):
        print("GRU not supported anymore")
        sys.exit()

    if(args.n):
        args.dir_path = "/graphganvol/mnist_graph_gan/mnist_superpixels"

    args.channels = [64, 32, 16, 1]

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
