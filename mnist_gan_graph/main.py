import setGPU

import torch
from model import Simple_GRU, Critic, Graph_Discriminator
from graph_dataset_mnist import MNISTGraphDataset
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm

import numpy as np

from os import listdir
from os.path import isfile, join
import sys

#Have to specify 'name' and 'start_epoch' if True
LOAD_MODEL = False
WGAN = False
TRAIN = True
MNIST8M = False
NUM = 3 #-1 means all numbers
INTENSITIES = True
GRAPH_D = True
SAME_PARAMS = True

node_size = 3 if INTENSITIES else 2
fe_out_size = 128
gru_hidden_size = 128
gru_num_layers = 3
dropout = 0.3
leaky_relu_alpha = 0.2
batch_size = 16 if GRAPH_D else 64
num_hits = 100
gen_in_dim = 100
lr = 0.00005
lr_disc = 0.0002
lr_gen = 0.0001
num_critic = 1
weight_clipping_limit = 0.1
num_iters = 4
hidden_node_size = 64
gp_weight = 10

torch.manual_seed(4)

name = "31_intensities_graph_d_3s"

onlyfiles = [f for f in listdir('figs/') if isfile(join('figs/', f))]
if (name + "_1.png" in onlyfiles):
    print("file name already used")
    # if(not LOAD_MODEL):
    #     sys.exit()

del onlyfiles

f = open("args/" + name + ".txt", "w+")
f.write(str(locals()))
f.close()

#Change to True !!
X = MNISTGraphDataset(num_hits, train=TRAIN, num=NUM, intensities=INTENSITIES, mnist8m=MNIST8M)
X_loaded = DataLoader(X, shuffle=True, batch_size=batch_size)

if(LOAD_MODEL):
    start_epoch = 10
    G = torch.load("models/" + name + "_G_" + str(start_epoch) + ".pt")
    D = torch.load("models/" + name + "_D_" + str(start_epoch) + ".pt")
else:
    start_epoch = 0
    G = Simple_GRU(node_size, fe_out_size, gru_hidden_size, gru_num_layers, num_iters, num_hits, dropout, leaky_relu_alpha, SAME_PARAMS, hidden_node_size=hidden_node_size).cuda()

    if(GRAPH_D):
        D = Graph_Discriminator(node_size, fe_out_size, gru_hidden_size, gru_num_layers, num_iters, num_hits, dropout, leaky_relu_alpha, SAME_PARAMS, hidden_node_size=hidden_node_size).cuda()
    else:
        D = Critic((num_hits, node_size), dropout, batch_size, wgan=WGAN).cuda()

G_optimizer = optim.Adam(G.parameters(), lr = lr_gen, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr = lr_disc, betas=(0.5, 0.999))

normal_dist = Normal(0, 0.2)

def wasserstein_loss(y_out, y_true):
    return -torch.mean(y_out * y_true)

if(WGAN):
    criterion = wasserstein_loss
else:
    criterion = torch.nn.BCELoss()

def gen(num_samples, noise=0):
    if(noise == 0):
        noise = normal_dist.sample((num_samples, num_hits, hidden_node_size)).cuda()

    x = noise
    del noise

    x = G(x)
    return x

def disp_sample_outputs(name, epoch, dlosses, glosses):
    fig = plt.figure(figsize=(10,10))
    gen_out = gen(100)
    gen_out = gen_out.view(100, num_hits, node_size).cpu().detach().numpy()

    if(INTENSITIES):
        gen_out = gen_out*[28, 28, 1]+[14, 14, 1]
    else:
        gen_out = gen_out*[28, 28]+[14, 14]

    for i in range(1, 101):
        fig.add_subplot(10, 10, i)
        im_disp = np.zeros((28,28)) - 0.5

        if(INTENSITIES):
            im_disp += np.min(gen_out[i-1])

        for x in gen_out[i-1]:
            x0 = int(round(x[0])) if x[0] < 27 else 27
            x0 = x0 if x0 > 0 else 0

            x1 = int(round(x[1])) if x[1] < 27 else 27
            x1 = x1 if x1 > 0 else 0

            im_disp[x1, x0] = x[2] if INTENSITIES else 0.5

        plt.imshow(im_disp, cmap=cm.gray_r, interpolation='nearest')
        plt.axis('off')

    plt.savefig("figs/"+name + "_" + str(epoch) + ".png")

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(dlosses)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(glosses)
    plt.show()
    # plt.savefig("losses/"+name + "_gan_" + str(epoch) + ".png")

    plt.savefig("losses/"+ name +"_"+ str(epoch) + ".png")

def save_models(name, epoch):
    torch.save(G, "models/" + name + "_G_" + str(epoch) + ".pt")
    torch.save(D, "models/" + name + "_D_" + str(epoch) + ".pt")

def gradient_penalty(real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1)
        alpha = alpha.expand_as(real_data).cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True).cuda()

        del alpha
        torch.cuda.empty_cache()

        # Calculate probability of interpolated examples
        prob_interpolated = D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones(prob_interpolated.size()).cuda(), create_graph=True, retain_graph=True)[0].cuda()

        # print(gradients)
        # print(gradients.shape)

        gradients = gradients.contiguous()

        # print(gradients)
        # print(gradients.shape)

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return gp_weight * ((gradients_norm - 1) ** 2).mean()

def train_D(x):
    D.train()
    D_optimizer.zero_grad()

    if(not WGAN):
        Y_real = torch.ones(x.shape[0], 1).cuda()
        Y_fake = torch.zeros(x.shape[0], 1).cuda()

    D_real_output = D(x)
    gen_ims = gen(x.shape[0])
    D_fake_output = D(gen_ims)

    if(WGAN):
        D_loss = D_fake_output.mean() - D_real_output.mean() + gradient_penalty(x, gen_ims)
    else:
        D_real_loss = criterion(D_real_output, Y_real)
        D_fake_loss = criterion(D_fake_output, Y_fake)

        D_loss = D_real_loss + D_fake_loss

    D_loss.backward()
    D_optimizer.step()

    # if(WGAN):
    #     for p in D.parameters():
    #         p.data.clamp_(-weight_clipping_limit, weight_clipping_limit)

    return D_loss.item()

def train_G():
    G.train()
    G_optimizer.zero_grad()

    if(not WGAN):
        Y_real = torch.ones(batch_size, 1).cuda()

    gen_ims = gen(batch_size)

    D_fake_output = D(gen_ims)

    if(WGAN):
        G_loss = -D_fake_output.mean()
    else:
        G_loss = criterion(D_fake_output, Y_real)

    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()

D_losses = []
G_losses = []

disp_sample_outputs(name, 0, D_losses, G_losses)

save_models(name, 0)

for i in range(start_epoch, 1000):
    print("Epoch %d" % (i+1))
    D_loss = 0
    G_loss = 0
    for batch_ndx, x in tqdm(enumerate(X_loaded), total=len(X_loaded)):
        x = x.cuda()
        D_loss += train_D(x)
        if(batch_ndx > 0 and batch_ndx % num_critic == 0):
            G_loss += train_G()

    D_losses.append(D_loss/len(X_loaded)/2)
    G_losses.append(G_loss/len(X_loaded))

    disp_sample_outputs(name, i+1, D_losses, G_losses)

    if(i%5==4):
        save_models(name, i+1)
