import setGPU

import torch
from model import Simple_GRU, Critic
from graph_dataset_mnist import MNISTGraphDataset
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

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
WGAN = True

input_size = 2
output_size = 2
gru_hidden_size = 100
gru_num_layers = 3
dropout = 0.3
batch_size = 1024
num_thresholded = 100
gen_in_dim = 100
lr = 0.00005
lr_disc = 0.0001
lr_gen = 0.00005
num_critic = 1
weight_clipping_limit = 1

torch.manual_seed(4)

#Change to True !!
X = MNISTGraphDataset(num_thresholded, train=True)
X_loaded = DataLoader(X, shuffle=True, batch_size=batch_size)

name = "22_wgan"

if(LOAD_MODEL):
    start_epoch = 10
    G = torch.load("models/" + name + "_G_" + str(start_epoch) + ".pt")
    D = torch.load("models/" + name + "_D_" + str(start_epoch) + ".pt")
else:
    start_epoch = 0
    G = Simple_GRU(input_size, output_size, gen_in_dim, gru_hidden_size, gru_num_layers, dropout, batch_size).cuda()
    D = Critic((num_thresholded, input_size), dropout, batch_size, wgan=True).cuda()

G_optimizer = optim.Adam(G.parameters(), lr = lr_gen, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr = lr_disc, betas=(0.5, 0.999))

normal_dist = Normal(0, 1)

def wasserstein_loss(y_out, y_true):
    return -torch.mean(y_out * y_true)

if(WGAN):
    criterion = wasserstein_loss
else:
    criterion = torch.nn.BCELoss()

def gen(batch=True, noise=0):
    batch_size_run = batch_size if batch else 1
    if(noise == 0):
        noise = normal_dist.sample((batch_size_run, gen_in_dim)).cuda()

    hidden = G.initHidden(batch)

    out, hidden = G(noise, hidden, init=True, batch=batch)
    output = out.clone().cuda().view(batch_size_run, 1, input_size)
    for i in range(num_thresholded-1):
        out, hidden = G(out, hidden, batch=batch)
        output = torch.cat((output, out.view(batch_size_run, 1, input_size)), 1)

    return output

def disp_sample_outputs(name, epoch, dlosses, glosses):
    fig = plt.figure(figsize=(10,10))
    gen_out = gen()

    gen_out = gen_out.view(batch_size, num_thresholded, input_size).cpu().detach().numpy()*[28, 28]+[14, 14]

    for i in range(1, 101):
        fig.add_subplot(10, 10, i)
        im_disp = np.zeros((28,28)) - 0.5

        for x in gen_out[i-1]:
            im_disp[min(27, int(np.round(x[1]))), min(27, int(np.round(x[0])))] = 0.5
        plt.imshow(im_disp, cmap=cm.gray_r, interpolation='nearest')
        plt.axis('off')

    plt.savefig("figs/"+name + "_" + str(epoch) + ".png")

    plt.figure()
    plt.plot(dlosses)
    plt.savefig("losses/"+name + "_disc_" + str(epoch) + ".png")

    plt.figure()
    plt.plot(glosses)
    plt.savefig("losses/"+name + "_gan_" + str(epoch) + ".png")

def save_models(name, epoch):
    torch.save(G, "models/" + name + "_G_" + str(epoch) + ".pt")
    torch.save(D, "models/" + name + "_D_" + str(epoch) + ".pt")

def train_D(x):
    D.train()
    D.zero_grad()

    Y_real = torch.ones(x.shape[0], 1).cuda()
    Y_fake = torch.zeros(batch_size, 1).cuda()

    D_real_output = D(x)
    D_real_loss = criterion(D_real_output, Y_real)

    gen_ims = gen()
    D_fake_output = D(gen_ims)
    D_fake_loss = criterion(D_fake_output, Y_fake)

    D_loss = D_real_loss + D_fake_loss

    D_loss.backward()
    D_optimizer.step()

    if(WGAN):
        for p in D.parameters():
            p.data.clamp_(-weight_clipping_limit, weight_clipping_limit)

    return D_loss.item()

def train_G():
    G.train()
    G.zero_grad()

    Y_real = torch.ones(batch_size, 1).cuda()

    gen_ims = gen()
    D_fake_output = D(gen_ims)
    G_loss = criterion(D_fake_output, Y_real)

    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()

onlyfiles = [f for f in listdir('figs/') if isfile(join('figs/', f))]
if (name + "_1.png" in onlyfiles):
    print("file name already used")
    if(not LOAD_MODEL):
        sys.exit()

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

    D_losses.append(D_loss/len(X_loaded))
    G_losses.append(G_loss/len(X_loaded))

    disp_sample_outputs(name, i+1, D_losses, G_losses)

    if(i%5==4):
        save_models(name, i+1)

# print(D_losses)
# print(G_losses)
