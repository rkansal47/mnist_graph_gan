# import setGPU

import torch
from model import Graph_Generator, Graph_Discriminator
from graph_dataset_hgcal import HGCALGraphDataset
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.switch_backend('agg')
# import matplotlib.cm as cm
#
# import numpy as np

import os
from os import listdir
from os.path import join, isdir
import sys

torch.cuda.set_device(0)

#Have to specify 'name' and 'start_epoch' if True
LOAD_MODEL = False
WGAN = False
TRAIN = True
COORDS = 'cartesian'

hit_feat_size = 4 # 3 coords + E
inp_feat_size = 4 # 3 coords + E
node_size = hit_feat_size + inp_feat_size
fe_hidden_size = 128
fe_out_size = 256
gru_hidden_size = 256
gru_num_layers = 3
dropout = 0.3
leaky_relu_alpha = 0.2
num_hits = 100
lr = 0.00005
lr_disc = 0.0001
lr_gen = 0.00005
num_critic = 1
num_iters = 4
hidden_node_size = 64
gp_weight = 10
beta1 = 0.5

batch_size = 64

# if(WGAN and GRAPH_D):
#     batch_size = 16
# elif(GRAPH_D):
#     batch_size = 32
# else:
#     batch_size = 64

torch.manual_seed(4)
torch.autograd.set_detect_anomaly(True)

name = "2_train"

onlydirs = [f for f in listdir('models/') if isdir(join('models/', f))]
if (name in onlydirs):
    print("name already used")
    if(not LOAD_MODEL):
        sys.exit()
else:
    os.mkdir('./losses/' + name)
    os.mkdir('./models/' + name)

del onlydirs

f = open("args/" + name + ".txt", "w+")
f.write(str(locals()))
f.close()

#Change to True !!
X = HGCALGraphDataset(num_hits, train=TRAIN, coords=COORDS)

print("loading")

X_loaded = DataLoader(X, shuffle=True, batch_size=batch_size)

print("loaded")

if(LOAD_MODEL):
    start_epoch = 255
    G = torch.load("models/" + name + "/G_" + str(start_epoch) + ".pt")
    D = torch.load("models/" + name + "/D_" + str(start_epoch) + ".pt")
else:
    start_epoch = 0
    G = Graph_Generator(hit_feat_size, inp_feat_size, fe_hidden_size, fe_out_size, gru_hidden_size, gru_num_layers, num_iters, num_hits, dropout, leaky_relu_alpha, hidden_node_size=hidden_node_size, coords=COORDS).cuda()
    D = Graph_Discriminator(node_size, fe_hidden_size, fe_out_size, gru_hidden_size, gru_num_layers, num_iters, num_hits, dropout, leaky_relu_alpha, hidden_node_size=hidden_node_size, coords=COORDS).cuda()

if(WGAN):
    G_optimizer = optim.RMSprop(G.parameters(), lr = lr_gen)
    D_optimizer = optim.RMSprop(D.parameters(), lr = lr_disc)
else:
    G_optimizer = optim.Adam(G.parameters(), lr = lr_gen, betas=(beta1, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr = lr_disc, betas=(beta1, 0.999))

normal_dist = Normal(0, 0.2)

def wasserstein_loss(y_out, y_true):
    return -torch.mean(y_out * y_true)

if(WGAN):
    criterion = wasserstein_loss
else:
    criterion = torch.nn.BCELoss()

def gen(num_samples, inp, noise=0):
    if(noise == 0):
        noise = normal_dist.sample((num_samples, num_hits, hidden_node_size)).cuda()

    x = noise
    del noise

    x = G(x, inp)
    return x

def plot_loss(name, epoch, dlosses, glosses):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(dlosses)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(glosses)
    plt.show()

    plt.savefig("losses/"+ name +"/"+ str(epoch) + ".png")

def save_models(name, epoch):
    torch.save(G, "models/" + name + "/G_" + str(epoch) + ".pt")
    torch.save(D, "models/" + name + "/D_" + str(epoch) + ".pt")

#from https://github.com/EmilienDupont/wgan-gp
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
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones(prob_interpolated.size()).cuda(), create_graph=True, retain_graph=True, allow_unused=True)[0].cuda()

        gradients = gradients.contiguous()

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return gp_weight * ((gradients_norm - 1) ** 2).mean()

def train_D(x, inp):
    D.train()
    D_optimizer.zero_grad()

    run_batch_size = inp.shape[0]
    inp = inp.repeat(1, num_hits).view(run_batch_size, num_hits, inp_feat_size)

    x = torch.cat((x[:,:,:hit_feat_size], inp[:]), 2)

    if(not WGAN):
        Y_real = torch.ones(run_batch_size, 1).cuda()
        Y_fake = torch.zeros(run_batch_size, 1).cuda()

    D_real_output = D(x)
    gen_ims = gen(run_batch_size, inp)
    D_fake_output = D(gen_ims)

    if(WGAN):
        D_loss = D_fake_output.mean() - D_real_output.mean() + gradient_penalty(x, gen_ims)
    else:
        D_real_loss = criterion(D_real_output, Y_real)
        D_fake_loss = criterion(D_fake_output, Y_fake)

        D_loss = D_real_loss + D_fake_loss

    D_loss.backward()
    D_optimizer.step()

    return D_loss.item()

def train_G(inp):
    G.train()
    G_optimizer.zero_grad()

    run_batch_size = inp.shape[0]

    if(not WGAN):
        Y_real = torch.ones(run_batch_size, 1).cuda()

    inp = inp.repeat(1, num_hits).view(run_batch_size, num_hits, inp_feat_size)

    gen_ims = gen(run_batch_size, inp)

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

# save_models(name, 0)

for i in range(start_epoch, 1000):
    print("Epoch %d" % (i+1))
    D_loss = 0
    G_loss = 0
    for batch_ndx, x in tqdm(enumerate(X_loaded), total=len(X_loaded)):
        # print(x)
        if(batch_ndx > 0 and batch_ndx % (num_critic+1) == 0):
            G_loss += train_G(x[1].cuda())
        else:
            D_loss += train_D(x[0].cuda(), x[1].cuda())

    D_losses.append(D_loss/len(X_loaded)/2)
    G_losses.append(G_loss/len(X_loaded))

    plot_loss(name, i+1, D_losses, G_losses)

    # if(i%5==4):
    save_models(name, i+1)
