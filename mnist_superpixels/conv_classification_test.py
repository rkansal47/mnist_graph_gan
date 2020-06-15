import setGPU

# from profile import profile
# from time import sleep

import torch
from superpixels_dataset import SuperpixelsDataset
from conv_classifer_model import Gaussian_Classifier
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from skimage.draw import draw

import torch.nn.functional as F

import torch.optim as optim
from tqdm import tqdm

import os
from os import listdir
from os.path import join, isdir
import sys
import tarfile
import urllib

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm

import numpy as np
torch.cuda.set_device(0)

url = 'http://ls7-www.cs.uni-dortmund.de/cvpr_geometric_dl/mnist_superpixels.tar.gz'

#Have to specify 'name' and 'start_epoch' if True
LOAD_MODEL = False
TRAIN=False

node_feat_size = 3 # 2 coords + I
#edge network params
fe_hidden_size = 256
fe_out_size = 128
#message passing network params (either GRU or vanilla network)
mp_hidden_size = 256
mp_num_layers = 2
dropout = 0.3
leaky_relu_alpha = 0.2
num_hits = 75
#learning rates
lr = 0.0005
#number of critic/discriminator iterations for every generator iteration
num_critic = 1
#number of rnn iterations
num_iters = 1
#latent vector size of each node (incl node feature size)
hidden_node_size = 64
#wgan gradient penalty weight
gp_weight = 10
beta1 = 0.9

batch_size = 10

kernel_size = 10

num_epochs = 5000

name = "7_lr_0.0005_kernel_size_10"

dirs = listdir('.')
if('cmodels' not in dirs):
    os.mkdir('./cmodels')
if('closses' not in dirs):
    os.mkdir('./closses')
if('cargs' not in dirs):
    os.mkdir('./cargs')
if('dataset' not in dirs):
    os.mkdir('./dataset')

    file_tmp = urllib.urlretrieve(url, filename=None)[0]
    tar = tarfile.open(file_tmp)
    tar.extractall('./dataset/')

del dirs

onlydirs = [f for f in listdir('cmodels/') if isdir(join('cmodels/', f))]
if (name in onlydirs):
    print("name already used")
    # if(not LOAD_MODEL):
        # sys.exit()
else:
    os.mkdir('./closses/' + name)
    os.mkdir('./cmodels/' + name)

del onlydirs

f = open("cargs/" + name + ".txt", "w+")
f.write(str(locals()))
f.close()


torch.manual_seed(4)
torch.autograd.set_detect_anomaly(True)

#Change to True !!
training_data = SuperpixelsDataset(num_hits, train=True)
testing_data = SuperpixelsDataset(num_hits, train=False)

# print("loading")

train_dl = DataLoader(training_data, shuffle=True, batch_size=batch_size)
test_dl = DataLoader(testing_data, shuffle=True, batch_size=batch_size)

# print("loaded")

if(LOAD_MODEL):
    start_epoch = 1000
    C = torch.load("cmodels/" + name + "/C_" + str(start_epoch) + ".pt")
else:
    start_epoch = 0
    C = Gaussian_Classifier(node_feat_size, fe_hidden_size, fe_out_size, mp_hidden_size, mp_num_layers, num_iters, num_hits, dropout, leaky_relu_alpha, kernel_size=kernel_size, hidden_node_size=hidden_node_size).cuda()

C_optimizer = optim.Adam(C.parameters(), lr = lr, betas=(beta1, 0.999))

normal_dist = Normal(0, 0.2)

def plot_losses(name, epoch, train_losses, test_losses):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(train_losses)
    ax1.set_title('training')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(test_losses)
    ax2.set_title('testing')

    plt.savefig("closses/"+ name +"/"+ str(epoch) + ".png")
    plt.close()

def save_model(name, epoch):
    torch.save(C, "cmodels/" + name + "/C_" + str(epoch) + ".pt")

def train_C(X, y):
    C.train()
    C_optimizer.zero_grad()

    output = C(X)

    # print(output.shape)
    # print(y.shape)

    #nll_loss takes class labels as target, so one-hot encoding is not needed
    C_loss = F.nll_loss(output, y)

    C_loss.backward()
    C_optimizer.step()

    return C_loss.item()

train_losses = []
test_losses = []

# @profile
def its_showtime():
    for i in range(start_epoch, num_epochs):
        print("Epoch %d %s" % ((i+1), name))
        C_loss = 0
        for batch_ndx, data in tqdm(enumerate(train_dl), total=len(train_dl)):
            C_loss += train_C(data[0], data[1])

        train_losses.append(C_loss/len(train_dl))

        test()
        if((i+1)%1==0):
            save_model(name, i+1)
            plot_losses(name, i+1, train_losses, test_losses)

def test():
    C.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dl:
            output = C(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dl.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_dl.dataset), 100. * correct / len(test_dl.dataset)))

its_showtime()
