import torch
from torch.distributions.normal import Normal
from model import Simple_GRU, Critic, Graph_Discriminator
import matplotlib.pyplot as plt


x = [0,2, 3, 4, 5, 8]
y = [3, 4, 2, 5, 6, 7]

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x)
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(y)
plt.show()

#Have to specify 'name' and 'start_epoch' if True
LOAD_MODEL = False
WGAN = True
TRAIN = False
NUM = 3 #-1 means all numbers
INTENSITIES = False
GRAPH_D = True

node_size = 3 if INTENSITIES else 2
fe_out_size = 128
gru_hidden_size = 100
gru_num_layers = 3
dropout = 0.3
leaky_relu_alpha = 0.2
batch_size = 128
num_hits = 100
gen_in_dim = 100
lr = 0.00005
lr_disc = 0.0002
lr_gen = 0.0001
num_critic = 1
weight_clipping_limit = 0.1
num_iters = 4


locals()

G = Simple_GRU(node_size, fe_out_size, gru_hidden_size, gru_num_layers, num_iters, num_hits, dropout, leaky_relu_alpha)

param_string = "LOAD_MODEL: " + str(LOAD_MODEL) + "\n"

normal_dist = Normal(0, 0.2)

batch_size = 3
num_hits = 5
node_size = 3

x = normal_dist.sample((batch_size, num_hits, node_size))

x

x1 = x.repeat(1,1,num_hits).view(batch_size,num_hits*num_hits,node_size)
x2 = x.repeat(1,num_hits,1)

x1

x2

norms = torch.norm(x2[:, :, :2]-x1[:, :, :2], dim=2).unsqueeze(2)

norms

pairs = torch.cat((x1, x2, norms) ,2).view(-1, 5)

pairs2 = pairs.view(batch_size, num_hits, num_hits, 2*node_size+1)
pairs2


torch.sum(pairs2,2)


x.unsqueeze(3).repeat(1,1,1,2)


y = x.view(-1)



torch.zeros()

x1t = 30
x1 = x1t if x1t < 27 else 27
x1 = x1 if x1 > 0 else 0

x1
