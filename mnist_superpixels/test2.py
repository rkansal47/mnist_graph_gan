import torch

num_hits = 3
node_size = 3
x = torch.tensor([[1.,3.,4.],[2.,5,3],[1,2,3]])



x1 = x.repeat(1, num_hits).view(num_hits,num_hits, node_size)


x1

x2 = x.repeat(num_hits, 1).view(num_hits, num_hits, node_size)

x2


A = torch.cat((x1, x2), 2).view(num_hits*num_hits, node_size*2)

A

dists = torch.norm(x2[:, :, :2]-x1[:, :, :2] + 1e-12, dim=2).unsqueeze(2)

dists

sorted, idx = torch.sort(dists, dim=1)

idx

x3 = torch.cat([x2, dists], dim=2)

x3

torch.zeros(x3.shape).scatter(0, idx, x3)


y = torch.tensor([[3,4],[5,3]])
w = torch.tensor([1,2])

w.unsqueeze(-1)*y
y*w

y1 = torch.tensor([[1, 4], [3, 5]])
mu = torch.tensor([2,2])
sigma = torch.tensor([4, 5])

(y1-mu)**2
torch.sum(y1-mu)**2*sigma
