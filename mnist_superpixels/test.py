import torch
import numpy as np

import matplotlib.pyplot as plt   

dataset = torch.load("dataset/training.pt")

for i in dataset:
    print(i.shape)
dataset[2]
dataset[1].shape[1]/60000

ints = dataset[0]
coords = dataset[3]

ints = ints[dataset[4]==3]-0.5
coords = coords[dataset[4]==3]

ints.unsqueeze(2).size()

ints.size()
coords.size()

torch.cat((ints.unsqueeze(2), coords), 2)

dataset3 = dataset[dataset[4]==3]

dataset[4]==2

dataset[4]
dataset[0][0]

dataset[0]

dataset[4]


coords = np.array(dataset[3][0].int())

list(coords[0])

im = np.zeros((28,28))

im[coords[0][0]][coords[0][1]]

dataset[3][0][0]

dataset[0][0]

for i in range(75):
    im[coords[i][1]][coords[i][0]] = float(dataset[0][0][i])


plt.imshow(im, cmap='gray')
plt.show()
