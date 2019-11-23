import torch
import numpy as np
from skimage.draw import draw

import matplotlib.pyplot as plt
import matplotlib.cm as cm

dataset = torch.load("dataset/training.pt")

ints = dataset[0]
coords = dataset[3]

ints = ints[dataset[4]==3]-0.5
coords = coords[dataset[4]==3]

ints.unsqueeze(2).size()

ints.size()
coords.size()


# data[0]
#
# data[0].shape


def draw_graph(graph, node_r, im_px):
    imd = im_px + node_r
    img = np.zeros((imd, imd), dtype=np.float)

    circles = []
    for node in graph:
        circles.append((draw.circle_perimeter(int(node[1]), int(node[0]), node_r), draw.circle(int(node[1]), int(node[0]), node_r), node[2]))

    for circle in circles:
        img[circle[1]] = circle[2]

    return img

node_r = 30
im_px = 1000
imd = im_px + node_r

img = np.zeros((imd, imd), dtype=np.float)

data = torch.cat((coords, ints.unsqueeze(2)), 2).numpy()
data = data*[im_px/28.0, im_px/28.0, 1] + [node_r/2, node_r/2, 0.55]

circles = []

for node in data[0]:
    circles.append((draw.circle_perimeter(int(node[1]), int(node[0]), node_r), draw.circle(int(node[1]), int(node[0]), node_r), node[2]))

for circle in circles:
    img[circle[1]] = circle[2]
    # img[circle[0]] = 1
plt.imshow(draw_graph(data[3], 30, 1000), cmap=cm.gray_r, interpolation='nearest')
plt.axis('off')

plt.show()


fig = plt.figure(figsize=(10,10))

for i in range(1, 100+1):
    fig.add_subplot(10, 10, i)
    im_disp = draw_graph(data[i-1], node_r, im_px)
    plt.imshow(im_disp, cmap=cm.gray_r, interpolation='nearest')
    plt.axis('off')

plt.show()
plt.savefig("test7.png")
