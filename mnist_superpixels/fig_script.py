import torch
import numpy as np
from skimage.draw import draw

import matplotlib.pyplot as plt
import matplotlib.cm as cm

dataset = torch.load("dataset/training.pt")

ints = dataset[0]
coords = dataset[3]

ints_arr = []
num_per_row = 5

node_r = 30
im_px = 1000

for i in range(10):
    intst = ints[dataset[4] == i][:num_per_row]
    coordst = coords[dataset[4] == i][:num_per_row]
    intst = intst - 0.5
    coordst = (coordst - 14) / 28
    X = torch.cat((coordst, intst.unsqueeze(2)), 2).numpy()
    X[X > 0.47] = 0.47
    X[X < -0.5] = -0.5
    X = X * [im_px, im_px, 1] + [(im_px + node_r) / 2, (im_px + node_r) / 2, 0.55]
    ints_arr.append(X)


def draw_graph(graph, node_r, im_px):
    imd = im_px + node_r
    img = np.zeros((imd, imd), dtype=np.float)

    circles = []
    for node in graph:
        circles.append((draw.circle_perimeter(int(node[1]), int(node[0]), node_r), draw.circle(int(node[1]), int(node[0]), node_r), node[2]))

    for circle in circles:
        img[circle[1]] = circle[2]

    return img


fig = plt.figure(figsize=(num_per_row, 10))
# plt.subplots_adjust(wspace=0, hspace=0)

for i in range(10):
    for j in range(num_per_row):
        fig.add_subplot(10, num_per_row, i * num_per_row + j + 1)
        im_disp = draw_graph(ints_arr[i][j], node_r, im_px)
        plt.imshow(im_disp, cmap=cm.gray_r, interpolation='nearest')
        plt.axis('off')

# fig.tight_layout()

plt.savefig("paper_figs/real_superpixels.png")
plt.show()
