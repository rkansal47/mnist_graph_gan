import torch
import numpy as np

from skimage.draw import draw
import matplotlib.pyplot as plt
import matplotlib.cm as cm

dataset = np.loadtxt('mnist_dataset/mnist_test.csv', delimiter=',', dtype=np.float32)

ints_arr = []
num_per_row = 5

num_hits = 100

node_r = 30
im_px = 1000

for i in range(10):
    X_pre = (dataset[dataset[:, 0] == i][:num_per_row, 1:] - 127.5) / 255.0

    imrange = np.linspace(-0.5, 0.5, num=28, endpoint=False)

    xs, ys = np.meshgrid(imrange, imrange)

    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    X = np.array(list(map(lambda x: np.array([xs, ys, x]).T, X_pre)))
    X = np.array(list(map(lambda x: x[x[:, 2].argsort()][-num_hits:], X)))

    X.shape

    X = X * [28, 28, 1] + [14, 14, 1]

    # X[X > 0.47] = 0.47
    # X[X < -0.5] = -0.5
    # X = X * [im_px, im_px, 1] + [(im_px + node_r) / 2, (im_px + node_r) / 2, 0.55]

    ints_arr.append(X)

ints_arr[0].shape

fig = plt.figure(figsize=(num_per_row, 10))
# plt.subplots_adjust(wspace=0, hspace=0)

for i in range(10):
    for j in range(num_per_row):
        fig.add_subplot(10, num_per_row, i * num_per_row + j + 1)

        im_disp = np.zeros((28, 28)) - 0.5

        # im_disp += np.min(ints_arr[i][j])

        for x in ints_arr[i][j]:
            x0 = int(round(x[0])) if x[0] < 27 else 27
            x0 = x0 if x0 > 0 else 0

            x1 = int(round(x[1])) if x[1] < 27 else 27
            x1 = x1 if x1 > 0 else 0

            im_disp[x1, x0] = x[2]

        plt.imshow(im_disp, cmap=cm.gray_r, interpolation='nearest')
        plt.axis('off')

# plt.savefig("paper_figs/real_" + str(num_hits) + "_sm.png")
plt.show()





#
#
# def draw_graph(graph, node_r, im_px):
#     imd = im_px + node_r
#     img = np.zeros((imd, imd), dtype=np.float)
#
#     circles = []
#     for node in graph:
#         circles.append((draw.circle_perimeter(int(node[1]), int(node[0]), node_r), draw.circle(int(node[1]), int(node[0]), node_r), node[2]))
#
#     for circle in circles:
#         img[circle[1]] = circle[2]
#
#     return img
#
#
# fig = plt.figure(figsize=(num_per_row, 10))
# # plt.subplots_adjust(wspace=0, hspace=0)
#
# for i in range(10):
#     for j in range(num_per_row):
#         fig.add_subplot(10, num_per_row, i * num_per_row + j + 1)
#         im_disp = draw_graph(ints_arr[i][j], node_r, im_px)
#         plt.imshow(im_disp, cmap=cm.gray_r, interpolation='nearest')
#         plt.axis('off')
#
# # fig.tight_layout()
#
# plt.savefig("paper_figs/real_100_sm_sp.png")
# plt.show()
