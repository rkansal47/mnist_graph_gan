import setGPU
import torch
import numpy as np

dataset = np.loadtxt('../mnist_dataset/mnist_train.csv', delimiter=',', dtype=np.float32)

print("Dataset Loaded")

X_pre = (dataset[:, 1:]-127.5)/255.0

imrange = np.linspace(-0.5, 0.5, num=28, endpoint=False)

xs, ys = np.meshgrid(imrange, imrange)

xs = xs.reshape(-1)
ys = ys.reshape(-1)

print("Processing")

X = np.array(list(map(lambda x: np.array([xs, ys, x]).T, X_pre)))
X = torch.tensor(np.array(list(map(lambda x: x[x[:,2].argsort()][-100:], X)))).cuda()

Y = torch.tensor(dataset[:, 0]).cuda()

print(X)
print(Y)
