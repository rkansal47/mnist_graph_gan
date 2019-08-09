import torch
from torch.utils.data import Dataset
import numpy as np

class MNISTGraphDataset(Dataset):
    def __init__(self, num_thresholded, train=True, intensities=False, num=-1, mnist8m=False):
        if(train):
            if(mnist8m):
                dataset = np.loadtxt('../mnist_dataset/mnist8m.csv', delimiter=',', dtype=np.float32)
                labels = np.loadtxt('../mnist_dataset/mnist8m.labels.csv', delimiter=',', dtype=np.float32)
            else:
                dataset = np.loadtxt('../mnist_dataset/mnist_train.csv', delimiter=',', dtype=np.float32)
        else:
            dataset = np.loadtxt('../mnist_dataset/mnist_test.csv', delimiter=',', dtype=np.float32)

        print("MNIST CSV Loaded")

        print(dataset.shape)

        if(num>-1):
            if(mnist8m):
                dataset = dataset[labels==num]
            else:
                dataset = dataset[dataset[:,0]==num]

        print(dataset.shape)

        if(mnist8m):
            X_pre = (dataset-127.5)/255.0
        else:
            X_pre = (dataset[:, 1:]-127.5)/255.0

        imrange = np.linspace(-0.5, 0.5, num=28, endpoint=False)

        xs, ys = np.meshgrid(imrange, imrange)

        xs = xs.reshape(-1)
        ys = ys.reshape(-1)

        self.X = np.array(list(map(lambda x: np.array([xs, ys, x]).T, X_pre)))

        if(not intensities):
            self.X = np.array(list(map(lambda x: x[x[:,2].argsort()][-num_thresholded:, :2], self.X)))
        else:
            self.X = np.array(list(map(lambda x: x[x[:,2].argsort()][-num_thresholded:], self.X)))

        self.X = torch.FloatTensor(self.X).cuda()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
