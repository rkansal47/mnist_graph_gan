import torch
from torch.utils.data import Dataset

class SuperpixelsDataset(Dataset):
    def __init__(self, num_thresholded, train=True, intensities=False, num=-1, mnist8m=False):
        if(train):
            dataset = torch.load('dataset/training.pt')
        else:
            dataset = torch.load('dataset/test.pt')

        print("Dataset Loaded")

        ints = dataset[0]
        coords = dataset[3]

        if(num>-1):
            ints = ints[dataset[4]==num]
            coords = coords[dataset[4]==num]

        ints = ints-0.5
        coords = (coords-13.5)/27

        X = torch.cat((coords, ints.unsqueeze(2)), 2)

        self.X = X.cuda()

        print(X.size())
        print("data processed")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
