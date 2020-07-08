import torch
from torch.utils.data import Dataset


class SuperpixelsDataset(Dataset):
    def __init__(self, dataset_path, num_thresholded, train=True, intensities=False, num=-1, mnist8m=False):
        if(train):
            dataset = torch.load(dataset_path + 'training.pt')
        else:
            dataset = torch.load(dataset_path + 'test.pt')

        print("Dataset Loaded. Shape: ")

        ints = dataset[0]
        coords = dataset[3]

        # self.y = torch.tensor(dataset[4], dtype=torch.long)

        if(num > -1):
            ints = ints[dataset[4] == num]
            coords = coords[dataset[4] == num]

        ints = ints - 0.5
        coords = (coords - 14) / 28

        X = torch.cat((coords, ints.unsqueeze(2)), 2)

        self.X = X

        # print(X.size())
        print(X.shape)
        print(X[0])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # return (self.X[idx], self.y[idx])
        return self.X[idx]
