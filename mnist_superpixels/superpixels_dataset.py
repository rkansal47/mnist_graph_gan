import torch
from torch.utils.data import Dataset


class SuperpixelsDataset(Dataset):
    def __init__(self, dataset_path, num_thresholded, train=True, intensities=False, num=-1, mnist8m=False):
        if(train):
            dataset1 = torch.load(dataset_path + 'training.pt')
            dataset2 = torch.load(dataset_path + 'test.pt')

            ints1 = dataset1[0]
            ints2 = dataset2[0]
            coords1 = dataset1[3]
            coords2 = dataset2[3]

            if(num > -1):
                ints1 = ints1[dataset1[4] == num]
                ints2 = ints2[dataset2[4] == num]
                coords1 = coords1[dataset1[4] == num]
                coords2 = coords2[dataset2[4] == num]

            ints = torch.cat((ints1, ints2), axis=0)
            coords = torch.cat((coords1, coords2), axis=0)
        else:
            dataset = torch.load(dataset_path + 'test.pt')

            ints = dataset[0]
            coords = dataset[3]

            if(num > -1):
                ints = ints[dataset[4] == num]
                coords = coords[dataset[4] == num]

        ints = ints - 0.5
        coords = (coords - 14) / 28

        X = torch.cat((coords, ints.unsqueeze(2)), 2)

        self.X = X

        print("Dataset Loaded. Shape: ")
        print(X.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # return (self.X[idx], self.y[idx])
        return self.X[idx]
