import torch
from torch.utils.data import Dataset


class JetsDataset(Dataset):
    def __init__(self, args):
        dataset = torch.load(args.dataset_path + 'all_g_jets_' + str(args.num_hits) + 'p_' + args.coords + '.pt')

        if args.coords == 'cartesian':
            maxp = torch.max(torch.abs(dataset))
            dataset = dataset / maxp

            cutoff = int(dataset.size(0) * args.ttsplit)

            if(args.train):
                self.X = dataset[:cutoff]
            else:
                self.X = dataset[cutoff:]
        else:
            dataset[:, :, 0] /= torch.max(torch.abs(dataset[:, :, 0]))
            dataset[:, :, 1] /= torch.max(torch.abs(dataset[:, :, 1]))
            dataset[:, :, 2] /= torch.max(torch.abs(dataset[:, :, 2]))
            self.X = dataset

        print("Dataset loaded")
        print(self.X.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
