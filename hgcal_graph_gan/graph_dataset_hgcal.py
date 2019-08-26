import torch
from torch.utils.data import Dataset
import h5py

# Loads the HGCAL Graphical Dataset

class HGCALGraphDataset(Dataset):
    def __init__(self, num_thresholded, train=True, coords='cartesian'):
        test_limit = 10000
        data_folder = "../hgcal_data/thresholded/"

        coords = 'xyz_' if coords == 'cartesian' else ''

        file = h5py.File(data_folder + "events_" + coords + str(num_thresholded) + ".hdf5", "r")

        if(train):
            self.events = file["events"][:]
            self.inp = file["in_particle"][:]
        else:
            self.events = file["events"][:test_limit]
            self.inp = file["in_particle"][:test_limit]

        print("CSV Loaded")
        print(self.events.shape)
        print(self.inp.shape)

        self.events = torch.FloatTensor(self.events)
        self.inp = torch.FloatTensor(self.inp)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return (self.events[idx], self.inp[idx])
