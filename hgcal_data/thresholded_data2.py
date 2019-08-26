import uproot
# import numpy as np
# import h5py

from os import listdir
from os.path import isfile, join

data_dir = "/eos/user/r/rkansal/HGCAL-Gun/"
num_thresholded = 100
total_events = 100

hit_feat_size = 4 # 3 coords + E
inp_feat_size = 4 # 3 coords + E

part = 0 #  particle type {electron: 0, photon: 1}

rootfiles = [data_dir + f for f in listdir(data_dir) if isfile(join(data_dir, f))]

event_file = "thresholded2/events_xyz_" + str(num_thresholded) + ".hdf5"

evf = h5py.File(event_file, "a")
events_dset = evf.create_dataset("events", (total_events, num_thresholded, 4))
inp_dset = evf.create_dataset("in_particle", (total_events, 4))

n = 0
n_events = 0

for f in rootfiles:
    print(str(n) + ": " + f)
    file = uproot.open(f)

    print(file.keys())

    coords = []

    cell_x = file[file.keys()[0]]['cell_x'].array()
    cell_y = file[file.keys()[0]]['cell_x'].array()
    num_clusters = cell_x.shape[0]

    print(num_clusters)
    n_events += num_clusters

    n+= 1

print("Total Events: %d" % n_events)
