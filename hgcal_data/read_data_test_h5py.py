import h5py
# import numpy as np

f = h5py.File("thresholded/events_xyz_400.hdf5")

events = f["events"][:10000]

type(events)

events.shape

events[7893]

events[800]
