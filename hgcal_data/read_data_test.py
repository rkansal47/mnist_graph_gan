import numpy as np
import pandas as pd
from pathlib import Path
import h5py


num_thresholded = 400

event_file = "../hgcal_data/thresholded/events_xyz_" + str(num_thresholded) + ".csv"
energy_file = "thresholded/energies.csv"
in_particle_file = "thresholded/in_particle.csv"

enf = open(energy_file, "rb")

energies = np.loadtxt("thresholded/energies.csv")

print(energies.shape)

enf.close()

evf = open(event_file, "rb")

print("events opened")

events = np.loadtxt("../hgcal_data/thresholded/events_xyz_400.csv")

events_pd = pd.read_csv("../hgcal_data/thresholded/events_xyz_400.csv", nrows=10000)

events_pd.shape

events_np = events_pd.to

print(events.shape)

evf.close()


h


events_bin2 = np.load("../hgcal_data/thresholded/events_xyz_400.bin")

events_bin2.shape

events = np.memmap(event_file, shape=(99728, num_thresholded, 4), dtype='float64')

events[0]

inp = np.memmap("../hgcal_data/thresholded/in_particle.csv", shape=(99728,4), dtype='float64', mode='r')

inp_arr = np.loadtxt("../hgcal_data/thresholded/in_particle.csv")

inp_arr_pd = pd.read_csv("../hgcal_data/thresholded/in_particle.csv")

inp_arr.shape

inp_arr[0]

inp_arr.dtype

inp.shape

inp[0]




p = Path('test2.npy')

with p.open('ab') as f:
    array = np.array([[1, 2, 3], [3,5,6]])
    np.save(f, array)

    array2 = np.array([[4,2,1]])
    np.save(f, array2)



f = open("test.npy", "ab")

f.close()

array = np.array([[1, 2, 3], [3,5,6]])
np.save(f, array)

f.close()

array2 = np.array([[4,2,1]])

np.save(f, array2)

f.close()

arrayl = np.load("test2.npy")

arrayl


f.close()
