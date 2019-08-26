import uproot
import numpy as np
import h5py

from os import listdir
from os.path import isfile, join

rootfiles = ["HGCAL/" + f for f in listdir('HGCAL/') if isfile(join('HGCAL/', f))]
num_thresholded = 100

# event_file = "thresholded/events_xyz_" + str(num_thresholded) + ".bin"
# energy_file = "thresholded/energies.csv"
# in_particle_file = "thresholded/in_particle.csv"

event_file = "thresholded/events_xyz_" + str(num_thresholded) + ".hdf5"

evf = h5py.File(event_file, "a")
events_dset = evf.create_dataset("events", (99728, num_thresholded, 4))
inp_dset = evf.create_dataset("in_particle", (99728, 4))

n = 0
n_events = 0
for f in rootfiles:
    print(n)
    # evf = open(event_file, "ab")
    # enf = open(energy_file, "ab")
    # ipf = open(in_particle_file, "ab")

    # if(n == 2):
    #     break
    print(f)

    file = uproot.open(f)

    sf = file['Delphes;1']['simcluster_features'].array()

    one_particle_events = []

    i = 0
    for sc in sf:
        if(len(sc)==1):
            one_particle_events.append(i)
        i += 1

    num_events = len(one_particle_events)

    print("One Particle Events: %d" % len(one_particle_events))

    rechit_features = file['Delphes;1']['rechit_features'].array()
    events = np.zeros((len(one_particle_events), num_thresholded, 4))
    in_particle_data = np.zeros((len(one_particle_events), 4))

    i = 0
    for rf in rechit_features[one_particle_events]:
        rfnp = np.array(rf)
        start_index = 0 if rfnp.shape[0] > num_thresholded else num_thresholded - rfnp.shape[0]
        events[i,start_index:] = rfnp[np.argsort(rfnp[:,0])][-num_thresholded:, [0,5,6,7]]
        # energies[i] = sf[i][0][0]
        in_particle_data[i] = sf[i][0]
        i += 1

    # print(events.reshape(-1, num_thresholded*4).shape)
    # print(energies.shape)
    # print(in_particle_data.shape)

    # np.save(evf, events)#.reshape(-1, num_thresholded*4))
    # np.savetxt(enf, energies)
    # np.savetxt(ipf, in_particle_data)

    # evf.close()
    # enf.close()
    # ipf.close()

    events_dset[n_events:n_events+num_events] = events[:]
    inp_dset[n_events:n_events+num_events] = in_particle_data[:]
    n += 1
    n_events += num_events

evf.close()
