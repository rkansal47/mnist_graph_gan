import uproot
import numpy as np


x = np.array([3., 6.])



file = uproot.open("HGCAL/tuple_100Of250_n100.root")

file['Delphes;1'].keys()

sf = file['Delphes;1']['simcluster_features'].array()

one_particle_events = []

i = 0
for sc in sf:
    print(len(sc))
    if(len(sc)==1):
        one_particle_events.append(i)
    i += 1

len(one_particle_events)

len(sf[0])

rechit_features = file['Delphes;1']['rechit_features'].array()
rechit_features.shape
rechit_features

one_particle_events

for rf in rechit_features[one_particle_events]:
    print(len(rf))


rechit_thresholded = []
threshold = 0.05
num_thresholded = 100

for rf in rechit_features[one_particle_events]:
    rfnp = np.array(rf)
    rechit_thresholded.append(rfnp[rfnp[:,0] > threshold])

events = np.zeros((len(one_particle_events), num_thresholded, 4))
energies = np.zeros(len(one_particle_events))


rfnp = np.array(rechit_features[one_particle_events[11]])

np.pad(rfnp, ((0, 13), (0, 0)), 'constant')

rechit_features[one_particle_events[11]]


i = 0
for rf in rechit_features[one_particle_events]:
    rfnp = np.array(rf)
    start_index = 0 if rfnp.shape[0] > 100 else 100 - rfnp.shape[0]
    events[i,start_index:] = rfnp[np.argsort(rfnp[:,0])][-num_thresholded:, [0,1,2,4]]
    energies[i] = sf[i][0][0]
    i += 1

events.shape
len(one_particle_events)

len(rechit_thresholded)

for hits in rechit_thresholded[0]:
    print(hits[0])

lengths = []

for rf in rechit_thresholded:
    print(len(rf))
    lengths.append((len(rf)))

min(lengths)
max(lengths)

for hits in rechit_features[51]:
    print(hits[0])

for i in one_particle_events:
    print("%d, %f" % (len(rechit_features[i]), sf[i][0][0]))


len(rechit_features[one_particle_events])

np_rf = np.array(rechit_features)


np_rf

np_rf.shape

np_rf[0]

len(np_rf[0])

len(np_rf[0][0])

file['Delphes;1'].keys()
rscf = file['Delphes;1']['rechit_simcluster_fractions'].array()


rscf.shape
len(rscf[0])
len(rscf[0][0])
