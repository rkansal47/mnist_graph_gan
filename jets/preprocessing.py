import h5py
from tqdm import tqdm

from os import listdir
from os.path import isfile, join

import torch
import numpy as np

dir_path = '/graphganvol/datasets/jets/30p/train/'
# dir_path = '/Users/raghav/Documents/Work/CERN/datasets/'

rootfiles = [dir_path + f for f in listdir(dir_path) if isfile(join(dir_path, f))]
print(rootfiles)
print(len(rootfiles))

pfbool = True  # particle features or jet features
jet_type = 't'
particle_features = ['etarel', 'phirel', 'ptrel']
jet_features = ['pt', 'eta', 'mass']

n = 0
tot_jets = 0
jets = []
for f in rootfiles:
    print(n)
    print(f)

    file = h5py.File(f, 'r')
    print(file.keys())

    n_all_jets = file['jets'].shape[0]
    jtid = list(file['jetFeatureNames']).index(b'j_' + jet_type.encode('UTF-8'))

    if pfbool:
        pfid = [list(file['particleFeatureNames']).index(b'j1_' + pf.encode('UTF-8')) for pf in particle_features]
        for i in tqdm(range(n_all_jets)):
            if file['jets'][i][jtid]:
                tjet = []
                for particle in file['jetConstituentList'][i]:
                    pfs = [particle[id] for id in pfid]
                    tjet.append(pfs)
                jets.append(tjet)

    else:
        jetfsid = [list(file['jetFeatureNames']).index(b'j_' + jf.encode('UTF-8')) for jf in jet_features]
        gjets = file['jets'][:, jtid]
        if n == 0:
            jets = file['jets'][:, jetfsid][gjets == 1]
        else:
            jets = np.concatenate((jets, file['jets'][:, jetfsid][gjets == 1]))

    n += 1
    file.close()

torch.save(torch.tensor(jets), './datasets/all_' + jet_type + '_jets_30p_polarrel.pt')

