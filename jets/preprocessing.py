import h5py
from tqdm import tqdm

from os import listdir
from os.path import isfile, join

import torch
import numpy as np

import sys

# dir_path = '/graphganvol/datasets/jets/150p/train/'
dir_path = '/Users/raghav/Documents/Work/CERN/datasets/'

rootfiles = [dir_path + f for f in listdir(dir_path) if isfile(join(dir_path, f))]

dir_path_test = '/graphganvol/datasets/jets/150p/test/'
rootfiles += [dir_path_test + f for f in listdir(dir_path_test) if isfile(join(dir_path_test, f))]

print(rootfiles)
print(len(rootfiles))

pfbool = False  # particle features or jet features
print(sys.argv[1])
jet_type = sys.argv[1]
particle_features = ['etarel', 'phirel', 'ptrel']
jet_features = ['pt', 'eta', 'mass']

n = 0
tot_jets = 0
jetjfs = []
jetpfs = []
for f in rootfiles:
    print(n)
    print(f)

    file = h5py.File(f, 'r')

    n_all_jets = file['jets'].shape[0]
    jtid = list(file['jetFeatureNames']).index(b'j_' + jet_type.encode('UTF-8'))

    pfid = [list(file['particleFeatureNames']).index(b'j1_' + pf.encode('UTF-8')) for pf in particle_features]
    for i in tqdm(range(n_all_jets)):
        if file['jets'][i][jtid]:
            tjet = []
            for particle in file['jetConstituentList'][i]:
                pfs = [particle[id] for id in pfid]
                tjet.append(pfs)
            jetpfs.append(tjet)

    jetfsid = [list(file['jetFeatureNames']).index(b'j_' + jf.encode('UTF-8')) for jf in jet_features]
    gjets = file['jets'][:, jtid]
    if n == 0:
        jetjfs = file['jets'][:, jetfsid][gjets == 1]
    else:
        jetjfs = np.concatenate((jetjfs, file['jets'][:, jetfsid][gjets == 1]))

    n += 1
    file.close()

    torch.save(torch.tensor(jetpfs), './datasets/all_' + jet_type + '_jets_150p_polarrel.pt')
    torch.save(torch.tensor(jetjfs), './datasets/all_' + jet_type + '_jets_150p_jetptetamass.pt')
