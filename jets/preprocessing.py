import h5py
from tqdm import tqdm

from os import listdir
from os.path import isfile, join

import torch

dir_path = '/graphganvol/datasets/jets/train'
# dir_path = '/Users/raghav/Documents/Work/CERN/datasets/'

rootfiles = [dir_path + f for f in listdir(dir_path) if isfile(join(dir_path, f))]

jet_type = 'g'
particle_features = ['etarel', 'phirel', 'ptrel']

n = 0
tot_jets = 0
jets = []
for f in rootfiles:
    print(n)
    print(f)

    file = h5py.File(f)
    print(file.keys())

    pfid = [list(file['particleFeatureNames']).index(b'j1_' + pf.encode('UTF-8')) for pf in particle_features]
    jtid = list(file['jetFeatureNames']).index(b'j_' + jet_type.encode('UTF-8'))

    n_jets = 0
    n_all_jets = file['jets'].shape[0]

    for i in tqdm(range(n_all_jets)):
        if file['jets'][i][jtid]:
            tjet = []
            for particle in file['jetConstituentList'][i]:
                pfs = [particle[id] for id in pfid]
                tjet.append(pfs)
            jets.append(tjet)
            n_jets += 1

    n += 1
    file.close()

torch.save(torch.tensor(jets), './datasets/all_g_jets_30p_polarrel.pt')
