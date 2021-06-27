import torch
import numpy as np


datasets = ['g', 't', 'q']

for j in datasets:
    jets = torch.load(f'datasets/all_{j}_jets_150p_polarrel_mask.pt') # .numpy()[:, :30, :]
    # np.savetxt(f'datasets/{j}_jets.csv', jets.reshape(jets.shape[0], 30 * 4))
    # torch.save(jets[:, :30, :], f'datasets/{j}_jets.pt')
    print(jets.shape[0])
