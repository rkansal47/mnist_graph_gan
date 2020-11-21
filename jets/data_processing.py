import torch
import energyflow.utils as ut
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

dataset = torch.load('datasets/all_t_jets_30p_polarrel.pt')


torch.count_nonzero(torch.gt(torch.norm(dataset, dim=2), 0).float())

(torch.norm(dataset, dim=2).reshape(-1) == 0).sum()
len(torch.norm(dataset, dim=2))

pnorm = torch.norm(dataset, dim=2)

pnorm.size()

mask = torch.gt(pnorm, 0).float() - 0.5

mask.unsqueeze(2).size()

dataset_masked = torch.cat((dataset.float(), mask.unsqueeze(2)), dim=2)

torch.save(dataset_masked, './datasets/all_t_jets_30p_polarrel_mask.pt')


dataset = torch.load('datasets/all_g_jets_100p_polarrel.pt')

pnorm = torch.norm(dataset, dim=2)
mask = torch.gt(pnorm, 0).float() - 0.5
dataset_masked = torch.cat((dataset.float(), mask.unsqueeze(2)), dim=2)

torch.save(dataset_masked, './datasets/all_g_jets_100p_polarrel_mask.pt')


datasett = torch.load('datasets/all_t_jets_30p_polarrel.pt')
X = datasett.detach().numpy()
X_efp_format = np.concatenate((np.expand_dims(X[:, :, 2], 2), X[:, :, :2], np.zeros((len(X), 30, 1))), axis=2)

Xims = []
for i in tqdm(range(len(X))):
    Xims.append(ut.pixelate(X_efp_format[i, :, :], npix=25, img_width=0.85, nb_chan=1, norm=False, charged_counts_only=False))


datasetg = torch.load('datasets/all_g_jets_30p_polarrel.pt')
Xg = datasetg.detach().numpy()
Xg_efp_format = np.concatenate((np.expand_dims(Xg[:, :, 2], 2), Xg[:, :, :2], np.zeros((len(Xg), 30, 1))), axis=2)

Xgims = []
for i in tqdm(range(len(Xg))):
    Xgims.append(ut.pixelate(Xg_efp_format[i, :, :], npix=25, img_width=0.85, nb_chan=1, norm=False, charged_counts_only=False))


plt.imshow(Xgims[6])

plt.imshow(Xims[6])

len(Xims)
len(Xgims)

np.save('gims', Xgims)
np.save('tims', Xims)
