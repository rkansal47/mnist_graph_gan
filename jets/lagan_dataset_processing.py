import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import utils


lagan_data = h5py.File('datasets/jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5', 'r')


X_pre = np.array(lagan_data['image'])
imrange = np.linspace(-1.2, 1.2, num=25)
xs, ys = np.meshgrid(imrange, imrange)

xs = xs.reshape(-1)
ys = ys.reshape(-1)

X = np.array(list(map(lambda x: np.array([xs, ys, x]).T, X_pre.reshape(-1, 25 * 25))))
Xsorted = np.array(list(map(lambda x: x[x[:, 2].argsort()][-200:], X)))


Xsorted[Xsorted[:, :, 2] == 0] = [0, 0, 0]

mask = (Xsorted[:, :, 2] != 0)

Xmask = np.concatenate((Xsorted, mask.reshape(-1, 200, 1)), axis=2)

torchX = torch.tensor(Xmask)

signal = np.array(lagan_data['signal'])

torch.save(torchX[signal == 1], 'datasets/lagan_signal.pt')
torch.save(torchX[signal == 0], 'datasets/lagan_background.pt')

#
# realjf = utils.jet_features(Xmask[signal == 1][:, :, :3], True, mask[signal == 1])
#
#
# Xmask[signal == 1][:, -75:, :]
#
# plt.hist(lagan_data['jet_pt'][signal == 1], bins=np.linspace(220, 340, 51), histtype='step', color='red')
# plt.hist(realjf[:, 1], bins=np.linspace(220, 340, 51), histtype='step')
#
# 
# plt.hist(lagan_data['jet_mass'][signal == 1], bins=np.linspace(40, 120, 51), histtype='step', color='red')
# plt.hist(realjf[:, 0], bins=np.linspace(40, 120, 51), histtype='step')
#
#
# plt.hist(Xmask[signal == 1][mask[signal == 1]][:, 2].reshape(-1), histtype='step', bins=np.linspace(0, 1, 101))
#
#
# lagan_data['jet_pt']

jet_features = np.concatenate((np.array(lagan_data['jet_pt']).reshape(-1, 1), np.array(lagan_data['jet_eta']).reshape(-1, 1), np.array(lagan_data['jet_mass']).reshape(-1, 1)), axis=1)

torch.save(torch.tensor(jet_features[signal == 1]), 'datasets/lagan_signal_jetptetamass.pt')
torch.save(torch.tensor(jet_features[signal == 0]), 'datasets/lagan_background_jetptetamass.pt')
