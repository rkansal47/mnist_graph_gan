import numpy as np
import matplotlib.pyplot as plt
import utils
import mplhep as hep


# plt.switch_backend('macosx')
plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)

model = '202_t30_mask_c_lrx2_dea/'
loss_dir = 'losses/' + model

loss_keys = ['fpnd_fix', 'mmd', 'coverage']


losses = {}

w1j = np.loadtxt(loss_dir + 'w1j_10000m.txt')
losses['w1m'] = w1j[:, 0]
losses['w1p'] = np.mean(np.loadtxt(loss_dir + 'w1_10000m.txt'), 1)

losses['w1efp'] = np.mean(w1j[:, 2:], 1)

for key in loss_keys:
    losses[key] = np.loadtxt(loss_dir + key + '.txt')



losses['fpnd_fix'].shape
plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)




%matplotlib inline
fig = plt.figure(figsize=(12, 10))
h = plt.hist2d(losses['w1m'], losses['fpnd_fix'], bins=50, range=[[0, 0.02], [0, 50]], cmap='jet')
c = plt.colorbar(h[3])
c.set_label('Number of batches')
plt.xlabel('W1-M')
plt.ylabel('FPND')
plt.title('W1-M vs FPND Correlation')
plt.savefig('correlation_figs/w1mvfpnd.pdf', bbox_inches='tight')

fig = plt.figure(figsize=(12, 10))
h = plt.hist2d(losses['w1m'], losses['w1efp'], bins=50, range=[[0, 0.015], [0, 0.0005]], cmap='jet')
plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
c = plt.colorbar(h[3])
c.set_label('Number of batches')
plt.xlabel('W1-M')
plt.ylabel('W1-EFP')
plt.title('W1-M vs W1-EFP Correlation')
plt.savefig('correlation_figs/w1mvw1efp.pdf', bbox_inches='tight')



fig = plt.figure(figsize=(12, 10))
h = plt.hist2d(losses['w1m'], losses['w1p'], bins=50, range=[[0, 0.02], [0, 0.01]], cmap='jet')
# plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
c = plt.colorbar(h[3])
c.set_label('Number of batches')
plt.xlabel('W1-M')
plt.ylabel('W1-P')
plt.title('W1-M vs W1-P Correlation')
plt.savefig('correlation_figs/w1mvw1p.pdf', bbox_inches='tight')




fig = plt.figure(figsize=(12, 10))
h = plt.hist2d(losses['w1p'], losses['fpnd_fix'], bins=50, range=[[0, 0.01], [0, 50]], cmap='jet')
# plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
c = plt.colorbar(h[3])
c.set_label('Number of batches')
plt.xlabel('W1-P')
plt.ylabel('FPND')
plt.title('W1-P vs FPND Correlation')
plt.savefig('correlation_figs/w1pvfpnd.pdf', bbox_inches='tight')



fig = plt.figure(figsize=(12, 10))
h = plt.hist2d(losses['w1m'], losses['mmd'], bins=50, range=[[0, 0.01], [0, 0.1]], cmap='jet')
# plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
c = plt.colorbar(h[3])
c.set_label('Number of batches')
plt.xlabel('W1-M')
plt.ylabel('MMD')
plt.title('W1-M vs MMD Correlation')
plt.savefig('correlation_figs/w1mvmmd.pdf', bbox_inches='tight')


fig = plt.figure(figsize=(12, 10))
h = plt.hist2d(losses['w1m'], losses['coverage'], bins=50, range=[[0, 0.01], [0, 1]], cmap='jet')
# plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
c = plt.colorbar(h[3])
c.set_label('Number of batches')
plt.xlabel('W1-M')
plt.ylabel('COV')
plt.title('W1-M vs COV Correlation')
plt.savefig('correlation_figs/w1mvcov.pdf', bbox_inches='tight')
