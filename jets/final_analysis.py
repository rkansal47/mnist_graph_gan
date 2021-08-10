import numpy as np
import os
import pandas as pd
import shutil

dirs = os.listdir('losses')

loss_map = {x.split('_')[0]: x for x in dirs}
loss_map

loss_keys = ['fpnd', 'coverage', 'mmd']
scores = []
names = []

for i in range(270, 279):
    name = loss_map[str(i)]
    print(name)
    names.append(name)
    score_arr = []
    loss_dir = 'losses/' + name + '/'

    w1j = np.loadtxt(loss_dir + 'w1j_10000m.txt')
    w1jstd = np.loadtxt(loss_dir + 'w1j_10000std.txt')
    w1m = w1j[:, 0]
    w1mstd = w1jstd[:, 0]
    min_epoch = np.argmin(w1m)

    final_models_dir = 'final_models/'
    if 'treegang_rgand' in name: final_models_dir += 'treeganfc_'
    if 'treegang_pointnetd' in name: final_models_dir += 'treeganpnet_'
    if 'pcgang' in name: final_models_dir += 'pcgan_'

    if 'g30' in name: final_models_dir += 'g/'
    if 't30' in name: final_models_dir += 't/'
    if 'q30' in name: final_models_dir += 'q/'

    print(final_models_dir)

    if not os.path.exists(final_models_dir): os.makedirs(final_models_dir)
    shutil.copy(f"args/{name}.txt", final_models_dir)
    shutil.copy(f"models/{name}/G_{min_epoch * 5}.pt", final_models_dir)
    shutil.copy(f"models/{name}/D_{min_epoch * 5}.pt", final_models_dir)


    score_arr.append(min_epoch * 5)
    score_arr.append(w1m[min_epoch])
    score_arr.append(w1mstd[min_epoch])

    w1p = np.mean(np.loadtxt(loss_dir + 'w1_10000m.txt'), 1)
    w1pstd = np.linalg.norm(np.loadtxt(loss_dir + 'w1_10000std.txt'), axis=1) / 3

    score_arr.append(w1p[min_epoch])
    score_arr.append(w1pstd[min_epoch])

    w1efp = np.mean(w1j[:, 2:], 1)
    w1efpstd = np.linalg.norm(w1j[:, 2:], axis=1) / 5
    score_arr.append(w1efp[min_epoch])
    score_arr.append(w1efpstd[min_epoch])

    for key in loss_keys:
        try: score_arr.append(np.loadtxt(loss_dir + key + '.txt')[min_epoch])
        except: score_arr.append(1e12)

    scores.append(score_arr)


pd.DataFrame(scores, names).to_csv('scores2.csv')
