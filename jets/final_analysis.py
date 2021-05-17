import numpy as np
import os
import pandas as pd

dirs = os.listdir('losses')

loss_map = {x.split('_')[0]: x for x in dirs}
loss_map

loss_keys = ['fpnd', 'coverage', 'mmd']
scores = []
names = []

for i in range(200, 236):
    name = loss_map[str(i)]
    names.append(name)
    score_arr = []
    loss_dir = 'losses/' + name + '/'

    w1j = np.loadtxt(loss_dir + 'w1j_10000m.txt')
    w1m = w1j[:, 0]
    min_epoch = np.argmin(w1m)
    score_arr.append(min_epoch * 5)
    score_arr.append(w1m[min_epoch])

    w1p = np.mean(np.loadtxt(loss_dir + 'w1_10000m.txt'), 1)
    score_arr.append(w1p[min_epoch])

    w1efp = np.mean(w1j[:, 2:], 1)
    score_arr.append(w1efp[min_epoch])

    for key in loss_keys:
        try: score_arr.append(np.loadtxt(loss_dir + key + '.txt')[min_epoch])
        except: score_arr.append(1e12)

    scores.append(score_arr)


scores


pd.DataFrame(scores, names).to_csv('scores.csv')
