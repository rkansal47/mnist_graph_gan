import numpy as np
import utils
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import evaluation

dirs = os.listdir('final_models')

num_samples = 50000

samples_dict = {'g': {}, 't': {}, 'q': {}}

for dir in dirs:
    if dir == '.DS_Store': continue

    model_name = dir.split('_')[0]

    if not (model_name == 'mp' or model_name == 'mppnet'):
        continue

    samples = np.load('final_models/' + dir + '/samples.npy')[:num_samples]

    path = 'final_models/' + dir + '/'
    files = os.listdir(path)
    for file in files:
        if file[-4:] == ".txt": args_file = file

    args = eval(open(path + args_file).read())
    args['datasets_path'] = 'datasets/'
    args = utils.objectview(args)


    dataset = dir.split('_')[1]

    if model_name == 'mp':
        samples_dict[dataset]['MP'] = samples
    elif model_name == 'mppnet':
        samples_dict[dataset]['MPPNET'] = samples

for dataset in samples_dict.keys():
    for key in samples_dict[dataset].keys():
        samples = samples_dict[dataset][key]
        masks = samples[:, :, 3:4] == 0.5
        samples_dict[dataset][key] = samples * masks



args_txt = {'g': 'args/218_g30_mask_c_dea_no_pos_diffs.txt', 't': 'args/206_t30_mask_c_lrx2_dea_no_pos_diffs.txt', 'q': 'args/230_q30_mask_c_lrx05_dea_no_pos_diffs.txt'}


for dataset in samples_dict.keys():
    print(dataset)
    args = eval(open(args_txt[dataset]).read())
    args['device'] = torch.device('cpu')
    args['datasets_path'] = './datasets/'
    args['evaluation_path'] = './evaluation/'
    args = utils.objectview(args)
    C, mu2, sigma2 = evaluation.load(args)
    for key in samples_dict[dataset].keys():
        print(key)
        print(evaluation.get_fpnd(args, C, samples_dict[dataset][key], mu2, sigma2))
