import numpy as np
import torch
import utils
from jets_dataset import JetsDataset
from model import Graph_GAN
from ext_models import rGANG, GraphCNNGANG

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dirs = os.listdir('final_models')
if '.DS_Store' in dirs: del dirs[0]

for dir in dirs:
    print(dir)

    path = 'final_models/' + dir + '/'
    files = os.listdir(path)
    for file in files:
        if file[-4:] == ".txt": args_file = file
        if file[0] == 'G' and file[-3:] == '.pt': G_file = file

    args = eval(open(path + args_file).read())
    args['device'] = device
    args['batch_size'] = 1024
    args['datasets_path'] = 'datasets/'
    args = utils.objectview(args)

    X = JetsDataset(args)

    labels = X[:][1]

    model_name = dir.split('_')[0]
    print(model_name)

    if model_name == 'fc':
        G = rGANG(args)
    elif model_name == 'graphcnn':
        G = GraphCNNGANG(args)
    elif model_name == 'mp':
        G = Graph_GAN(True, args)

    G.load_state_dict(torch.load(path + G_file, map_location=device))

    G.eval()
    gen_out = utils.gen_multi_batch(args, G, 100000, labels=labels, use_tqdm=True)
    np.save(path + "samples.npy", gen_out)
