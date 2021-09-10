import numpy as np
import torch
import utils
from jets_dataset import JetsDataset
from model import Graph_GAN
from ext_models import rGANG, GraphCNNGANG, TreeGANG
from pcgan_model import latent_G
from pcgan_model import G as G_pc


import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dirs = os.listdir('final_models')
# if '.DS_Store' in dirs: del dirs[0]

for dir in dirs:
    print(dir)
    if dir == '.DS_Store': continue

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


    model_name = dir.split('_')[0]
    print(model_name)

    if 'treegan' in model_name:
        continue
        G = TreeGANG(args.treegang_features, args.treegang_degrees, args.treegang_support).to(device)
        pcgan_args = None
    elif model_name == 'pcgan':
        Gpc = G_pc(args.node_feat_size, args.pcgan_z1_dim, args.pcgan_z2_dim).to(device)
        Gpc.load_state_dict(torch.load(f"models/pcgan/pcgan_G_pc_{args.jets}.pt", map_location=args.device))
        G = latent_G(args.pcgan_latent_dim, args.pcgan_z1_dim).to(device)
        pcgan_args = {'sample_points': True, 'G_pc': Gpc}
    else: continue
    # if model_name == 'fcpnet':
    #     G = rGANG(args).to(device)
    # elif model_name == 'graphcnnpnet':
    #     G = GraphCNNGANG(args).to(device)
    # elif model_name == 'mppnet':
    #     G = Graph_GAN(True, args).to(device)
    # else: continue

    # X = JetsDataset(args)

    # labels = X[:][1].to(device)
    labels = None

    print(G)
    print(path + G_file)
    G.load_state_dict(torch.load(path + G_file, map_location=device))

    G.eval()
    gen_out = utils.gen_multi_batch(args, G, 100000, labels=labels, use_tqdm=True, pcgan_args=pcgan_args)
    np.save(path + "samples.npy", gen_out)
