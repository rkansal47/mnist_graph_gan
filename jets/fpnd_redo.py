import numpy as np
import utils
import os
from tqdm import tqdm
import torch
from jets_dataset import JetsDataset
from torch.utils.data import DataLoader, TensorDataset
import evaluation
import matplotlib.pyplot as plt
from particlenet import ParticleNet

dirs = os.listdir('final_models')

num_samples = 50000

args_txt = {'g': 'args/218_g30_mask_c_dea_no_pos_diffs.txt', 't': 'args/206_t30_mask_c_lrx2_dea_no_pos_diffs.txt', 'q': 'args/230_q30_mask_c_lrx05_dea_no_pos_diffs.txt'}


samples_dict = {'g': {}, 't': {}, 'q': {}}



for dataset in samples_dict.keys():
    print(dataset)
    args = eval(open(args_txt[dataset]).read())
    args['device'] = torch.device('cuda')
    args['datasets_path'] = './datasets/'
    args['fpnd_batch_size'] = 512
    args['evaluation_path'] = './evaluation/'
    args = utils.objectview(args)
    C, mu2, sigma2 = evaluation.load(args)

    X = JetsDataset(args, train=False)
    rng = np.random.default_rng()

    X_loaded = DataLoader(TensorDataset(torch.tensor(X[:50000][0])), batch_size=256)

    C.eval()
    for i, gen_jets in tqdm(enumerate(X_loaded), total=len(X_loaded)):
        gen_jets = gen_jets[0]
        if args.mask:
            mask = gen_jets[:, :, 3:4] >= 0
            gen_jets = (gen_jets * mask)[:, :, :3]
        if(i == 0): activations = C(gen_jets.to(args.device), ret_activations=True).cpu().detach()
        else: activations = torch.cat((C(gen_jets.to(args.device), ret_activations=True).cpu().detach(), activations), axis=0)

    activations = activations.numpy()

    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    np.savetxt('evaluation/' + dataset + "mu2.txt", mu)
    np.savetxt('evaluation/' + dataset + "sigma2.txt", sigma)


    for i, gen_jets in tqdm(enumerate(X_loaded), total=len(X_loaded)):
        gen_jets = gen_jets[0]
        if args.mask:
            mask = gen_jets[:, :, 3:4] >= 0
            gen_jets = (gen_jets * mask)[:, :, :3]
        if(i == 0): activations = C(gen_jets.to(args.device), ret_activations=True, relu_activations=True).cpu().detach()
        else: activations = torch.cat((C(gen_jets.to(args.device), ret_activations=True, relu_activations=True).cpu().detach(), activations), axis=0)

    activations = activations.numpy()

    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    np.savetxt('evaluation/' + dataset + "mu2_relu.txt", mu)
    np.savetxt('evaluation/' + dataset + "sigma2_relu.txt", sigma)

fpnds = {}

for dir in dirs:
    print(dir)
    if dir == '.DS_Store': continue

    model_name = dir.split('_')[0]

    samples = np.load('final_models/' + dir + '/samples.npy')[:num_samples]

    path = 'final_models/' + dir + '/'
    files = os.listdir(path)
    for file in files:
        if file[-4:] == ".txt": args_file = file

    args = eval(open(path + args_file).read())
    args['device'] = torch.device('cuda')
    args['datasets_path'] = './datasets/'
    args['evaluation_path'] = './evaluation/'
    args['fpnd_batch_size'] = 512
    args = utils.objectview(args)


    if args.mask:
        masks = samples[:, :, 3:4] == 0.5
        samples = samples * masks

    dataset = dir.split('_')[1]

    samples_dict[dataset][model_name] = samples

    C, mu2, sigma2 = evaluation.load(args)

    fpnds[dir] = evaluation.get_fpnd(args, C, samples[:50000], mu2, sigma2)
    print(fpnds[dir])


dirs


fpnds

C, mu2, sigma2 = evaluation.load(args)

mu2

fpnds

fpnds_dict = {'g': {}, 't': {}, 'q': {}}

for dataset in ['g', 't', 'q']:
    for key in fpnds.keys():
        fpnds_dict[key.split('_')[1]][key.split('_')[0]] = fpnds[key]



fpnds_dict['g']

for dataset in ['g', 't', 'q']:
    print(dataset)
    for s in ['fc', 'graphcnn', 'fcpnet', 'graphcnnpnet', 'mp', 'mplfc', 'fcmp', 'graphcnnmp', 'mpfc', 'mppnet']:
        print(s + "\t" + str(fpnds_dict[dataset][s]))

fpnds_dict








fpnds_relu = {}

for dir in dirs:
    print(dir)
    if dir == '.DS_Store': continue

    model_name = dir.split('_')[0]

    samples = np.load('final_models/' + dir + '/samples.npy')[:num_samples]

    path = 'final_models/' + dir + '/'
    files = os.listdir(path)
    for file in files:
        if file[-4:] == ".txt": args_file = file

    args = eval(open(path + args_file).read())
    args['device'] = torch.device('cuda')
    args['datasets_path'] = './datasets/'
    args['evaluation_path'] = './evaluation/'
    args['fpnd_batch_size'] = 512
    args = utils.objectview(args)


    if args.mask:
        masks = samples[:, :, 3:4] == 0.5
        samples = samples * masks

    dataset = dir.split('_')[1]

    # samples_dict[dataset][model_name] = samples

    C = ParticleNet(args.num_hits, args.node_feat_size, device=args.device).to(args.device)
    C.load_state_dict(torch.load(args.evaluation_path + "C_state_dict.pt", map_location=args.device))

    fullpath = args.evaluation_path + args.jets
    mu2 = np.loadtxt(fullpath + "mu2_relu.txt")
    sigma2 = np.loadtxt(fullpath + "sigma2_relu.txt")

    gen_out_loaded = DataLoader(TensorDataset(torch.tensor(samples[:50000])), batch_size=args.fpnd_batch_size)

    C.eval()
    for i, gen_jets in tqdm(enumerate(gen_out_loaded), total=len(gen_out_loaded)):
        gen_jets = gen_jets[0]
        if args.mask:
            mask = gen_jets[:, :, 3:4] >= 0
            gen_jets = (gen_jets * mask)[:, :, :3]
        if(i == 0): activations = C(gen_jets.to(args.device), ret_activations=True, relu_activations=True).cpu().detach()
        else: activations = torch.cat((C(gen_jets.to(args.device), ret_activations=True, relu_activations=True).cpu().detach(), activations), axis=0)

    activations = activations.numpy()

    mu1 = np.mean(activations, axis=0)
    sigma1 = np.cov(activations, rowvar=False)

    fpnd = utils.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    fpnds_relu[dir] = fpnd
    print(fpnds_relu[dir])








del(C)

# for dataset in samples_dict.keys():
#     for key in samples_dict[dataset].keys():
#         samples = samples_dict[dataset][key]
#         masks = samples[:, :, 3:4] == 0.5
#         samples_dict[dataset][key] = samples * masks




activations_dict = {}

samples_dict['t']['real'] = JetsDataset(utils.objectview(eval(open(args_txt['t']).read())), train=False)[:50000][0]

for key in samples_dict['t']:
    # if key != 'real': continue
    X_loaded = DataLoader(TensorDataset(torch.tensor(samples_dict['t'][key][:50000])), batch_size=512)

    for i, gen_jets in tqdm(enumerate(X_loaded), total=len(X_loaded)):
        if(i == 0): activations = C(gen_jets[0][:, :, :3].to(args.device), ret_activations=True).cpu().detach()
        else: activations = torch.cat((C(gen_jets[0][:, :, :3].to(args.device), ret_activations=True).cpu().detach(), activations), axis=0)

    activations_dict[key] = activations.numpy()



gen_jets[0].shape

activations_dict


plt.figure(figsize=(12, 10))
for key in activations_dict:
    _ = plt.hist(activations_dict[key][:, 231], label=key, histtype='step', bins=np.linspace(0, 1.5, 20), linewidth=4 if key == 'real' or key == 'mp' else 3, linestyle='solid' if key == 'real' or key == 'mp' else 'dotted')

plt.legend()





for dataset in samples_dict.keys():
    print(dataset)
    args = eval(open(args_txt[dataset]).read())
    args['device'] = torch.device('cuda')
    args['datasets_path'] = './datasets/'
    args['evaluation_path'] = './evaluation/'
    args = utils.objectview(args)
    C, mu2, sigma2 = evaluation.load(args)

    X = JetsDataset(args, train=False)
    rng = np.random.default_rng()

    X_loaded = DataLoader(TensorDataset(torch.tensor(X[:50000][0])), batch_size=256)

    len(X_loaded)
    C.eval()
    for i, gen_jets in tqdm(enumerate(X_loaded), total=len(X_loaded)):
        gen_jets = gen_jets[0]
        if args.mask:
            mask = gen_jets[:, :, 3:4] >= 0
            gen_jets = (gen_jets * mask)[:, :, :3]
        if(i == 0): activations = C(gen_jets.to(args.device), ret_activations=True).cpu().detach()
        else: activations = torch.cat((C(gen_jets.to(args.device), ret_activations=True).cpu().detach(), activations), axis=0)

    activations = activations.numpy()

    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    np.savetxt('evaluation/' + dataset + "mu2.txt", mu)
    np.savetxt('evaluation/' + dataset + "sigma2.txt", sigma)




dataset = 'g'
args = eval(open(args_txt[dataset]).read())
args['device'] = torch.device('cuda')
args['datasets_path'] = './datasets/'
args['evaluation_path'] = './evaluation/'
args = utils.objectview(args)
C, mu2, sigma2 = evaluation.load(args)


mu2

sigma2

X = JetsDataset(args, train=False)
rng = np.random.default_rng()

# fpnds = []

for i in range(3):
    print(i)
    X_rand_sample = rng.choice(25000, size=10000)
    X_rand_sample2 = rng.choice(25000, size=10000)
    X_loaded = DataLoader(TensorDataset(torch.tensor(X[X_rand_sample][0])), batch_size=256)
    X_loaded2 = DataLoader(TensorDataset(torch.tensor(X[X_rand_sample2][0])), batch_size=256)

    len(X_loaded)
    C.eval()
    for i, gen_jets in tqdm(enumerate(X_loaded), total=len(X_loaded)):
        gen_jets = gen_jets[0]
        if args.mask:
            mask = gen_jets[:, :, 3:4] >= 0
            gen_jets = (gen_jets * mask)[:, :, :3]
        if(i == 0): activations = C(gen_jets.to(args.device), ret_activations=True).cpu().detach()
        else: activations = torch.cat((C(gen_jets.to(args.device), ret_activations=True).cpu().detach(), activations), axis=0)

    activations = activations.numpy()

    mu1 = np.mean(activations, axis=0)
    sigma1 = np.cov(activations, rowvar=False)

    for i, gen_jets in tqdm(enumerate(X_loaded), total=len(X_loaded)):
        gen_jets = gen_jets[0]
        if args.mask:
            mask = gen_jets[:, :, 3:4] >= 0
            gen_jets = (gen_jets * mask)[:, :, :3]
        if(i == 0): activations = C(gen_jets.to(args.device), ret_activations=True).cpu().detach()
        else: activations = torch.cat((C(gen_jets.to(args.device), ret_activations=True).cpu().detach(), activations), axis=0)

    activations = activations.numpy()

    mu2 = np.mean(activations, axis=0)
    sigma2 = np.cov(activations, rowvar=False)

    fpnds.append(utils.calculate_frechet_distance(mu1, sigma1, mu2, sigma2))


fpnds

samples_dict[dataset]

for dataset in samples_dict.keys():
    print(dataset)
    args = eval(open(args_txt[dataset]).read())
    args['device'] = torch.device('cuda')
    args['datasets_path'] = './datasets/'
    args['evaluation_path'] = './evaluation/'
    args = utils.objectview(args)
    C, mu2, sigma2 = evaluation.load(args)

    X = JetsDataset(args, train=False)
    print(evaluation.get_fpnd(args, C, X[:][0], mu2, sigma2))
