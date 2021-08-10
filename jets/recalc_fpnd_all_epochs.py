import numpy as np
import torch
import utils
from jets_dataset import JetsDataset
from torch.utils.data import DataLoader
from model import Graph_GAN
import evaluation


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = '202_t30_mask_c_lrx2_dea'
model_dir = 'models/' + model_name
losses_dir = 'losses/' + model_name
args_file = 'args/' + model_name + '.txt'

args = eval(open(args_file).read())
args['device'] = device
args['batch_size'] = 2048
args['datasets_path'] = 'datasets/'
args = utils.objectview(args)

X = JetsDataset(args, train=False)
X_test_loaded = DataLoader(X, batch_size=args.batch_size, pin_memory=True)
labels = X[:][1].to(device)

G = Graph_GAN(True, args).to(device)

C, mu2, sigma2 = evaluation.load(args, X_test_loaded)

fpnd = []

for i in range(5, 2001, 5):
    G.load_state_dict(torch.load(f"{model_dir}/G_{i}.pt", map_location=device))

    G.eval()
    gen_out = utils.gen_multi_batch(args, G, args.eval_tot_samples, labels=labels, use_tqdm=True)

    fpnd.append(evaluation.get_fpnd(args, C, gen_out, mu2, sigma2))

np.savetxt(f"{losses_dir}/fpnd_fix.txt", np.array(fpnd))
