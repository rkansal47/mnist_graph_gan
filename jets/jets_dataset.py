import torch
from torch.utils.data import Dataset
import logging

class JetsDataset(Dataset):
    def __init__(self, args):
        mask = '_mask' if args.mask else ''
        if args.real_only:
            dataset = torch.load(args.datasets_path + 'all_t_jets_30p_polarrel_30only.pt')
        else:
            dataset = torch.load(args.datasets_path + 'all_' + args.jets + '_jets_150p_' + args.coords + mask + '.pt').float()[:, :args.num_hits, :]

        jet_features = torch.load(args.datasets_path + 'all_' + args.jets + '_jets_150p_jetptetamass.pt').float()[:, :args.clabels]

        if args.coords == 'cartesian':
            args.maxp = float(torch.max(torch.abs(dataset)))
            dataset = dataset / args.maxp

            cutoff = int(dataset.size(0) * args.ttsplit)

            if(args.train):
                self.X = dataset[:cutoff]
            else:
                self.X = dataset[cutoff:]
        elif args.coords == 'polarrel' or args.coords == 'polarrelabspt':
            args.maxepp = [float(torch.max(torch.abs(dataset[:, :, i]))) for i in range(3)]
            if args.mask_feat:
                args.maxepp.append(1.0)

            logging.debug("Max Vals: " + str(args.maxepp))
            for i in range(3):
                dataset[:, :, i] /= args.maxepp[i]

            dataset[:, :, 2] -= 0.5     # pT is normalized between -0.5 and 0.5 so the peak pT lies in linear region of tanh
            # dataset[:, :, 2] *= 2
            dataset *= args.norm
            self.X = dataset
            args.pt_cutoff = torch.unique(self.X[:, :, 2], sorted=True)[1]  # smallest particle pT after 0
            logging.debug("Cutoff: " + str(args.pt_cutoff))

        if args.clabels == 1:
            args.maxjf = [torch.max(torch.abs(jet_features))]
            jet_features /= args.maxjf[0]
        else:
            [float(torch.max(torch.abs(jet_features[:, :, i]))) for i in range(args.clabels)]
            for i in range(args.clabels):
                jet_features[:, i] /= args.maxjf[i]

        self.jet_features = jet_features * args.norm

        logging.info("Dataset shape: " + str(self.X.shape))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.jet_features[idx]
