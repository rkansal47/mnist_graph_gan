import torch
from torch.utils.data import Dataset
import logging

class JetsDataset(Dataset):
    def __init__(self, args):
        if args.dataset == 'jets':
            mask = '_mask' if args.mask else ''
            if args.real_only: dataset = torch.load(args.datasets_path + 'all_t_jets_30p_polarrel_30only.pt')
            else: dataset = torch.load(args.datasets_path + 'all_' + args.jets + '_jets_150p_' + args.coords + mask + '.pt').float()[:, :args.num_hits, :]

            jet_features = torch.load(args.datasets_path + 'all_' + args.jets + '_jets_150p_jetptetamass.pt').float()[:, :args.clabels]
        elif args.dataset == 'jets-lagan':
            sig = 'signal' if args.jets == 'sig' else 'background'
            dataset = torch.load("{}lagan_{}.pt".format(args.datasets_path, sig)).float()[:, -args.num_hits:, :]
            jet_features = torch.load("{}lagan_{}_jetptetamass.pt".format(args.datasets_path, sig)).float()
            logging.debug('dataset: ' + str(dataset))

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
            if hasattr(args, 'mask_feat') and args.mask_feat: args.maxepp.append(1.0)

            logging.debug("Max Vals: " + str(args.maxepp))
            for i in range(3):
                dataset[:, :, i] /= args.maxepp[i]

            dataset[:, :, 2] -= 0.5     # pT is normalized between -0.5 and 0.5 so the peak pT lies in linear region of tanh
            if args.dataset == 'jets-lagan': dataset[:, :, 3] -= 0.5
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

        if hasattr(args, 'mask_c') and args.mask_c:
            num_particles = (torch.sum(dataset[:, :, 3] + 0.5, dim=1) / args.num_hits).unsqueeze(1)
            logging.debug("num particles: " + str(torch.sum(dataset[:, :, 3] + 0.5, dim=1)))

            if args.clabels: self.jet_features = torch.cat((self.jet_features, num_particles), dim=1)
            else: self.jet_features = num_particles


        if hasattr(args, 'noise_padding') and args.noise_padding:
            logging.debug("pre-noise padded dataset: \n {}".format(dataset[:2, -10:]))

            noise_padding = torch.randn((len(dataset), args.num_hits, 3)) / 6

            # DOUBLE CHECK
            # noise_padding[:, :, 2] = torch.relu(noise_padding[:, :, 2])
            # noise_padding[:, :, 2] -= 0.5
            # # noise_padding[noise_padding[:, :, 2] < -0.5][:, :, 2] = -0.5

            noise_padding[:, :, 2] += 0.5
            mask = (dataset[:, :, 3] + 0.5).bool()
            noise_padding[mask] = 0
            dataset += (torch.cat((noise_padding, torch.zeros((len(dataset), args.num_hits, 1))), dim=2))

            logging.debug("noise padded dataset: \n {}".format(dataset[:2, -10:]))

        logging.info("Dataset shape: " + str(self.X.shape))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.jet_features[idx]
