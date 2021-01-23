# import setGPU

import torch
import setup, utils, save_outputs, evaluation, augment
from jets_dataset import JetsDataset
from torch.utils.data import DataLoader

from tqdm import tqdm

# from parallel import DataParallelModel, DataParallelCriterion

import logging


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(4)
    torch.autograd.set_detect_anomaly(True)

    args, tqdm_out = setup.init()
    # args = setup.init()
    args.device = device
    logging.info("Args initalized")

    X = JetsDataset(args)
    X_loaded = DataLoader(X, shuffle=True, batch_size=args.batch_size, pin_memory=True)
    logging.info("Data loaded")

    G, D = setup.models(args)
    logging.info("Models loaded")

    G_optimizer, D_optimizer = setup.optimizers(args, G, D)
    logging.info("Optimizers loaded")

    losses = setup.losses(args)

    if args.fid: C, mu2, sigma2 = evaluation.load(args, X_loaded)

    Y_real = torch.ones(args.batch_size, 1).to(args.device)
    Y_fake = torch.zeros(args.batch_size, 1).to(args.device)

    mse = torch.nn.MSELoss()
    # if args.multi_gpu:
    #     mse = DataParallelCriterion(mse)

    def train_D(data, labels=None, gen_data=None, epoch=0, print_output=False):
        logging.debug("dtrain")
        log = logging.info if print_output else logging.debug

        D.train()
        D_optimizer.zero_grad()
        G.eval()

        run_batch_size = data.shape[0]

        D_real_output = D(data.clone(), labels, epoch=epoch)

        log("D real output: ")
        log(D_real_output[:10])

        if gen_data is None:
            gen_data = utils.gen(args, G, run_batch_size, labels=labels)

        if args.augment:
            p = args.aug_prob if not args.adaptive_prob else losses['p'][-1]
            data = augment.augment(args, data, p)
            gen_data = augment.augment(args, gen_data, p)

        log("G output: ")
        log(gen_data[:2, :10, :])

        D_fake_output = D(gen_data, labels, epoch=epoch)

        log("D fake output: ")
        log(D_fake_output[:10])

        D_loss, D_loss_items = utils.calc_D_loss(args, D, data, gen_data, D_real_output, D_fake_output, run_batch_size, Y_real, Y_fake, mse)
        D_loss.backward()

        D_optimizer.step()
        return D_loss_items

    def train_G(data, labels=None, epoch=0):
        logging.debug("gtrain")
        G.train()
        G_optimizer.zero_grad()

        run_batch_size = labels.shape[0] if labels is not None else args.batch_size

        gen_data = utils.gen(args, G, run_batch_size, labels=labels)

        if args.augment:
            p = args.aug_prob if not args.adaptive_prob else losses['p'][-1]
            gen_data = augment.augment(args, gen_data, p)

        D_fake_output = D(gen_data, labels, epoch=epoch)

        logging.debug("D fake output:")
        logging.debug(D_fake_output[:10])

        G_loss = utils.calc_G_loss(args, D_fake_output, Y_real, run_batch_size, mse)

        G_loss.backward()
        G_optimizer.step()

        return G_loss.item()

    def train():
        if(args.fid): losses['fid'].append(evaluation.get_fid(args, C, G, mu2, sigma2))
        if(args.start_epoch == 0 and args.save_zero):
            if args.w1: gen_out = evaluation.calc_w1(args, X[:][0], G, losses, X_loaded=X_loaded)
            else: gen_out = None
            save_outputs.save_sample_outputs(args, D, G, X[:args.num_samples][0], args.name, 0, losses, X_loaded=X_loaded, gen_out=gen_out)

        for i in range(args.start_epoch, args.num_epochs):
            logging.info("Epoch {} starting".format(i + 1))
            D_losses = ['Dr', 'Df', 'D']
            if args.gp: D_losses.append('gp')
            epoch_loss = {'G': 0}
            for key in D_losses: epoch_loss[key] = 0

            lenX = len(X_loaded)

            bar = tqdm(enumerate(X_loaded), total=lenX, mininterval=0.1, desc="Epoch {}".format(i + 1))
            for batch_ndx, data in bar:
                if args.clabels: labels = data[1].to(args.device)
                else: labels = None

                data = data[0].to(args.device)

                if args.num_critic > 1 or (batch_ndx == 0 or (batch_ndx - 1) % args.num_gen == 0):
                    D_loss_items = train_D(data, labels=labels, epoch=i, print_output=(batch_ndx == lenX))  # print outputs for the last iteration of each epoch
                    for key in D_losses: epoch_loss[key] += D_loss_items[key]

                if args.num_critic == 1 or (batch_ndx - 1) % args.num_critic == 0:
                    epoch_loss['G'] += train_G(data, labels=labels, epoch=i)

                if args.bottleneck:
                    if(batch_ndx == 10):
                        return

            logging.info("Epoch {} Training Over".format(i + 1))

            for key in D_losses: losses[key].append(epoch_loss[key] / (lenX / args.num_gen))
            losses['G'].append(epoch_loss['G'] / (lenX / args.num_critic))
            for key in epoch_loss.keys(): logging.info("{} loss: {:.3f}".format(key, losses[key][-1]))

            if((i + 1) % 5 == 0):
                optimizers = (D_optimizer, G_optimizer)
                save_outputs.save_models(args, D, G, optimizers, args.name, i + 1)
                # if args.w1: evaluation.calc_w1(args, X[:][0], G, normal_dist, losses, X_loaded=X_loaded)

            if(args.fid and (i + 1) % 1 == 0):
                losses['fid'].append(evaluation.get_fid(args, C, G, mu2, sigma2))

            if((i + 1) % args.save_epochs == 0):
                if args.w1: gen_out = evaluation.calc_w1(args, X[:][0], G, losses, X_loaded=X_loaded)
                else: gen_out = None
                save_outputs.save_sample_outputs(args, D, G, X[:args.num_samples][0], args.name, i + 1, losses, X_loaded=X_loaded, gen_out=gen_out)

    train()


if __name__ == "__main__":
    main()
