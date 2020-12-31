import torch

from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import numpy as np
from scipy import linalg


def add_bool_arg(parser, name, help, default=False, no_name=None):
    varname = '_'.join(name.split('-'))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=varname, action='store_true', help=help)
    if(no_name is None):
        no_name = 'no-' + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument('--' + no_name, dest=varname, action='store_false', help=no_help)
    parser.set_defaults(**{varname: default})


def mask_manual(args, gen_data):
    print("Before Mask: ")
    print(gen_data[0])
    if args.mask_exp:
        pts = gen_data[:, :, 2].unsqueeze(2)
        upper = (pts > args.pt_cutoff).float()
        lower = 1 - upper
        exp = torch.exp((pts - args.pt_cutoff) / args.pt_cutoff)
        mask = upper + lower * exp - 0.5
    else:
        mask = (gen_data[:, :, 2] > args.pt_cutoff).unsqueeze(2).float() - 0.5

    gen_data = torch.cat((gen_data, mask), dim=2)
    print("After Mask: ")
    print(gen_data[0])
    return gen_data


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


def gen(args, G, dist=None, num_samples=0, noise=None, labels=None, X_loaded=None):
    if(noise is None):
        noise = dist.sample((num_samples, args.num_hits, args.latent_node_size if args.latent_node_size else args.hidden_node_size))
    else: num_samples = noise.size(0)

    if args.clabels and labels is None:
        labels = next(iter(X_loaded))[1].to(args.device)
        while(labels.size(0) < num_samples):
            labels = torch.cat((labels, next(iter(X_loaded))[1]), axis=0)
        labels = labels[:num_samples]

    gen_data = G(noise, labels)
    if args.mask_manual: gen_data = mask_manual(args, gen_data)

    return gen_data


# from https://github.com/EmilienDupont/wgan-gp
def gradient_penalty(args, D, real_data, generated_data, batch_size):
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1).to(args.device)
    alpha = alpha.expand_as(real_data)
    interpolated = alpha * real_data + (1 - alpha) * generated_data
    interpolated = Variable(interpolated, requires_grad=True).to(args.device)

    del alpha
    torch.cuda.empty_cache()

    # Calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones(prob_interpolated.size()).to(args.device), create_graph=True, retain_graph=True, allow_unused=True)[0].to(args.device)
    gradients = gradients.contiguous()

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    gp = args.gp * ((gradients_norm - 1) ** 2).mean()
    # print("gradient penalty")
    # print(gp)
    return gp


bce = torch.nn.BCELoss()
mse = torch.nn.MSELoss()


def calc_D_loss(args, D, data, gen_data, real_outputs, fake_outputs, run_batch_size, Y_real, Y_fake):
    if args.debug:
        print("real outputs")
        print(real_outputs[:10])

        print("fake outputs")
        print(fake_outputs[:10])

    if(args.loss == 'og' or args.loss == 'ls'):
        if args.label_smoothing:
            Y_real = torch.empty(run_batch_size).uniform_(0.7, 1.2).to(args.device)
            Y_fake = torch.empty(run_batch_size).uniform_(0.0, 0.3).to(args.device)
        else:
            Y_real = Y_real[:run_batch_size]
            Y_fake = Y_fake[:run_batch_size]

        # randomly flipping labels for D
        if args.label_noise:
            Y_real[torch.rand(run_batch_size) < args.label_noise] = 0
            Y_fake[torch.rand(run_batch_size) < args.label_noise] = 1

    if(args.loss == 'og'):
        D_real_loss = bce(real_outputs, Y_real)
        D_fake_loss = bce(fake_outputs, Y_fake)
    elif(args.loss == 'ls'):
        D_real_loss = mse(real_outputs, Y_real)
        D_fake_loss = mse(fake_outputs, Y_fake)
    elif(args.loss == 'w'):
        D_real_loss = -real_outputs.mean()
        D_fake_loss = fake_outputs.mean()
    elif(args.loss == 'hinge'):
        D_real_loss = torch.nn.ReLU()(1.0 - real_outputs).mean()
        D_fake_loss = torch.nn.ReLU()(1.0 + fake_outputs).mean()

    D_loss = D_real_loss + D_fake_loss

    if(args.gp):
        gp = gradient_penalty(args, D, data, gen_data, run_batch_size)
        gpitem = gp.item()
        D_loss += gp
    else: gpitem = None

    return (D_loss, {'Dr': D_real_loss.item(), 'Df': D_fake_loss.item(), 'gp': gpitem, 'D': D_real_loss.item() + D_fake_loss.item()})


def calc_G_loss(args, fake_outputs, Y_real, run_batch_size):
    if args.debug: print(fake_outputs[:10])

    if(args.loss == 'og'):
        G_loss = bce(fake_outputs, Y_real[:run_batch_size])
    elif(args.loss == 'ls'):
        G_loss = mse(fake_outputs, Y_real[:run_batch_size])
    elif(args.loss == 'w' or args.loss == 'hinge'):
        G_loss = -fake_outputs.mean()

    return G_loss


# from https://github.com/mseitzer/pytorch-fid
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def rand_mix(args, X1, X2, p):
    if p == 1: return X1

    assert X1.size(0) == X2.size(0), "Error: different batch sizes of rand mix data"
    batch_size = X1.size(0)

    rand = torch.rand(batch_size, 1, 1).to(args.device)
    mix = torch.zeros(batch_size, 1, 1).to(args.device)
    mix[rand < p] = 1

    return X1 * (1 - mix) + X2 * mix
