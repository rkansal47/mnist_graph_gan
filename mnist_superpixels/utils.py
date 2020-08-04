import torch
from torch_geometric.data import Batch, Data

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


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


def gen(args, G, dist, num_samples, noise=0, disp=False):
    if(noise == 0):
        if(args.gcnn):
            rand = dist.sample((num_samples * 5, 2 + args.channels[0]))
            noise = Data(pos=rand[:, :2], x=rand[:, 2:])
        else:
            noise = dist.sample((num_samples, args.num_hits, args.hidden_node_size))

    gen_data = G(noise)

    if args.gcnn and disp: return torch.cat((gen_data.pos, gen_data.x), 1).view(num_samples, 75, 3)
    return gen_data


# transform my format to torch_geometric's
def tg_transform(args, X):
    batch_size = X.size(0)

    pos = X[:, :, :2]

    x1 = pos.repeat(1, 1, 75).reshape(batch_size, 75 * 75, 2)
    x2 = pos.repeat(1, 75, 1)

    diff_norms = torch.norm(x2 - x1 + 1e-12, dim=2)

    # diff = x2-x1
    # diff = diff[diff_norms < args.cutoff]

    norms = diff_norms.reshape(batch_size, 75, 75)
    neighborhood = torch.nonzero(norms < args.cutoff, as_tuple=False)
    # diff = diff[neighborhood[:, 1] != neighborhood[:, 2]]

    neighborhood = neighborhood[neighborhood[:, 1] != neighborhood[:, 2]]  # remove self-loops
    unique, counts = torch.unique(neighborhood[:, 0], return_counts=True)
    # edge_slices = torch.cat((torch.tensor([0]).to(device), counts.cumsum(0)))
    edge_index = (neighborhood[:, 1:] + (neighborhood[:, 0] * 75).view(-1, 1)).transpose(0, 1)

    # normalizing edge attributes
    # edge_attr_list = list()
    # for i in range(batch_size):
    #     start_index = edge_slices[i]
    #     end_index = edge_slices[i + 1]
    #     temp = diff[start_index:end_index]
    #     max = torch.max(temp)
    #     temp = temp/(2 * max + 1e-12) + 0.5
    #     edge_attr_list.append(temp)
    #
    # edge_attr = torch.cat(edge_attr_list)

    # edge_attr = diff/(2 * args.cutoff) + 0.5

    x = X[:, :, 2].reshape(batch_size * 75, 1) + 0.5
    pos = 28 * pos.reshape(batch_size * 75, 2) + 14

    row, col = edge_index
    edge_attr = (pos[col] - pos[row]) / (2 * 28 * args.cutoff) + 0.5

    zeros = torch.zeros(batch_size * 75, dtype=int).to(args.device)
    zeros[torch.arange(batch_size) * 75] = 1
    batch = torch.cumsum(zeros, 0) - 1

    return Batch(batch=batch, x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr, y=None, pos=pos)


# from https://github.com/EmilienDupont/wgan-gp
def gradient_penalty(args, D, real_data, generated_data, batch_size):
    # Calculate interpolation
    if(not args.gcnn):
        alpha = torch.rand(batch_size, 1, 1).to(args.device)
        alpha = alpha.expand_as(real_data)
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated = Variable(interpolated, requires_grad=True).to(args.device)
    else:
        alpha = torch.rand(batch_size, 1, 1).to(args.device)
        alpha_x = alpha.expand((batch_size, 75, 1))
        interpolated_x = alpha_x * real_data.x.reshape(batch_size, 75, 1) + (1 - alpha_x) * generated_data.x.reshape(batch_size, 75, 1)
        alpha_pos = alpha.expand((batch_size, 75, 2))
        interpolated_pos = alpha_pos * real_data.pos.reshape(batch_size, 75, 2) + (1 - alpha_pos) * generated_data.pos.reshape(batch_size, 75, 2)
        interpolated_X = Variable(torch.cat(((interpolated_pos - 14) / 28, interpolated_x - 0.5), dim=2), requires_grad=True)
        interpolated = tg_transform(args, interpolated_X)

    del alpha
    torch.cuda.empty_cache()

    # Calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # Calculate gradients of probabilities with respect to examples
    if(not args.gcnn):
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones(prob_interpolated.size()).to(args.device), create_graph=True, retain_graph=True, allow_unused=True)[0].to(args.device)
    if(args.gcnn):
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated_X, grad_outputs=torch.ones(prob_interpolated.size()).to(args.device), create_graph=True, retain_graph=True, allow_unused=True)[0].to(args.device)

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


def convert_to_batch(args, data, batch_size):
    zeros = torch.zeros(batch_size * 75, dtype=int).to(args.device)
    zeros[torch.arange(batch_size) * 75] = 1
    batch = torch.cumsum(zeros, 0) - 1

    return Batch(batch=batch, x=data.x, pos=data.pos, edge_index=data.edge_index, edge_attr=data.edge_attr)


bce = torch.nn.BCELoss()
mse = torch.nn.MSELoss()


def calc_D_loss(args, D, data, gen_data, real_outputs, fake_outputs, run_batch_size):
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
            Y_real = torch.ones(run_batch_size, 1).to(args.device)
            Y_fake = torch.zeros(run_batch_size, 1).to(args.device)

        # randomly flipping labels for D
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


def calc_G_loss(args, fake_outputs):
    if args.debug: print(fake_outputs[:10])

    if(args.loss == 'og' or args.loss == 'ls'):
        Y_real = torch.ones(args.batch_size, 1).to(args.device)

    if(args.loss == 'og'):
        G_loss = bce(fake_outputs, Y_real)
    elif(args.loss == 'ls'):
        G_loss = mse(fake_outputs, Y_real)
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


def rand_translate(args, X):
    batch_size = X.size(0)
    shift_xy = (torch.rand(batch_size, args.num_hits, 2).to(args.device) - 0.5) * args.translate_ratio
    zeros = torch.zeros(batch_size, args.num_hits, 1).to(args.device)
    add = torch.cat((shift_xy, zeros), axis=2)
    return X + add
