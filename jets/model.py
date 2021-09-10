import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm
import logging


class LinearNet(nn.Module):
    """
    Module for fully connected networks with leaky relu activations

    Args:
        layers (list): list with layers of the fully connected network, optionally containing the input and output sizes inside e.g. [input_size, ... hidden layers ..., output_size]
        input_size (list, optional): size of input, if 0 or unspecified, first element of `layers` will be treated as the input size
        output_size (list, optional): size of output, if 0 or unspecified, last element of `layers` will be treated as the output size
        layers (list): list with layers of the fully connected network, specified as [input size, ... hidden layers ..., output size]
        final_linear (bool, optional): keep the final layer operation linear i.e. no normalization, no nonlinear activation
        leaky_relu_alpha (float, optional): negative slope of leaky relu
        dropout_p (float, optional): dropout fraction after each layer
        batch_norm (bool, optional): use batch norm or not
        spectral_norm (bool, optional): use spectral norm or not
    """

    def __init__(self, layers: list, input_size: int = 0, output_size: int = 0, final_linear: bool = False, leaky_relu_alpha: float = 0.2, dropout_p: float = 0, batch_norm: bool = False, spectral_norm: bool = False):
        super(LinearNet, self).__init__()

        self.final_linear = final_linear
        self.leaky_relu_alpha = leaky_relu_alpha
        self.batch_norm = batch_norm
        self.dropout = nn.Dropout(p=dropout_p)

        layers = layers.copy()

        if input_size: layers.insert(0, input_size)
        if output_size: layers.append(output_size)

        self.net = nn.ModuleList()
        if batch_norm: self.bn = nn.ModuleList()
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            self.net.append(linear)
            if batch_norm: self.bn.append(nn.BatchNorm1d(layers[i + 1]))

        if spectral_norm:
            for i in range(len(self.net)):
                if i != len(self.net) - 1 or not final_linear:
                    self.net[i] = SpectralNorm(self.net[i])

    def forward(self, x: torch.Tensor):
        """
        Runs input `x` through linear layers and returns output

        Args:
            x (torch.Tensor): input tensor of shape [batch size, # input features]
        """
        for i in range(len(self.net)):
            x = self.net[i](x)
            if i != len(self.net) - 1 or not self.final_linear:
                x = F.leaky_relu(x, negative_slope=self.leaky_relu_alpha)
                if self.batch_norm: x = self.bn[i](x)
            x = self.dropout(x)

        return x

    def __repr__(self):
        # TODO!
        return ""


class MPLayer(nn.Module):
    """
    MPLayer as described in [paper link]
    TODO: mathematical formulation

    Args:
        node_size (int): input node feature size
        fe_layers (list): list of edge network layer output sizes
        fn_layers (list): list of node network layer output sizes
        pos_diffs (bool, optional): use some measure of the distance between nodes as the edge features between them
        all_ef (bool, optional): use the euclidean distance between all the node features as an edge feature - only active is pos_diffs = True
        coords (str, optional): the coordinate system used for node features ('polarrel', 'polar', or 'cartesian'), only useful if delta_coords or delta_r is True
        delta_coords (bool, optional): use the vector difference between the two nodes as edge features
        delta_r (bool, optional): use the delta R between two nodes as edge features
        int_diffs (bool, optional): (Not implemented yet!) use the difference between pT as an edge feature
        fully_connected (bool, optional): fully connected graph for message passing
        num_knn (int, optional): if not fully connected, # of nodes to use for knn for message passing
        self_loops (bool, optional): if not fully connected, allow for self loops in message passing
        sum (bool, optional): sum as the message aggregation operation, as opposed to mean
        **linear_args (optional): additional arguments for linear layers, given to LinearNet modules
    """

    def __init__(self, node_size: int, fe_layers: list, fn_layers: list,
                 pos_diffs: bool = False, all_ef: bool = True, coords: str = 'polarrel', delta_coords: bool = False, delta_r: bool = True, int_diffs: bool = False,
                 clabels: int = 0, mask_fne_np: bool = False,
                 fully_connected: bool = True, num_knn: int = 0, self_loops: bool = True, sum: bool = True,
                 **linear_args):

        self.node_size = node_size
        self.fe_layers = fe_layers
        self.fn_layers = fn_layers

        self.pos_diffs = pos_diffs
        self.all_ef = all_ef
        self.coords = coords
        self.delta_coords = delta_coords
        self.delta_r = delta_r
        self.int_diffs = int_diffs

        self.clabels = clabels
        self.mask_fne_np = mask_fne_np

        self.fully_connected = fully_connected
        self.num_knn = num_knn
        self.self_loops = self_loops
        self.sum = sum

        self.leaky_relu_alpha = leaky_relu_alpha
        self.batch_norm = batch_norm
        self.dropout = nn.Dropout(p=dropout_p)

        # # of edge features to pass into edge network (e.g. node distances, pT difference etc.)
        num_ef = 0
        if pos_diffs:
            if delta_coords: num_ef += 3 if coords == 'cartesian' else 2
            if delta_r or all_ef: num_ef += 1  # currently can't add delta_r and all_ef edge features both together

        num_ef += int(int_diffs)
        self.num_ef = num_ef

        # edge network input is node 1 features + node 2 features + edge features (optional) + conditional labels (optional) + # particles (optional)
        fe_in_size = 2 * node_size + num_ef + clabels + mask_fne_np
        self.fe = LinearNet(self.fe_layers, input_size=fe_in_size, final_linear=False, **linear_args)

        # node network input is edge network output + node features + conditional labels (optional) + # particles (optional)
        fe_out_size = self.fe_layers[-1]
        fn_in_size = fe_out_size + node_size + clabels + mask_fne_np
        self.fn = LinearNet(self.fn_layers, input_size=fn_in_size, final_linear=True, **linear_args)  # node network output is 'linear' i.e. final layer does not apply normalization or nonlinear activations


    def forward(self, x, mask_bool=False, mask=None, labels=None, num_particles=None):
        """
        Runs through message passing

        Args:
            x (torch.Tensor): input tensor of shape [batch size, # nodes, # node features]
            mask_bool (bool, optional): use masking
            mask (torch.Tensor, optional): if using masking, tensor of masks for each node of shape [batch size, # nodes, 1 (mask)]
            labels (torch.Tensor, optional): if using conditioning labels during message passing, tensor of labels for each jet of shape [batch size, # labels]
            num_particles (torch.Tensor, optional): if using # of particles as an extra conditioning label, tensor of num particles for each jet of shape [batch size, 1]
        """
        batch_size = x.size(0)
        num_nodes = x.size(1)

        # get inputs to edge network
        A, A_mask = self._getA(x, batch_size, num_nodes, mask_bool, mask)

        num_knn = num_nodes if self.fully_connected else self.num_knn  # if fully connected num_knn is the size of the graph

        if self.clabels: A = torch.cat((A, labels[:, :self.clabels].repeat(num_nodes * num_knn, 1)), axis=1)  # add conditioning labels
        if self.mask_fne_np: A = torch.cat((A, num_particles.repeat(num_nodes * num_knn, 1)), axis=1)  # add # of real (i.e. not zero-padded) particles in the graph

        # run through edge network
        A = self.fe(A)
        A = A.view(batch_size, num_nodes, num_knn, self.fe_layers[-1])

        if mask_bool:  # if use masking, mask out 0-masked particles by multiplying them with the mask
            if self.fully_connected: A = A * mask.unsqueeze(1)
            else: A = A * A_mask.view(batch_size, self.args.num_hits, num_knn, 1)

        # aggregate and concatenate with node features
        A = torch.sum(A, 2) if self.sum else torch.mean(A, 2)
        x = torch.cat((A, x), 2).view(batch_size * num_nodes, self.fe_layers[-1] + node_size)

        if self.clabels x = torch.cat((x, labels[:, :self.clabels].repeat(num_nodes, 1)), axis=1)  # add conditioning labels
        if self.mask_fne_np: x = torch.cat((x, num_particles.repeat(num_nodes, 1)), axis=1)  # add # of real (i.e. not zero-padded) particles in the graph

        # run through node network
        x = self.fn(x)
        x = x.view(batch_size, num_nodes, self.fn_layers[-1])

        return x

    def _getA(self, x, batch_size, num_nodes, mask_bool, mask):
        """
        returns tensor of inputs to the edge networks
        """
        num_coords = 3 if self.args.coords == 'cartesian' else 2
        out_size = 2 * self.node_size + self.num_ef

        A_mask = None

        if self.fully_connected:
            x1 = x.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, node_size)
            x2 = x.repeat(1, num_nodes, 1)

            if self.pos_diffs:
                # TODO: move this logic to the Graph GAN module
                # if self.args.all_ef and not (self.D and i == 0): diffs = x2 - x1  # for first iteration of D message passing use only physical coords

                if self.all_ef: diffs = x2 - x1
                else: diffs = x2[:, :, :num_coords] - x1[:, :, :num_coords]
                dists = torch.norm(diffs + 1e-12, dim=2).unsqueeze(2)

                if self.delta_r and self.delta_coords:
                    A = torch.cat((x1, x2, diffs, dists), 2)
                elif self.delta_r or self.all_ef:
                    A = torch.cat((x1, x2, dists), 2)
                elif self.delta_coords:
                    A = torch.cat((x1, x2, diffs), 2)

                A = A.view(batch_size * num_nodes * num_nodes, out_size)
            else:
                A = torch.cat((x1, x2), 2).view(batch_size * num_nodes * num_nodes, out_size)
        else:
            x1 = x.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, node_size)

            if mask_bool:
                mul = 1e4  # multiply masked particles by this so they are not selected as a nearest neighbour
                x2 = (((1 - mul) * mask + mul) * x).repeat(1, num_nodes, 1)
            else:
                x2 = x.repeat(1, num_nodes, 1)

            # TODO: move this logic to the Graph GAN module
            # if (self.all_ef or not self.pos_diffs) and not (self.D and i == 0): diffs = x2 - x1  # for first iteration of D message passing use only physical coords
            if (self.all_ef or not self.pos_diffs): diffs = x2 - x1
            else: diffs = x2[:, :, :num_coords] - x1[:, :, :num_coords]

            dists = torch.norm(diffs + 1e-12, dim=2).reshape(batch_size, num_nodes, num_nodes)

            sorted = torch.sort(dists, dim=2)
            self_loops_idx = int(self.self_loops is False)  # if self_loops is True then 0 else 1 so that we skip the node itself in the line below if no self loops

            dists = sorted[0][:, :, self_loops_idx:self.num_knn + self_loops_idx].reshape(batch_size, num_nodes * self.num_knn, 1)
            sorted = sorted[1][:, :, self_loops_idx:self.num_knn + self_loops_idx].reshape(batch_size, num_nodes * self.num_knn, 1)

            sorted.reshape(batch_size, num_nodes * self.num_knn, 1).repeat(1, 1, node_size)

            x1_knn = x.repeat(1, 1, self.num_knn).view(batch_size, num_nodes * self.num_knn, node_size)

            if mask_bool:
                x2_knn = torch.gather(torch.cat((x, mask), dim=2), 1, sorted.repeat(1, 1, node_size + 1))
                A_mask = x2_knn[:, :, -1:]
                x2_knn = x2_knn[:, :, :-1]
            else:
                x2_knn = torch.gather(x, 1, sorted.repeat(1, 1, node_size))

            if self.args.pos_diffs:
                A = torch.cat((x1_knn, x2_knn, dists), dim=2)
            else:
                A = torch.cat((x1_knn, x2_knn), dim=2)
            # logging.debug("A \n {} \n".format(A[0]))

        return A, A_mask

    def __repr__(self):
        # TODO!
        return ""


class MPNet(nn.Module):
    def __init__(self, args, mp_iters, linear_args, mp_args):
        super().__init__()
        self.mp_iters = mp_iters

        self.first_layer_node_size = self.args.latent_node_size if self.args.latent_node_size else self.args.hidden_node_size

        if not self.args.fe1: self.args.fe1 = self.args.fe.copy()
        self.args.fn1 = self.args.fn.copy()

        # args for LinearNet layers
        linear_args = {
            'leaky_relu_alpha': self.args.leaky_relu_alpha, 'dropout_p': self.args.dropout_p, 'batch_norm': self.args.batch_norm, 'spectral_norm': self.args.spectral_norm,
        }

        # args for MPLayers
        common_args = {
            'pos_diffs': self.args.pos_diffs, 'all_ef': self.args.all_ef, 'coords': self.args.coords, 'delta_coords': self.args.deltacoords, 'delta_r': self.args.deltar, 'int_diffs': self.args.int_diffs,
            'mask_fne_np': self.args.mask_fne_np,
            'fully_connected': self.args.fully_connected, 'num_knn': self.args.num_knn, 'self_loops': self.args.self_loops, 'sum': self.sum,
        }

        # latent fully connected layer
        if args.lfc:
            self.lfc = nn.Linear(self.args.lfc_latent_size, self.args.num_hits * self.first_layer_node_size)

        # initial gen mask FCN
        if self.args.mask_learn or self.args.mask_learn_sep:
            self.fmg = LinearNet(self.args.fmg, input_size=self.first_layer_node_size, output_size=1 if self.args.mask_learn else self.args.num_hits, final_linear=True, **linear_args)

        self.MPLayers = nn.ModuleList()
        # first layer
        # TODO: FOR D FIRST LAYER SHOULD USE ONLY DELTA COORDS IF ALL EF IS TRUE
        self.MPLayers.append(MPLayer(args, self.first_layer_node_size, self.args.fe1, self.args.fn1, clabels=self.args.clabels_first_layer, **common_args, **linear_args))

        # rest of layers
        for i in range(self.args.mp_iters - 1):
            self.MPLayers.append(MPLayer(args, self.args.hidden_node_size, self.args.fe, self.args.fn, clabels=self.args.clabels, **common_args, **linear_args))

        # TODO: FINAL LAYER OUTPUT THE RIGHT # OF NODES (OPTIMIZATION)

        if self.args.glorot: self.init_params()


    def forward(self, x, labels=None, epoch=0):
        batch_size = x.shape[0]

        logging.debug(f"x: {x}")

        if self.args.lfc:
            x = self.lfc(x).reshape(batch_size, self.args.num_hits, self.first_layer_node_size)

        mask_bool, mask, nump = self._get_mask(x)

        # message passing
        for i in range(self.args.mp_iters):
            x = self.MPLayers[i](x, mask_bool, mask, labels, nump)

        x = torch.tanh(x[:, :, :self.args.node_feat_size]) if self.args.gtanh else x[:, :, :self.args.node_feat_size]

        if mask_bool:
            if self.args.mask_feat_bin and (self.args.mask_learn or self.args.mask_learn_sep): mask = (x[:, :, 3:4] < 0).float()     # inversing mask sign for positive mask initializations
            x = torch.cat((x, mask - 0.5), dim=2)

        return x


    def _get_mask(self, x):
        """
        Develops mask for input tensor `x` depending on the chosen masking strategy

        returns:
            mask_bool (bool): is masking being used
            mask (torch.Tensor): if mask_bool then tensor of masks of shape [batch size, # nodes, 1 (mask)], if not mask_bool then None
            nump (torch.Tensor): if mask_bool then tensor of # of particles per jet of shape [batch size, 1 (num particles)], if not mask_bool then None
        """

        mask_bool = (self.args.mask_learn or self.args.mask_c or self.args.mask_learn_sep) and epoch >= self.args.mask_epoch

        if self.args.mask_learn:
            mask = self.fmg(x)
            mask = torch.sign(mask) if self.args.mask_learn_bin else torch.sigmoid(mask)

        if self.args.mask_c:
            nump = (labels[:, self.args.clabels] * self.args.num_hits).int() - 1
            mask = (x[:, :, 0].argsort(1).argsort(1) <= nump.unsqueeze(1)).unsqueeze(2).float()
            logging.debug("x \n {} \n num particles \n {} \n gen mask \n {}".format(x[:2, :, 0], nump[:2], mask[:2, :, 0]))

        if self.args.mask_fne_np:
            nump = torch.mean(mask, dim=1)
            logging.debug("nump \n {}".format(nump[:2]))

        if self.args.mask_learn_sep:
            nump = x[:, -1, :]
            x = x[:, :-1, :]

            nump = self.fmg(nump)
            nump = torch.argmax(nump, dim=1)
            mask = (x[:, :, 0].argsort(1).argsort(1) <= nump.unsqueeze(1)).unsqueeze(2).float()

        if not mask_bool:
            mask = None
            nump = None

        return mask_bool, mask, nump


    def init_params(self):
        logging.info("glorot-ing")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight, self.args.glorot)

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __repr__(self):
        # TODO!

        # if(self.args.lfc):
        #     logging.info("lfcn: ")
        #     logging.info(self.lfc)
        #
        # logging.info("fe: ")
        # logging.info(self.fe)
        #
        # logging.info("fn: ")
        # logging.info(self.fn)
        #
        # if(self.args.dea):
        #     logging.info("fnd: ")
        #     logging.info(self.fnd)
        #
        # if self.G and (hasattr(self.args, 'mask_learn') and self.args.mask_learn) or (hasattr(self.args, 'mask_learn_sep') and self.args.mask_learn_sep):
        #     logging.info("fmg: ")
        #     logging.info(self.fmg)



class MPGenerator(nn.Module):
    def __init__(self, args):
        super(MPGenerator, self).__init__()
        self.args = args
        self.args.spectral_norm = self.args.spectral_norm_gen
        self.args.batch_norm = self.args.batch_norm_gen
        self.args.mp_iters = self.args.mp_iters_gen
        self.args.dropout_p = self.args.gen_dropout
        self.args.fe1 = self.args.fe1g

        self.first_layer_node_size = self.args.latent_node_size if self.args.latent_node_size else self.args.hidden_node_size

        if not self.args.fe1: self.args.fe1 = self.args.fe.copy()
        self.args.fn1 = self.args.fn.copy()

        # args for LinearNet layers
        linear_args = {
            'leaky_relu_alpha': self.args.leaky_relu_alpha, 'dropout_p': self.args.dropout_p, 'batch_norm': self.args.batch_norm, 'spectral_norm': self.args.spectral_norm,
        }

        # args for MPLayers
        common_args = {
            'pos_diffs': self.args.pos_diffs, 'all_ef': self.args.all_ef, 'coords': self.args.coords, 'delta_coords': self.args.deltacoords, 'delta_r': self.args.deltar, 'int_diffs': self.args.int_diffs,
            'mask_fne_np': self.args.mask_fne_np,
            'fully_connected': self.args.fully_connected, 'num_knn': self.args.num_knn, 'self_loops': self.args.self_loops, 'sum': self.sum,
        }

        # latent fully connected layer
        if args.lfc:
            self.lfc = nn.Linear(self.args.lfc_latent_size, self.args.num_hits * self.first_layer_node_size)

        # initial gen mask FCN
        if self.args.mask_learn or self.args.mask_learn_sep:
            self.fmg = LinearNet(self.args.fmg, input_size=self.first_layer_node_size, output_size=1 if self.args.mask_learn else self.args.num_hits, final_linear=True, **linear_args)

        self.MPLayers = nn.ModuleList()
        # first layer
        # TODO: FOR D FIRST LAYER SHOULD USE ONLY DELTA COORDS IF ALL EF IS TRUE
        self.MPLayers.append(MPLayer(args, self.first_layer_node_size, self.args.fe1, self.args.fn1, clabels=self.args.clabels_first_layer, **common_args, **linear_args))

        # rest of layers
        for i in range(self.args.mp_iters - 1):
            self.MPLayers.append(MPLayer(args, self.args.hidden_node_size, self.args.fe, self.args.fn, clabels=self.args.clabels, **common_args, **linear_args))

        # TODO: FINAL LAYER OUTPUT THE RIGHT # OF NODES (OPTIMIZATION)

        if self.args.glorot: self.init_params()


    def forward(self, x, labels=None, epoch=0):
        batch_size = x.shape[0]

        logging.debug(f"x: {x}")

        if self.args.lfc:
            x = self.lfc(x).reshape(batch_size, self.args.num_hits, self.first_layer_node_size)

        mask_bool, mask, nump = self._get_mask(x)

        # message passing
        for i in range(self.args.mp_iters):
            x = self.MPLayers[i](x, mask_bool, mask, labels, nump)

        x = torch.tanh(x[:, :, :self.args.node_feat_size]) if self.args.gtanh else x[:, :, :self.args.node_feat_size]

        if mask_bool:
            if self.args.mask_feat_bin and (self.args.mask_learn or self.args.mask_learn_sep): mask = (x[:, :, 3:4] < 0).float()     # inversing mask sign for positive mask initializations
            x = torch.cat((x, mask - 0.5), dim=2)

        return x


    def _get_mask(self, x):
        """
        Develops mask for input tensor `x` depending on the chosen masking strategy

        returns:
            mask_bool (bool): is masking being used
            mask (torch.Tensor): if mask_bool then tensor of masks of shape [batch size, # nodes, 1 (mask)], if not mask_bool then None
            nump (torch.Tensor): if mask_bool then tensor of # of particles per jet of shape [batch size, 1 (num particles)], if not mask_bool then None
        """

        mask_bool = (self.args.mask_learn or self.args.mask_c or self.args.mask_learn_sep) and epoch >= self.args.mask_epoch

        if self.args.mask_learn:
            mask = self.fmg(x)
            mask = torch.sign(mask) if self.args.mask_learn_bin else torch.sigmoid(mask)

        if self.args.mask_c:
            nump = (labels[:, self.args.clabels] * self.args.num_hits).int() - 1
            mask = (x[:, :, 0].argsort(1).argsort(1) <= nump.unsqueeze(1)).unsqueeze(2).float()
            logging.debug("x \n {} \n num particles \n {} \n gen mask \n {}".format(x[:2, :, 0], nump[:2], mask[:2, :, 0]))

        if self.args.mask_fne_np:
            nump = torch.mean(mask, dim=1)
            logging.debug("nump \n {}".format(nump[:2]))

        if self.args.mask_learn_sep:
            nump = x[:, -1, :]
            x = x[:, :-1, :]

            nump = self.fmg(nump)
            nump = torch.argmax(nump, dim=1)
            mask = (x[:, :, 0].argsort(1).argsort(1) <= nump.unsqueeze(1)).unsqueeze(2).float()

        if not mask_bool:
            mask = None
            nump = None

        return mask_bool, mask, nump


    def init_params(self):
        logging.info("glorot-ing")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight, self.args.glorot)

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __repr__(self):
        # TODO!

        # if(self.args.lfc):
        #     logging.info("lfcn: ")
        #     logging.info(self.lfc)
        #
        # logging.info("fe: ")
        # logging.info(self.fe)
        #
        # logging.info("fn: ")
        # logging.info(self.fn)
        #
        # if(self.args.dea):
        #     logging.info("fnd: ")
        #     logging.info(self.fnd)
        #
        # if self.G and (hasattr(self.args, 'mask_learn') and self.args.mask_learn) or (hasattr(self.args, 'mask_learn_sep') and self.args.mask_learn_sep):
        #     logging.info("fmg: ")
        #     logging.info(self.fmg)


class MPDiscriminator(nn.Module):
    def __init__(self, gen, args):
        super(MPDiscriminator, self).__init__()
        self.args = args

        self.args.spectral_norm = self.args.spectral_norm_disc
        self.args.batch_norm = self.args.batch_norm_disc
        self.args.mp_iters = self.args.mp_iters_disc
        self.args.dropout_p = self.args.disc_dropout
        self.args.fe1 = self.args.fe1d

        self.first_layer_node_size = self.args.node_feat_size

        if not self.args.fe1: self.args.fe1 = self.args.fe.copy()
        self.args.fn1 = self.args.fn.copy()

        # args for LinearNet layers
        linear_args = {
            'leaky_relu_alpha': self.args.leaky_relu_alpha, 'dropout_p': self.args.dropout_p, 'batch_norm': self.args.batch_norm, 'spectral_norm': self.args.spectral_norm,
        }

        # args for MPLayers
        common_args = {
            'pos_diffs': self.args.pos_diffs, 'all_ef': self.args.all_ef, 'coords': self.args.coords, 'delta_coords': self.args.deltacoords, 'delta_r': self.args.deltar, 'int_diffs': self.args.int_diffs,
            'mask_fne_np': self.args.mask_fne_np,
            'fully_connected': self.args.fully_connected, 'num_knn': self.args.num_knn, 'self_loops': self.args.self_loops, 'sum': self.sum,
        }

        self.MPLayers = nn.ModuleList()
        # first layer
        # TODO: FOR D FIRST LAYER SHOULD USE ONLY DELTA COORDS IF ALL EF IS TRUE
        self.MPLayers.append(MPLayer(args, self.first_layer_node_size, self.args.fe1, self.args.fn1, clabels=self.args.clabels_first_layer, **common_args, **linear_args))

        # rest of layers
        for i in range(self.args.mp_iters - 1):
            self.MPLayers.append(MPLayer(args, self.args.hidden_node_size, self.args.fe, self.args.fn, clabels=self.args.clabels, **common_args, **linear_args))

        # TODO: FINAL LAYER OUTPUT THE RIGHT # OF NODES (OPTIMIZATION)

        # final disc FCN

        if self.args.dea:
            self.fnd = LinearNet(self.args.fnd, input_size=self.args.hidden_node_size + int(self.args.mask_fnd_np), output_size=1, final_linear=True, **linear_args)

        if self.args.glorot: self.init_params()


    def forward(self, x, labels=None, epoch=0):
        batch_size = x.shape[0]

        logging.debug(f"x: {x}")

        mask_bool, mask, nump = self._get_mask(x)

        # message passing
        for i in range(self.args.mp_iters):
            x = self.MPLayers[i](x, mask_bool, mask, labels, nump)


        do_mean = not (self.args.sum and self.args.dea)
        if mask_bool:
            x = x * mask
            x = torch.sum(x, 1)
            if do_mean: x = x / (torch.sum(mask, 1) + 1e-12)
        else:
            x = torch.mean(x, 1) if do_mean else torch.sum(x, 1)

        if self.args.dea:  # feed into final FC network
            if self.args.mask_fnd_np:
                num_particles = torch.mean(mask, dim=1)
                x = torch.cat((num_particles, x), dim=1)

            x = self.fnd(x)
        else:
            x = x[:, :, :1]

        return x if (self.args.loss == 'w' or self.args.loss == 'hinge') else torch.sigmoid(x)


    def _get_mask(self, x):
        """
        Develops mask for input tensor `x` depending on the chosen masking strategy

        returns:
            mask_bool (bool): is masking being used
            mask (torch.Tensor): if mask_bool then tensor of masks of shape [batch size, # nodes, 1 (mask)], if not mask_bool then None
            nump (torch.Tensor): if mask_bool then tensor of # of particles per jet of shape [batch size, 1 (num particles)], if not mask_bool then None
        """

        mask_bool = (self.args.mask_manual or self.args.mask_real_only or self.args.mask_learn or self.args.mask_c or self.args.mask_learn_sep) and epoch >= self.args.mask_epoch

        if mask_bool or self.args.mask_fnd_np: mask = x[:, :, 3:4] + 0.5
        if self.args.mask_manual or self.args.mask_learn or self.args.mask_c or self.args.mask_learn_sep: x = x[:, :, :3]

        if self.args.mask_fne_np:
            nump = torch.mean(mask, dim=1)
            logging.debug("nump \n {}".format(nump[:2]))

        if not mask_bool:
            mask = None
            nump = None

        return mask_bool, mask, nump


    def init_params(self):
        logging.info("glorot-ing")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight, self.args.glorot)

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __repr__(self):
        # TODO!

        # if(self.args.lfc):
        #     logging.info("lfcn: ")
        #     logging.info(self.lfc)
        #
        # logging.info("fe: ")
        # logging.info(self.fe)
        #
        # logging.info("fn: ")
        # logging.info(self.fn)
        #
        # if(self.args.dea):
        #     logging.info("fnd: ")
        #     logging.info(self.fnd)
        #
        # if self.G and (hasattr(self.args, 'mask_learn') and self.args.mask_learn) or (hasattr(self.args, 'mask_learn_sep') and self.args.mask_learn_sep):
        #     logging.info("fmg: ")
        #     logging.info(self.fmg)




#
# class Graph_GAN(nn.Module):
#     def __init__(self, gen, args):
#         super(Graph_GAN, self).__init__()
#         self.args = args
#
#         self.G = gen
#         self.D = not gen
#
#         self.args.spectral_norm = self.args.spectral_norm_gen if self.G else self.args.spectral_norm_disc
#         self.args.batch_norm = self.args.batch_norm_gen if self.G else self.args.batch_norm_disc
#         self.args.mp_iters = self.args.mp_iters_gen if self.G else self.args.mp_iters_disc
#         self.args.fe1 = self.args.fe1g if self.G else self.args.fe1d
#         if self.G: self.args.dea = False
#         if self.D: self.args.lfc = False
#
#
#         if self.G: self.first_layer_node_size = self.args.latent_node_size if self.args.latent_node_size else self.args.hidden_node_size
#         else: self.first_layer_node_size = self.args.node_feat_size
#
#         if not self.args.fe1: self.args.fe1 = self.args.fe.copy()
#         self.args.fn1 = self.args.fn.copy()
#
#
#         # setting up # of nodes in each network layer
#
#         anc = 0     # anc - # of extra params to pass into edge network
#         if self.args.pos_diffs:
#             if self.args.deltacoords:
#                 if self.args.coords == 'cartesian':
#                     anc += 3
#                 elif self.args.coords == 'polar' or self.args.coords == 'polarrel':
#                     anc += 2
#             if self.args.deltar:
#                 anc += 1
#
#         anc += int(self.args.int_diffs)
#
#         if args.lfc: self.lfc = nn.Linear(self.args.lfc_latent_size, self.args.num_hits * self.first_layer_node_size)
#
#         # edge and node networks
#         # both are ModuleLists of ModuleLists
#         # with shape: # of MP layers   X   # of FC layers in each MP layer
#
#         self.fe = nn.ModuleList()
#         self.fn = nn.ModuleList()
#
#         if(self.args.batch_norm):
#             self.bne = nn.ModuleList()
#             self.bnn = nn.ModuleList()
#
#
#         # building first MP layer networks:
#
#         self.args.fe1_in_size = 2 * self.first_layer_node_size + anc + self.args.clabels_first_layer + self.args.mask_fne_np
#         self.args.fe1.insert(0, self.args.fe1_in_size)
#         self.args.fe1_out_size = self.args.fe1[-1]
#
#         fe_iter = nn.ModuleList()
#         if self.args.batch_norm: bne = nn.ModuleList()
#         for j in range(len(self.args.fe1) - 1):
#             linear = nn.Linear(self.args.fe1[j], self.args.fe1[j + 1])
#             fe_iter.append(linear)
#             if self.args.batch_norm: bne.append(nn.BatchNorm1d(self.args.fe1[j + 1]))
#
#         self.fe.append(fe_iter)
#         if self.args.batch_norm: self.bne.append(bne)
#
#
#         self.args.fn1.insert(0, self.args.fe1_out_size + self.first_layer_node_size + self.args.clabels_first_layer + self.args.mask_fne_np)
#         self.args.fn1.append(self.args.hidden_node_size)
#
#         fn_iter = nn.ModuleList()
#         if self.args.batch_norm: bnn = nn.ModuleList()
#         for j in range(len(self.args.fn1) - 1):
#             linear = nn.Linear(self.args.fn1[j], self.args.fn1[j + 1])
#             fn_iter.append(linear)
#             if self.args.batch_norm: bnn.append(nn.BatchNorm1d(self.args.fn1[j + 1]))
#
#         self.fn.append(fn_iter)
#         if self.args.batch_norm: self.bnn.append(bnn)
#
#
#         # building networks for the rest of the MP layers:
#
#         self.args.fe_in_size = 2 * self.args.hidden_node_size + anc + self.args.clabels_hidden_layers + self.args.mask_fne_np
#         self.args.fe.insert(0, self.args.fe_in_size)
#         self.args.fe_out_size = self.args.fe[-1]
#
#         self.args.fn.insert(0, self.args.fe_out_size + self.args.hidden_node_size + self.args.clabels_hidden_layers + self.args.mask_fne_np)
#         self.args.fn.append(self.args.hidden_node_size)
#
#         for i in range(self.args.mp_iters - 1):
#             fe_iter = nn.ModuleList()
#             if self.args.batch_norm: bne = nn.ModuleList()
#             for j in range(len(self.args.fe) - 1):
#                 linear = nn.Linear(self.args.fe[j], self.args.fe[j + 1])
#                 fe_iter.append(linear)
#                 if self.args.batch_norm: bne.append(nn.BatchNorm1d(self.args.fe[j + 1]))
#
#             self.fe.append(fe_iter)
#             if self.args.batch_norm: self.bne.append(bne)
#
#
#             fn_iter = nn.ModuleList()
#             if self.args.batch_norm: bnn = nn.ModuleList()
#             for j in range(len(self.args.fn) - 1):
#                 linear = nn.Linear(self.args.fn[j], self.args.fn[j + 1])
#                 fn_iter.append(linear)
#                 if self.args.batch_norm: bnn.append(nn.BatchNorm1d(self.args.fn[j + 1]))
#
#             self.fn.append(fn_iter)
#             if self.args.batch_norm: self.bnn.append(bnn)
#
#
#         # final disc FCN
#
#         if(self.args.dea):
#             self.args.fnd.insert(0, self.args.hidden_node_size + int(self.args.mask_fnd_np))
#             self.args.fnd.append(1)
#
#             self.fnd = nn.ModuleList()
#             self.bnd = nn.ModuleList()
#             for i in range(len(self.args.fnd) - 1):
#                 linear = nn.Linear(self.args.fnd[i], self.args.fnd[i + 1])
#                 self.fnd.append(linear)
#                 if self.args.batch_norm: self.bnd.append(nn.BatchNorm1d(self.args.fnd[i + 1]))
#
#
#         # initial gen mask FCN
#
#         if self.G and (hasattr(self.args, 'mask_learn') and self.args.mask_learn) or (hasattr(self.args, 'mask_learn_sep') and self.args.mask_learn_sep):
#             self.args.fmg.insert(0, self.first_layer_node_size)
#             self.args.fmg.append(1 if self.args.mask_learn else self.args.num_hits)
#
#             self.fmg = nn.ModuleList()
#             self.bnmg = nn.ModuleList()
#             for i in range(len(self.args.fmg) - 1):
#                 linear = nn.Linear(self.args.fmg[i], self.args.fmg[i + 1])
#                 self.fmg.append(linear)
#                 if self.args.batch_norm: self.bnmg.append(nn.BatchNorm1d(self.args.fmg[i + 1]))
#
#
#         p = self.args.gen_dropout if self.G else self.args.disc_dropout
#         self.dropout = nn.Dropout(p=p)
#
#         if self.args.glorot: self.init_params()
#
#         if self.args.spectral_norm:
#             for ml in self.fe:
#                 for i in range(len(ml)):
#                     ml[i] = SpectralNorm(ml[i])
#
#             for ml in self.fn:
#                 for i in range(len(ml)):
#                     ml[i] = SpectralNorm(ml[i])
#
#             if self.args.dea:
#                 for i in range(len(self.fnd)):
#                     self.fnd[i] = SpectralNorm(self.fnd[i])
#
#             if self.args.mask_learn:
#                 for i in range(len(self.fmg)):
#                     self.fmg[i] = SpectralNorm(self.fmg[i])
#
#         if(self.args.lfc):
#             logging.info("lfcn: ")
#             logging.info(self.lfc)
#
#         logging.info("fe: ")
#         logging.info(self.fe)
#
#         logging.info("fn: ")
#         logging.info(self.fn)
#
#         if(self.args.dea):
#             logging.info("fnd: ")
#             logging.info(self.fnd)
#
#         if self.G and (hasattr(self.args, 'mask_learn') and self.args.mask_learn) or (hasattr(self.args, 'mask_learn_sep') and self.args.mask_learn_sep):
#             logging.info("fmg: ")
#             logging.info(self.fmg)
#
#
#     def forward(self, x, labels=None, epoch=0):
#         batch_size = x.shape[0]
#
#         logging.debug(f"x: {x}")
#
#         if self.args.lfc:
#             x = self.lfc(x).reshape(batch_size, self.args.num_hits, self.first_layer_node_size)
#             logging.debug(f"LFC'd x: {x}")
#
#         try:
#             mask_bool = (self.D and (self.args.mask_manual or self.args.mask_real_only or self.args.mask_learn or self.args.mask_c or self.args.mask_learn_sep)) \
#                         or (self.G and (self.args.mask_learn or self.args.mask_c or self.args.mask_learn_sep)) \
#                         and epoch >= self.args.mask_epoch
#
#             if self.D and (mask_bool or self.args.mask_fnd_np): mask = x[:, :, 3:4] + 0.5
#             if self.D and (self.args.mask_manual or self.args.mask_learn or self.args.mask_c or self.args.mask_learn_sep): x = x[:, :, :3]
#
#             if self.G and self.args.mask_learn:
#                 mask = F.leaky_relu(self.fmg[0](x), negative_slope=self.args.leaky_relu_alpha)
#                 if(self.args.batch_norm): mask = self.bnmg[0](mask)
#                 mask = self.dropout(mask)
#                 for i in range(len(self.fmg) - 1):
#                     mask = F.leaky_relu(self.fmg[i + 1](mask), negative_slope=self.args.leaky_relu_alpha)
#                     if(self.args.batch_norm): mask = self.bnmg[i](mask)
#                     mask = self.dropout(mask)
#
#                 mask = torch.sign(mask) if self.args.mask_learn_bin else torch.sigmoid(mask)
#                 logging.debug("gen mask \n {}".format(mask[:2, :, 0]))
#
#             if self.G and self.args.mask_c:
#                 nump = (labels[:, self.args.clabels] * self.args.num_hits).int() - 1
#                 mask = (x[:, :, 0].argsort(1).argsort(1) <= nump.unsqueeze(1)).unsqueeze(2).float()
#                 logging.debug("x \n {} \n num particles \n {} \n gen mask \n {}".format(x[:2, :, 0], nump[:2], mask[:2, :, 0]))
#
#             if self.args.mask_fne_np:
#                 nump = torch.mean(mask, dim=1)
#                 logging.debug("nump \n {}".format(nump[:2]))
#
#             if self.G and self.args.mask_learn_sep:
#                 nump = x[:, -1, :]
#                 x = x[:, :-1, :]
#
#                 for i in range(len(self.fmg)):
#                     nump = F.leaky_relu(self.fmg[i](nump), negative_slope=self.args.leaky_relu_alpha)
#                     if(self.args.batch_norm): nump = self.bnmg[i](nump)
#                     nump = self.dropout(nump)
#
#                 nump = torch.argmax(nump, dim=1)
#                 mask = (x[:, :, 0].argsort(1).argsort(1) <= nump.unsqueeze(1)).unsqueeze(2).float()
#
#                 logging.debug("x \n {} \n num particles \n {} \n gen mask \n {}".format(x[:2, :, 0], nump[:2], mask[:2, :, 0]))
#
#         except AttributeError:
#             mask_bool = False
#
#         if not mask_bool: mask = None
#
#         for i in range(self.args.mp_iters):
#             clabel_iter = self.args.clabels and ((i == 0 and self.args.clabels_first_layer) or (i and self.args.clabels_hidden_layers))
#
#             node_size = x.size(2)
#             fe_in_size = self.args.fe_in_size if i else self.args.fe1_in_size
#             fe_out_size = self.args.fe_out_size if i else self.args.fe1_out_size
#
#             if clabel_iter: fe_in_size -= self.args.clabels
#             if self.args.mask_fne_np: fe_in_size -= 1
#
#             # message passing
#             A, A_mask = self.getA(x, i, batch_size, fe_in_size, mask_bool, mask)
#
#             # logging.debug('A \n {} \n A_mask \n {}'.format(A[:2, :10], A_mask[:2, :10]))
#
#             num_knn = self.args.num_hits if (hasattr(self.args, 'fully_connected') and self.args.fully_connected) else self.args.num_knn
#
#             if (A != A).any(): logging.warning("Nan values in A \n x: \n {} \n A: \n {}".format(x, A))
#
#             if clabel_iter: A = torch.cat((A, labels[:, :self.args.clabels].repeat(self.args.num_hits * num_knn, 1)), axis=1)
#             if self.args.mask_fne_np: A = torch.cat((A, nump.repeat(self.args.num_hits * num_knn, 1)), axis=1)
#
#             for j in range(len(self.fe[i])):
#                 A = F.leaky_relu(self.fe[i][j](A), negative_slope=self.args.leaky_relu_alpha)
#                 if(self.args.batch_norm): A = self.bne[i][j](A)  # try before activation
#                 A = self.dropout(A)
#
#             if (A != A).any(): logging.warning("Nan values in A after message passing \n x: \n {} \n A: \n {}".format(x, A))
#
#             # message aggregation into new features
#             A = A.view(batch_size, self.args.num_hits, num_knn, fe_out_size)
#             if mask_bool:
#                 if self.args.fully_connected: A = A * mask.unsqueeze(1)
#                 else: A = A * A_mask.reshape(batch_size, self.args.num_hits, num_knn, 1)
#
#             # logging.debug('A \n {}'.format(A[:2, :10]))
#
#             A = torch.sum(A, 2) if self.args.sum else torch.mean(A, 2)
#             x = torch.cat((A, x), 2).view(batch_size * self.args.num_hits, fe_out_size + node_size)
#
#             if (x != x).any(): logging.warning("Nan values in x after message passing \n x: \n {} \n A: \n {}".format(x, A))
#
#             if clabel_iter: x = torch.cat((x, labels[:, :self.args.clabels].repeat(self.args.num_hits, 1)), axis=1)
#             if self.args.mask_fne_np: x = torch.cat((x, nump.repeat(self.args.num_hits, 1)), axis=1)
#
#             for j in range(len(self.fn[i]) - 1):
#                 x = F.leaky_relu(self.fn[i][j](x), negative_slope=self.args.leaky_relu_alpha)
#                 if(self.args.batch_norm): x = self.bnn[i][j](x)
#                 x = self.dropout(x)
#
#             x = self.dropout(self.fn[i][-1](x))
#             x = x.view(batch_size, self.args.num_hits, self.args.hidden_node_size)
#
#             if (x != x).any(): logging.warning("Nan values in x after fn \n x: \n {} \n A: \n {}".format(x, A))
#
#         if(self.G):
#             x = torch.tanh(x[:, :, :self.args.node_feat_size]) if self.args.gtanh else x[:, :, :self.args.node_feat_size]
#             if mask_bool:
#                 x = torch.cat((x, mask - 0.5), dim=2)
#             if hasattr(self.args, 'mask_feat_bin') and self.args.mask_feat_bin:
#                 mask = (x[:, :, 3:4] < 0).float() - 0.5     # inversing mask sign for positive mask initializations
#                 x = torch.cat((x[:, :, :3], mask), dim=2)
#             return x
#         else:
#             if(self.args.dea):
#                 if mask_bool:
#                     x = x * mask
#                     x = torch.sum(x, 1)
#                     if not self.args.sum: x = x / (torch.sum(mask, 1) + 1e-12)
#                 else: x = torch.sum(x, 1) if self.args.sum else torch.mean(x, 1)
#
#                 if hasattr(self.args, 'mask_fnd_np') and self.args.mask_fnd_np:
#                     num_particles = torch.mean(mask, dim=1)
#                     x = torch.cat((num_particles, x), dim=1)
#
#                 for i in range(len(self.fnd) - 1):
#                     x = F.leaky_relu(self.fnd[i](x), negative_slope=self.args.leaky_relu_alpha)
#                     if(self.args.batch_norm): x = self.bnd[i](x)
#                     x = self.dropout(x)
#
#                 x = self.dropout(self.fnd[-1](x))
#             else:
#                 x = x[:, :, :1]
#                 if mask_bool:
#                     logging.debug("D output pre mask")
#                     logging.debug(mask[:2, :, 0])
#                     logging.debug(x[:2, :, 0])
#                     x = x * mask
#                     logging.debug("post mask")
#                     logging.debug(x[:2, :, 0])
#                     x = torch.sum(x, 1) / (torch.sum(mask, 1) + 1e-12)
#                 else:
#                     x = torch.mean(x, 1)
#
#             return x if (self.args.loss == 'w' or self.args.loss == 'hinge') else torch.sigmoid(x)
#
#     def getA(self, x, i, batch_size, fe_in_size, mask_bool, mask):
#         node_size = x.size(2)
#         num_coords = 3 if self.args.coords == 'cartesian' else 2
#
#         A_mask = None
#
#         if self.args.fully_connected:
#             x1 = x.repeat(1, 1, self.args.num_hits).view(batch_size, self.args.num_hits * self.args.num_hits, node_size)
#             x2 = x.repeat(1, self.args.num_hits, 1)
#
#             if self.args.pos_diffs:
#                 if self.args.all_ef and not (self.D and i == 0): diffs = x2 - x1  # for first iteration of D message passing use only physical coords
#                 else: diffs = x2[:, :, :num_coords] - x1[:, :, :num_coords]
#                 dists = torch.norm(diffs + 1e-12, dim=2).unsqueeze(2)
#
#                 if self.args.deltar and self.args.deltacoords:
#                     A = torch.cat((x1, x2, diffs, dists), 2)
#                 elif self.args.deltar:
#                     A = torch.cat((x1, x2, dists), 2)
#                 elif self.args.deltacoords:
#                     A = torch.cat((x1, x2, diffs), 2)
#
#                 A = A.view(batch_size * self.args.num_hits * self.args.num_hits, fe_in_size)
#             else:
#                 A = torch.cat((x1, x2), 2).view(batch_size * self.args.num_hits * self.args.num_hits, fe_in_size)
#
#         else:
#             x1 = x.repeat(1, 1, self.args.num_hits).view(batch_size, self.args.num_hits * self.args.num_hits, node_size)
#
#             if mask_bool:
#                 mul = 1e4  # multiply masked particles by this so they are not selected as a nearest neighbour
#                 x2 = (((1 - mul) * mask + mul) * x).repeat(1, self.args.num_hits, 1)
#             else:
#                 x2 = x.repeat(1, self.args.num_hits, 1)
#
#             if (self.args.all_ef or not self.args.pos_diffs) and not (self.D and i == 0): diffs = x2 - x1  # for first iteration of D message passing use only physical coords
#             else: diffs = x2[:, :, :num_coords] - x1[:, :, :num_coords]
#
#             dists = torch.norm(diffs + 1e-12, dim=2).reshape(batch_size, self.args.num_hits, self.args.num_hits)
#
#             sorted = torch.sort(dists, dim=2)
#             self_loops = int(self.args.self_loops is False)
#
#             # logging.debug("x \n {} \n x1 \n {} \n x2 \n {} \n diffs \n {} \n dists \n {} \n sorted[0] \n {} \n sorted[1] \n {}".format(x[0], x1[0], x2[0], diffs[0], dists[0], sorted[0][0], sorted[0][1]))
#
#             dists = sorted[0][:, :, self_loops:self.args.num_knn + self_loops].reshape(batch_size, self.args.num_hits * self.args.num_knn, 1)
#             sorted = sorted[1][:, :, self_loops:self.args.num_knn + self_loops].reshape(batch_size, self.args.num_hits * self.args.num_knn, 1)
#
#             sorted.reshape(batch_size, self.args.num_hits * self.args.num_knn, 1).repeat(1, 1, node_size)
#
#             x1_knn = x.repeat(1, 1, self.args.num_knn).view(batch_size, self.args.num_hits * self.args.num_knn, node_size)
#
#             if mask_bool:
#                 x2_knn = torch.gather(torch.cat((x, mask), dim=2), 1, sorted.repeat(1, 1, node_size + 1))
#                 A_mask = x2_knn[:, :, -1:]
#                 x2_knn = x2_knn[:, :, :-1]
#             else:
#                 x2_knn = torch.gather(x, 1, sorted.repeat(1, 1, node_size))
#
#             if self.args.pos_diffs:
#                 A = torch.cat((x1_knn, x2_knn, dists), dim=2)
#             else:
#                 A = torch.cat((x1_knn, x2_knn), dim=2)
#             # logging.debug("A \n {} \n".format(A[0]))
#
#         return A, A_mask
#
#     def init_params(self):
#         logging.info("glorot-ing")
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 torch.nn.init.xavier_uniform(m.weight, self.args.glorot)
#
#     def load(self, backup):
#         for m_from, m_to in zip(backup.modules(), self.modules()):
#             if isinstance(m_to, nn.Linear):
#                 m_to.weight.data = m_from.weight.data.clone()
#                 if m_to.bias is not None:
#                     m_to.bias.data = m_from.bias.data.clone()
#
#     def reset_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.reset_parameters()
