from itertools import chain, repeat
from collections import defaultdict

import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable

def in_jupyter():
    try:
        from IPython import get_ipython
        from ipykernel.zmqshell import ZMQInteractiveShell
        assert isinstance(get_ipython(), ZMQInteractiveShell)
    except Exception:
        return False
    return True

try:
    if in_jupyter():
        # from tqdm import tqdm as pbar
        from tqdm import tqdm_notebook as pbar
    else:
        from tqdm import tqdm as pbar
except ImportError:
    def pbar(it, *a, **kw):
        return it

# avoid log(0)
_eps = 1e-15

def _freeze(*args):
    for module in args:
        for p in module.parameters():
            p.requires_grad = False
def _unfreeze(*args):
    for module in args:
        for p in module.parameters():
            p.requires_grad = True

def _take_epochs(X, n_epochs):
    """Get a fractional number of epochs from X, rounded to the batch
    X: torch.utils.DataLoader
    n_epochs: number of iterations through the data.
    """
    n_batches = int(np.ceil(len(X)*n_epochs))
    n_shuffles = int(np.ceil(n_epochs))
    return islice(chain.from_iterable(repeat(X,n_shuffles)),n_batches)

def _wrap(x):
    """ensure x is a Variable."""
    if isinstance(x, Variable):
        return x
    # if x isn't a Tensor, attempt to construct one from it
    if not isinstance(x, torch._TensorBase):
        x = torch.Tensor(x)
    return Variable(x)

class AlphaGAN(nn.Module):
    def __init__(self, encoder, generator, D, C, latent_dim, lambd=1):
        """α-GAN as described in Rosca, Mihaela, et al.
            "Variational Approaches for Auto-Encoding Generative Adversarial Networks."
            arXiv preprint arXiv:1706.04987 (2017).
        encoder: nn.Module mapping X to Z
        generator: nn.Module mapping Z to X
        D: nn.module discriminating real from generated/reconstructed X
        C: nn.module discriminating prior from posterior Z
        latent_dim: dimensionality of Z
        lambd: scale parameter for the generator distribution
            a.k.a weight for the reconstruction loss
        """
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.D = D
        self.C = C
        self.latent_dim = latent_dim
        self.lambd = lambd
        # self.rec_loss = nn.L1Loss()

    def sample_prior(self, n):
        """Sample self.latent_dim-dimensional unit normal.
        n: batch size
        """
        return Variable(torch.randn(n, self.latent_dim), requires_grad=False)

    def rec_loss(self, x_rec, x):
        """L1 reconstruction error or Laplace log likelihood"""
        return self.lambd * (x_rec-x).abs().mean()

    def autoencoder_loss(self, x):
        """Return reconstruction loss, adversarial loss"""
        z_prior, x_prior = self(len(x), mode='sample')
        z, x_rec = self(x)
        xs = torch.cat((x_prior, x_rec), 0)
        return dict(
            reconstruction = self.rec_loss(x_rec, x),
            adversarial = -(self.D(xs) + _eps).log().mean(),
            code = -(self.C(z) + _eps).log().mean()
        )

    def discriminator_loss(self, x):
        """Return discriminator (D) loss, code discriminator (C) loss"""
        z_prior, x_prior = self(len(x), mode='sample')
        z, x_rec = self(x)
        xs = torch.cat((x_prior, x_rec), 0)
        return dict(
            D = - (self.D(x) + _eps).log().mean()
                - (1 - self.D(xs) + _eps).log().mean(),
            C = - (self.C(z_prior) + _eps).log().mean()
                - (1 - self.C(z) + _eps).log().mean()
        )

    def fit_step(self, X, loss_fn, optimizer=None, n_epochs=1):
        """Optimize for one epoch.
        X: torch.utils.DataLoader
        loss_fn: return dict of loss component Variables
        optimizer: if None, just compute loss components (e.g. for validation)
        n_epochs: fractional number of passes through the data
        """
        batch_losses = defaultdict(list)
        loss_name = loss_fn.__name__
        # train discriminator
        for x in pbar(_take_epochs(X, n_epochs), desc=loss_name, leave=False):
            if optimizer:
                self.zero_grad()
            loss_components = loss_fn(_wrap(x))
            if optimizer:
                loss = sum(loss_components.values())
                loss.backward()
                optimizer.step()
            for k,v in loss_components.items():
                batch_losses[k].append(v.data.cpu().numpy())
        return {k+'_'+loss_name:np.mean(v) for k,v in batch_losses.items()}

    def fit(self,
            X_train, X_valid=None,
            disc_opt_fn=None, ae_opt_fn=None,
            disc_epochs=.5, ae_epochs=1,
            log_fn=None,
            n_epochs=100):
        """
        X_train: torch.utils.data.DataLoader
        X_valid: torch.utils.data.DataLoader or None
        disc_opt_fn: takes parameter set, returns torch.optim.Optimizer
            if None, a default Optimizer will be used
        ae_opt_fn: takes parameter set, returns torch.optim.Optimizer
            if None, a default Optimizer will be used
        disc_epochs: number of discriminator passes through the data each epoch
            (may be fractional)
        ae_epochs: number of autoencoder passes through the data each epoch
            (may be fractional)
        log_fn: takes diagnostic dict, called after every epoch
        n_epochs: number of discriminator, autoencoder training iterations
        """
        _unfreeze(self)

        disc_opt_fn = disc_opt_fn or (lambda p: torch.optim.Adam(p, lr=3e-4))
        ae_opt_fn = ae_opt_fn or (lambda p: torch.optim.Adam(p, lr=3e-4))

        disc_opt = disc_opt_fn(chain(
            self.D.parameters(), self.C.parameters()))
        ae_opt = ae_opt_fn(chain(
            self.encoder.parameters(), self.generator.parameters()))

        for i in pbar(range(n_epochs), desc='epoch'):
            diagnostic = defaultdict(dict)

            # train discriminator
            _freeze(self.encoder, self.generator)
            _unfreeze(self.C, self.D)
            diagnostic['train'].update( self.fit_step(
                X_train, self.discriminator_loss, disc_opt, disc_epochs ))

            # train autoencoder
            _freeze(self.C, self.D)
            _unfreeze(self.encoder, self.generator)
            diagnostic['train'].update( self.fit_step(
                X_train, self.autoencoder_loss, ae_opt, ae_epochs ))

            # validation
            _freeze(self)
            if not X_valid is None:
                diagnostic['valid'].update( self.fit_step(
                    X_valid, self.discriminator_loss ))
                diagnostic['valid'].update( self.fit_step(
                    X_valid, self.autoencoder_loss ))

            # log the dict of losses
            log_fn(diagnostic)

    # def encoder_loss(self, x):
    #     """Loss for step 4 of algorithm 1"""
    #     z, x_rec = self(x)
    #     return self.rec_loss(x_rec, x) - self.C(z).log().mean()
    #
    # def generator_loss(self,x):
    #     """Loss for step 5 of algorithm 1"""
    #     z_prior, x_prior = self(len(x), mode='sample')
    #     z, x_rec = self(x)
    #     return self.rec_loss(x_rec, x) - self.D(
    #          torch.cat((x_prior, x_rec), 0)
    #     ).log().mean()

    # def D_loss(self, x):
    #     """Loss for step 6 of algorithm 1"""
    #     z_prior, x_prior = self(len(x), mode='sample')
    #     z, x_rec = self(x)
    #     return (
    #         - self.D(x).log().mean()
    #         - (1 - self.D( torch.cat((x_prior, x_rec), 0) )).log().mean()
    #     )
    #
    # def C_loss(self, x):
    #     """Loss for step 7 of algorithm 1"""
    #     z_prior = self.sample_prior(len(x))
    #     z = self.encoder(x)
    #     return (
    #         - self.C(z).log()
    #         - (1-self.C(z_prior)).log()
    #     ).mean()

    # def fit(self,
    #         X_train, X_valid=None,
    #         optim_fns=None,
    #         epoch_len=[1,2,1,1],
    #         log_fn=None,
    #         n_epochs=100):
    #     _unfreeze(self)
    #     optim_fns = optim_fns or [lambda p: torch.optim.Adam(p, lr=2e-4)]*4
    #     modules = [self.encoder, self.generator, self.D, self.C]
    #     optimizers = [opt(m.parameters()) for opt, m in zip(optim_fns, modules)]
    #     loss_fns = [
    #         self.encoder_loss, self.generator_loss, self.D_loss, self.C_loss
    #     ]
    #     for i in pbar(range(n_epochs), desc='epoch'):
    #         diagnostic = {}
    #         for mod, n, optimizer, loss_fn in zip(
    #             modules, epoch_len, optimizers, loss_fns
    #             ):
    #             ################ DEBUG
    #             if mod==self.D: continue
    #             if mod==self.C: continue
    #             ################
    #             _freeze(self)
    #             _unfreeze(mod)
    #             batch_losses = []
    #             it_name = loss_fn.__name__
    #             # training
    #             for x in pbar(
    #                 _take_epochs(X_train,n), desc=it_name, leave=False
    #                 ):
    #                 self.zero_grad()
    #                 loss = loss_fn(_wrap(x))
    #                 loss.backward()
    #                 optimizer.step()
    #                 batch_losses.append(loss.data.cpu().numpy())
    #             diagnostic['train_'+it_name] = np.mean(batch_losses)
    #             # validation
    #             if not X_valid is None:
    #                 diagnostic['valid_'+it_name] = np.mean([
    #                     loss_fn(x).data.cpu().numpy() for x in X_valid ])
    #         # log the dict of losses
    #         log_fn(diagnostic)

    def forward(self, *args, mode=None):
        """
        mode:
            None: return z ~ Q(z|x), x_rec ~ P(x|z); args[0] is x.
            sample: return z ~ P(z), x ~ P(x|z); args[0] is number of samples.
            generate: return x ~ P(x|z); args[0] is z.
            encode: return z ~ Q(z|x); args[0] is x.
            reconstruct: like None, but only return x_rec.
        """
        # get code from prior, args, or by encoding.
        if mode=='sample':
            n = args[0]
            z = self.sample_prior(n)
        elif mode=='generate':
            z = _wrap(args[0])
        else:
            x = _wrap(args[0])
            z = self.encoder(x)
        # step there if reconstruction not desired
        if mode=='encode':
            return z
        # run code through generator
        x_rec = self.generator(z)
        if mode=='reconstruct' or mode=='generate':
            return x_rec
        # None, 'sample' return code and reconstruction
        return z, x_rec
