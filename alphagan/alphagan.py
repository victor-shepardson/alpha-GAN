from itertools import chain, repeat, islice
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
    X: torch.utils.DataLoader (has len(), iterates over batches)
    n_epochs: number of iterations through the data.
    """
    n_batches = int(np.ceil(len(X)*n_epochs))
    _take_iters(X, n_batches)

def _take_batches(X, n_batches):
    """Get a integer number of batches from X, reshuffling as necessary
    X: torch.utils.DataLoader (has len(), iterates over batches)
    n_iters: number of batches
    """
    n_shuffles = int(np.ceil(len(X)/n_batches))
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
    def __init__(self, E, G, D, C, latent_dim, lambd=1):
        """α-GAN as described in Rosca, Mihaela, et al.
            "Variational Approaches for Auto-Encoding Generative Adversarial Networks."
            arXiv preprint arXiv:1706.04987 (2017).
        E: nn.Module mapping X to Z
        G: nn.Module mapping Z to X
        D: nn.module discriminating real from generated/reconstructed X
        C: nn.module discriminating prior from posterior Z
        latent_dim: dimensionality of Z
        lambd: scale parameter for the G distribution
            a.k.a weight for the reconstruction loss
        """
        super().__init__()
        self.E = E
        self.G = G
        self.D = D
        self.C = C
        self.latent_dim = latent_dim
        self.lambd = lambd

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

    def fit_step(self, X, loss_fn, optimizer=None, n_iters=None):
        """Optimize for one epoch.
        X: torch.utils.DataLoader
        loss_fn: return dict of loss component Variables
        optimizer: if falsy, just compute loss components (e.g. for validation)
        n_iters: number of batches. If falsy, use all data.
        """
        batch_losses = defaultdict(list)
        loss_name = loss_fn.__name__
        it = _take_batches(X, n_iters) if n_iters else X
        # train discriminator
        for x in pbar(it, desc=loss_name, leave=False):
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
            disc_iters=8, ae_iters=16,
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

        # default optimizers
        disc_opt_fn = disc_opt_fn or (lambda p: torch.optim.Adam(p, lr=3e-4))
        ae_opt_fn = ae_opt_fn or (lambda p: torch.optim.Adam(p, lr=3e-4))

        disc_opt = disc_opt_fn(chain(
            self.D.parameters(), self.C.parameters()))
        ae_opt = ae_opt_fn(chain(
            self.E.parameters(), self.G.parameters()))

        for i in pbar(range(n_epochs), desc='epoch'):
            diagnostic = defaultdict(dict)

            # train discriminators
            self.train() # e.g. for BatchNorm
            _freeze(self.E, self.G)
            _unfreeze(self.C, self.D)
            diagnostic['train'].update( self.fit_step(
                X_train, self.discriminator_loss, disc_opt, disc_iters ))

            # validate discriminators
            if not X_valid is None:
                self.eval()
                _freeze(self)
                diagnostic['valid'].update( self.fit_step(
                    X_valid, self.discriminator_loss ))

            # train autoencoder
            self.train()
            _freeze(self.C, self.D)
            _unfreeze(self.E, self.G)
            diagnostic['train'].update( self.fit_step(
                X_train, self.autoencoder_loss, ae_opt, ae_iters ))

            # validate autoencoder
            if not X_valid is None:
                self.eval()
                _freeze(self)
                diagnostic['valid'].update( self.fit_step(
                    X_valid, self.autoencoder_loss ))

            # log the dict of losses
            log_fn(diagnostic)

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
            z = self.E(x)
        # step there if reconstruction not desired
        if mode=='encode':
            return z
        # run code through G
        x_rec = self.G(z)
        if mode=='reconstruct' or mode=='generate':
            return x_rec
        # None, 'sample' return code and reconstruction
        return z, x_rec
