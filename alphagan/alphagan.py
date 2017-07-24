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
        return self._wrap(torch.randn(n, self.latent_dim))#, requires_grad=False)

    def rec_loss(self, x_rec, x):
        """L1 reconstruction error or Laplace log likelihood"""
        return self.lambd * (x_rec-x).abs().mean()

    def autoencoder_loss(self, x):
        """Return reconstruction loss, adversarial loss"""
        z_prior, x_prior = self(len(x), mode='sample')
        z, x_rec = self(x)
        xs = torch.cat((x_prior, x_rec), 0)
        return dict(
            reconstruction_loss = self.rec_loss(x_rec, x),
            adversarial_loss = -(self.D(xs) + _eps).log().mean(),
            code_adversarial_loss = -(self.C(z) + _eps).log().mean()
        )

    def discriminator_loss(self, x):
        """Return discriminator (D) loss, code discriminator (C) loss"""
        z_prior, x_prior = self(len(x), mode='sample')
        z, x_rec = self(x)
        xs = torch.cat((x_prior, x_rec), 0)
        return dict(
            discriminator_loss = - (self.D(x) + _eps).log().mean()
                - (1 - self.D(xs) + _eps).log().mean(),
            code_discriminator_loss = - (self.C(z_prior) + _eps).log().mean()
                - (1 - self.C(z) + _eps).log().mean()
        )

    def _epoch(self, X, disc_opt=None, ae_opt=None,
                 n_batches=None, n_iter=(1,1)):
        """Evaluate/optimize for one epoch.
        X: torch.nn.DataLoader
        disc_opt: torch.nn.Optimizer for discriminators or None if not training
        ae_opt: torch.nn.Optimizer for autoencoder or None if not training
        n_batches: number of batches to draw or None for all data
        n_iter: tuple of steps per batch for discriminators, autoencoder
        """
        iter_losses = defaultdict(list)
        it = _take_batches(X, n_batches) if n_batches else X
        desc = 'training batch' if disc_opt else 'validating batch'
        for x in pbar(it, desc=desc, leave=False):
            x = self._wrap(x)
            for i in range(n_iter[0]):
                self.zero_grad()
                _freeze(self.E, self.G)
                _unfreeze(self.C, self.D)
                loss_components = self.discriminator_loss(x)
                if disc_opt:
                    loss = sum(loss_components.values())
                    loss.backward()
                    disc_opt.step()
                    del loss
                for k,v in loss_components.items():
                    iter_losses[k].append(v.data.cpu().numpy())
            for _ in range(n_iter[1]):
                self.zero_grad()
                _freeze(self.C, self.D)
                _unfreeze(self.E, self.G)
                loss_components = self.autoencoder_loss(x)
                if ae_opt:
                    loss = sum(loss_components.values())
                    loss.backward()
                    ae_opt.step()
                    del loss
                for k,v in loss_components.items():
                    iter_losses[k].append(v.data.cpu().numpy())
        return {k:np.mean(v) for k,v in iter_losses.items()}

    def fit(self,
            X_train, X_valid=None,
            disc_opt_fn=None, ae_opt_fn=None,
            n_iter=1, n_batches=None, n_epochs=100,
            log_fn=None, log_every=1,
            checkpoint_fn=None, checkpoint_every=10):
        """
        X_train: torch.utils.data.DataLoader
        X_valid: torch.utils.data.DataLoader or None
        disc_opt_fn: takes parameter set, returns torch.optim.Optimizer
            if None, a default optimizer will be used
        ae_opt_fn: takes parameter set, returns torch.optim.Optimizer
            if None, a default optimizer will be used
        n_iter: int/tuple # of discriminator, autoencoder optimizer steps/batch
        n_batches: # of batches per epoch or None for all data
        n_epochs: number of discriminator, autoencoder training iterations
        log_fn: takes diagnostic dict, called after every nth epoch
        log_every: call log function every nth epoch
        checkpoint_fn: takes model, epoch. called after nth every epoch
        checkpoint_every: call checkpoint function every nth epoch
        """
        _unfreeze(self)

        try:
            assert len(n_iter)==2
        except Exception:
            n_iter = (n_iter,)*2

        # default optimizers
        disc_opt_fn = disc_opt_fn or (lambda p: torch.optim.Adam(
            p, lr=2e-4, betas=(.5,.9)
        ))
        ae_opt_fn = ae_opt_fn or (lambda p: torch.optim.Adam(
            p, lr=2e-4, betas=(.5,.9)
        ))
        disc_opt = disc_opt_fn(chain(
            self.D.parameters(), self.C.parameters()))
        ae_opt = ae_opt_fn(chain(
            self.E.parameters(), self.G.parameters()))

        for i in pbar(range(n_epochs), desc='epoch'):
            diagnostic = defaultdict(dict)
            report = log_fn and (i%log_every==0 or i==n_epochs-1)
            checkpoint = checkpoint_every and checkpoint_fn and (
                (i+1)%checkpoint_every==0 or i==n_epochs-1 )

            # train for one epoch
            self.train()
            diagnostic['train'].update( self._epoch(
                X_train, disc_opt, ae_opt,
                n_batches=n_batches, n_iter=n_iter ))

            # validate for one epoch
            self.eval()
            diagnostic['valid'].update(self._epoch(X_valid))

            # log the dict of losses
            if report:
                log_fn(diagnostic)

            if checkpoint:
                checkpoint_fn(self, i+1)

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
            z = self._wrap(args[0])
        else:
            x = self._wrap(args[0])
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

    def is_cuda(self):
        return self.E[0].weight.is_cuda

    def _wrap(self, x):
        """ensure x is a Variable on the correct device"""
        if not isinstance(x, Variable):
            # if x isn't a Tensor, attempt to construct one from it
            if not isinstance(x, torch._TensorBase):
                x = torch.Tensor(x)
            x = Variable(x)
        if self.is_cuda():
            x = x.cuda()
        return x
