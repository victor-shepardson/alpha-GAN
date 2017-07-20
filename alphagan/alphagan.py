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
        from tqdm import tqdm_notebook as pbar
    else:
        from tqdm import tqdm as pbar
except ImportError:
    def pbar(it, *a, **kw):
        return it

def _freeze(module):
    for p in module.parameters():
        p.requires_grad = False
def _unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True

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
        self.rec_loss = nn.L1Loss()

    def sample_prior(self, n):
        return Variable(torch.randn(n, self.latent_dim), requires_grad=False)

    def encoder_loss(self, x):
        """Loss for step 4 of algorithm 1"""
        _freeze(self)
        _unfreeze(self.encoder)
        z, x_rec = self(x)
        return self.rec_loss(x_rec, x) - self.C(z).log().mean(0)

    def generator_loss(self, x):
        """Loss for step 5 of algorithm 1"""
        _freeze(self)
        _unfreeze(self.generator)
        z_prior, x_prior = self(len(x), mode='sample')
        z, x_rec = self(x)
        return self.rec_loss(x_rec, x) - (
            self.D(x_rec).log()
            + self.D(x_prior).log()
        ).mean(0)

    def D_loss(self, x):
        """Loss for step 6 of algorithm 1"""
        _freeze(self)
        _unfreeze(self.D)
        z_prior, x_prior = self(len(x), mode='sample')
        z, x_rec = self(x)
        return - (
            self.D(x).log()
            + (1 - self.D(x_rec)).log()
            + (1 - self.D(x_prior)).log()
        ).mean(0)

    def C_loss(self, x):
        """Loss for step 7 of algorithm 1"""
        _freeze(self)
        _unfreeze(self.C)
        z_prior = self.sample_prior(len(x))
        z = self.encoder(x)
        return - (
            self.C(z).log() + (1-self.C(z_prior)).log()
        ).mean(0)

    def fit(self,
            X_train, X_valid=None,
            optim_fns=None,
            log_fn=None,
            n_epochs=100):
        _unfreeze(self)
        optim_fns = optim_fns or [torch.optim.Adam]*4
        optimizers = [opt(m.parameters()) for opt, m in zip(optim_fns, (
            self.encoder, self.generator, self.D, self.C
        ))]
        loss_fns = [
            self.encoder_loss, self.generator_loss, self.D_loss, self.C_loss
        ]
        for i in pbar(range(n_epochs), desc='epoch'):
            diagnostic = {}
            for optimizer, loss_fn in zip(optimizers, loss_fns):
                batch_losses = []
                it_name = loss_fn.__name__
                # training
                for x in pbar(X_train, desc=it_name, leave=False):
                    optimizer.zero_grad()
                    loss = loss_fn(_wrap(x))
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.data.cpu().numpy())
                diagnostic['train_'+it_name] = np.mean(batch_losses)
                # validation
                if not X_valid is None:
                    diagnostic['valid_'+it_name] = np.mean([
                        loss_fn(x).data.cpu().numpy() for x in X_valid ])
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
