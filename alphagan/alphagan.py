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
    def __init__(self, E, G, D, C, latent_dim,
                 lambd=1, z_lambd=0, code_weight=1, adversarial_weight=1):
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
        z_lambd: if nonzero, weight for code reconstruction loss
        code_weight: weight for code loss. if zero, C won't be trained
        adversarial_weight: weight for adversarial loss. if zero, D won't be trained
        """
        super().__init__()
        self.E = E
        self.G = G
        self.D = D
        self.C = C
        self.latent_dim = latent_dim
        self.lambd = lambd
        self.z_lambd = z_lambd
        self.code_weight = code_weight
        self.adversarial_weight = adversarial_weight

    def sample_prior(self, n):
        """Sample self.latent_dim-dimensional unit normal.
        n: batch size
        """
        return self._wrap(torch.randn(n, self.latent_dim))#, requires_grad=False)

    def rec_loss(self, x_rec, x):
        """L1 reconstruction error or Laplace log likelihood"""
        return (x_rec-x).abs().mean()
    
    def autoencoder_loss(self, x):
        """Return reconstruction loss, adversarial loss"""
        _freeze(self) # save memory by excluding parameters from autograd
        _unfreeze(self.E, self.G)
        z_prior, x_prior = self(len(x), mode='sample')
        z, x_rec = self(x)
        xs = torch.cat((x_prior, x_rec), 0)
#         return dict(
#             reconstruction_loss = self.rec_loss(x_rec, x),
#             adversarial_loss = -(self.D(xs) + _eps).log().mean(),
#             code_adversarial_loss = -(self.C(z) + _eps).log().mean()
#         )
        ret = {}
        if self.code_weight != 0:
            ret['code_adversarial_loss'] = -(self.C(z) + _eps).log().mean()
        if self.adversarial_weight != 0:
            ret['adversarial_loss'] = -(self.D(xs) + _eps).log().mean()
        if self.lambd != 0:
            ret['reconstruction_loss'] = self.lambd*self.rec_loss(x_rec, x)
        if self.z_lambd != 0:
            zs = torch.cat((z_prior, z), 0)
            z_rec = self(xs, mode='encode')
            ret['code_reconstruction_loss'] = self.z_lambd*self.rec_loss(z_rec, zs)
        return ret

    def discriminator_loss(self, x):
        """Return discriminator (D) loss"""
        _freeze(self) # save memory by excluding parameters from autograd
        _unfreeze(self.D)
        z_prior, x_prior = self(len(x), mode='sample')
        z, x_rec = self(x)
        xs = torch.cat((x_prior, x_rec), 0)
        return {
            'discriminator_loss':
                - (self.D(x) + _eps).log().mean()
                - (1 - self.D(xs) + _eps).log().mean()
        }

    def code_discriminator_loss(self, x):
        """Return code discriminator (C) loss"""
        _freeze(self) # save memory by excluding parameters from autograd
        _unfreeze(self.C)
        z_prior = self.sample_prior(len(x))
        z = self(x, mode='encode')
        return {
            'code_discriminator_loss':
                - (self.C(z_prior) + _eps).log().mean()
                - (1 - self.C(z) + _eps).log().mean()
        }

    def _epoch(self, X, loss_fns,
               optimizers=None, n_iter=(1,1,1), n_batches=None):
        """Evaluate/optimize for one epoch.
        X: torch.nn.DataLoader
        loss_fns: each takes an input batch and returns dict of loss component Variables
        optimizers: sequence of torch.nn.Optimizer for each loss or None if not training
        n_iter: sequence of optimization steps per batch for each loss
        n_batches: number of batches to draw or None for all data
        """
        optimizers = optimizers or [None]*3
        iter_losses = defaultdict(list)
        
        it = _take_batches(X, n_batches) if n_batches else X
        desc = 'training batch' if optimizers else 'validating batch'
        for x in pbar(it, desc=desc, leave=False):
            x = self._wrap(x)
            for opt, iters, loss_fn in zip(optimizers, n_iter, loss_fns):
                for _ in range(iters):
                    loss_components = loss_fn(x)
                    if opt:
                        loss = sum(loss_components.values())
                        self.zero_grad()
                        loss.backward()
                        opt.step()
                        del loss
                    for k,v in loss_components.items():
                        iter_losses[k].append(v.data.cpu().numpy())
                    del loss_components
        return {k:np.mean(v) for k,v in iter_losses.items()}

    def fit(self,
            X_train, X_valid=None,
            EG_opt_fn=None, D_opt_fn=None, C_opt_fn=None,
            n_iter=1, n_batches=None, n_epochs=100,
            log_fn=None, log_every=1,
            checkpoint_fn=None, checkpoint_every=10):
        """
        X_train: torch.utils.data.DataLoader
        X_valid: torch.utils.data.DataLoader or None
        EG_opt_fn, D_opt_fn, C_opt_fn: takes parameter set, returns torch.optim.Optimizer
            if None, a default optimizer will be used
        n_iter: int/tuple # of E/G, D, C optimizer steps/batch
        n_batches: # of batches per epoch or None for all data
        n_epochs: number of discriminator, autoencoder training iterations
        log_fn: takes diagnostic dict, called after every nth epoch
        log_every: call log function every nth epoch
        checkpoint_fn: takes model, epoch. called after nth every epoch
        checkpoint_every: call checkpoint function every nth epoch
        """
        _unfreeze(self)

        try:
            assert len(n_iter)==3
        except TypeError:
            n_iter = (n_iter,)*3

        # default optimizers
        EG_opt_fn = EG_opt_fn or (lambda p: torch.optim.Adam(
            p, lr=8e-4, betas=(.5,.9)
        ))
        D_opt_fn = D_opt_fn or (lambda p: torch.optim.Adam(
            p, lr=8e-4, betas=(.5,.9)
        ))
        C_opt_fn = C_opt_fn or (lambda p: torch.optim.Adam(
            p, lr=8e-4, betas=(.5,.9)
        ))
        
        # define optimization order/separation of networks
        optimizers, loss_fns = [], []
        # encoder/generator together
        optimizers.append(EG_opt_fn(chain(
            self.E.parameters(), self.G.parameters()
        )))
        loss_fns.append(self.autoencoder_loss)
        # discriminator
        if self.adversarial_weight != 0:
            optimizers.append(D_opt_fn(self.D.parameters()))
            loss_fns.append(self.discriminator_loss)
        # code discriminator
        if self.code_weight != 0:
            optimizers.append(C_opt_fn(self.C.parameters()))
            loss_fns.append(self.code_discriminator_loss)
# #         discriminators together
#         optimizers.append(D_opt_fn(chain(
#             self.D.parameters(), self.C.parameters()
#         )))
#         loss_fns.append(lambda x: self.code_discriminator_loss(x)+self.discriminator_loss(x))

        for i in pbar(range(n_epochs), desc='epoch'):
            diagnostic = defaultdict(dict)
            report = log_fn and (i%log_every==0 or i==n_epochs-1)
            checkpoint = checkpoint_every and checkpoint_fn and (
                (i+1)%checkpoint_every==0 or i==n_epochs-1 )
            # train for one epoch
            self.train()
            diagnostic['train'].update( self._epoch(
                X_train, loss_fns, optimizers, n_iter, n_batches ))
            # validate for one epoch
            self.eval()
            diagnostic['valid'].update(self._epoch(X_valid, loss_fns))
            # log the dict of loss components
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
        # default, 'sample': return code and reconstruction
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

# # experiment with using WGAN-GP for the GAN losses
# # requires pytorch > 0.12.2 (current master)
# class AlphaWGAN(AlphaGAN):
#     """Alternative WGAN-GP based losses"""
#     def gradient_penalty(self, model, x, x_prior, w=10):
#         """WGAN-GP gradient penalty"""
#         alpha = self._wrap(torch.rand(len(x)))
#         x_hat = x*alpha + x_prior*(1-alpha)
#         scores = model(x_hat).sum()
# #         ones = self._wrap(torch.ones(x.size()))
#         grad = torch.autograd.grad(
#             scores, x_hat,
# #             grad_outputs=ones,
#             create_graph=True, only_inputs=True)
#         grad_norm_dist = grad.norm(2, dim=1) - 1
#         return w*grad_norm_dist*grad_norm_dist
    
#     def autoencoder_loss(self, x):
#         """Return reconstruction loss, adversarial loss"""
#         _freeze(self) # save memory by excluding parameters from autograd
#         _unfreeze(self.E, self.G)
#         z_prior, x_prior = self(len(x), mode='sample')
#         z, x_rec = self(x)
#         xs = torch.cat((x_prior, x_rec), 0)
#         return dict(
#             reconstruction_loss = self.rec_loss(x_rec, x),
#             adversarial_loss = -self.D(xs).mean(),
#             code_adversarial_loss = -self.C(z).mean()
#         )

#     def discriminator_loss(self, x):
#         """Return discriminator (D) loss"""
#         _freeze(self) # save memory by excluding parameters from autograd
#         _unfreeze(self.D)
#         z_prior, x_prior = self(len(x), mode='sample')
#         z, x_rec = self(x)
#         xs = torch.cat((x_prior, x_rec), 0)
#         return dict(
#             discriminator_loss = 
#                 + self.D(xs).mean()
#                 - self.D(x).mean() 
#                 + self.gradient_penalty(self.D, x, x_prior)
#         )

#     def code_discriminator_loss(self, x):
#         """Return code discriminator (C) loss"""
#         _freeze(self) # save memory by excluding parameters from autograd
#         _unfreeze(self.C)
#         z_prior = self.sample_prior(len(x))
#         z = self(x, mode='encode')
#         return dict(
#             code_discriminator_loss = 
#                 + self.C(z).mean() 
#                 - self.C(z_prior).mean()
#                 + self.gradient_penalty(self.C, z, z_prior)
#         )