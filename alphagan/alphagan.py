import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable

class AlphaGAN(nn.Module):
    def __init__(self, encoder, generator, C, D, latent_dim, lambd=1):
        """α-GAN as described in Rosca, Mihaela, et al.
            "Variational Approaches for Auto-Encoding Generative Adversarial Networks."
            arXiv preprint arXiv:1706.04987 (2017).
        encoder: nn.Module mapping X to Z
        generator: nn.Module mapping Z to X
        C: nn.module discriminating prior from posterior Z
        D: nn.module discriminating real from synthetic X
        latent_dim: dimensionality of Z
        lambd: scale parameter for the generator distribution
            a.k.a weight for the reconstruction loss
        """
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.C, self.D = C, D
        self.latent_dim = latent_dim
        self.lambd = lambd
        self.rec_loss = nn.L1Loss()

    def sample_prior(self, n):
        return Variable(torch.randn(n, self.latent_dim), requires_grad=False)

    def encoder_loss(self, x):
        """Loss for step 4 of algorithm 1"""
        z = self.encoder(x)
        x_rec = self.generator(z)
        return self.rec_loss(x_rec, x) - self.C(z).log().mean(0)

    def generator_loss(self, x):
        """Loss for step 5 of algorithm 1"""
        z_prior = self.sample_prior(len(x))
        z = self.encoder(x)
        x_rec = self.generator(z)
        x_prior = self.generator(z_prior)
        return self.rec_loss(x_rec, x) - (
            self.D(x_rec).log()
            + self.D(x_prior).log()
        ).mean(0)

    def discriminator_loss(self, x):
        """Loss for step 6 of algorithm 1"""
        z_prior = self.sample_prior(len(x))
        z = self.encoder(x)
        x_rec = self.generator(z)
        x_prior = self.generator(z_prior)
        return - (
            self.D(x).log()
            + (1 - self.D(x_rec)).log()
            + (1 - self.D(x_prior)).log()
        ).mean(0)

    def code_discriminator_loss(self, x):
        """Loss for step 7 of algorithm 1"""
        z_prior = self.sample_prior(len(x))
        z = self.encoder(x)
        return - (
            self.C(z).log() + (1-self.C(z_prior)).log()
        ).mean(0)

    def fit(self, X_train, X_valid=None, optimizer=None, log_fn=None):
        optimizer = optimizer or torch.optim.Adam(lr=1e-3)
        for loss_fn in (self.encoder_loss,
                        self.generator_loss,
                        self.discriminator_loss,
                        self.code_discriminator_loss):
            diagnostic = {}
            batch_losses = []
            # training
            for x in X_train:
                optimizer.zero_grad()
                loss = loss_fn(x)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.cpu().numpy())
            diagnostic['train_'+loss_fn.__name__] = np.mean(batch_losses)
            # validation
            diagnostic['valid_'+loss_fn.__name__] = np.mean([
                loss_fn(x).cpu().numpy() for x in X_valid ])
            log_fn(diagnostic)

    # def forward(self, *args, mode='loss'):
        # pass
