# alpha-GAN
pytorch implementation of Rosca, Mihaela, et al. "Variational Approaches for Auto-Encoding Generative Adversarial Networks." arXiv preprint arXiv:1706.04987 (2017).

**I've got visually reasonable results on CIFAR-100 (see notebook). Having difficulty getting both good reconstructions and diverse, natural-looking samples at once though.**

## Deviations From The Paper

In the original paper, prior and posterior terms appear to be swapped in the code discriminator loss (equations 16 and 17 in Algorithm 1).

Algorithm 1 in the paper is generally vague as to how each network should be updated. In this implementation:

- Encoder and generator are trained jointly
- Discriminator and code discriminator are trained jointly
- As in other GAN implementations, discriminator is updated first, then generator for each batch.

## Basic Usage

```#
from alphagan import AlphaGAN

E, G, D, C = ... #torch.nn.Module

model = AlphaGAN(E, G, D, C, lambd=10, latent_dim=32)
if use_gpu:
  model = model.cuda()

X_train, X_valid = ... #torch.utils.data.DataSet

train_loader, valid_loader = ... #torch.utils.data.DataLoader

model.fit(train_loader, valid_loader, n_iter=(1,2), n_epochs=40, log_fn=print)

# encode and reconstruct
z_valid, x_recon = model(X_valid[0])

# sample from the generative model
z, x_gen = model(batch_size, mode='sample')
```

Supply any torch.nn.Module decoder, generator, discriminator, and code discriminator at construction and any torch.optim.Optimizer and torch.utils.DataLoader to fit().

## Examples

alphagan/examples/CIFAR.ipynb

### Progress Bars

Install tqdm for progress bars. To get working nested progress bars in jupyter notebooks: `pip install -e git+https://github.com/dvm-shlee/tqdm.git@master#egg=tqdm`
