# alpha-GAN
Unofficial pytorch implementation of Rosca, Mihaela, et al. "Variational Approaches for Auto-Encoding Generative Adversarial Networks." arXiv preprint arXiv:1706.04987 (2017).

**I've got visually reasonable results on CIFAR-10 (see notebook). As the authors state, alpha-GAN is sensitive to changes in the network architectures. It seems important to keep batch normalization out of the code discriminator (C).**

## Deviations From The Paper

In the original paper (v1 on arXiv), prior and posterior terms are be swapped in the code discriminator loss (equations 16 and 17 in Algorithm 1). Authors have confirmed.

Algorithm 1 in the paper is vague as to how each network should be updated; it doesn't explain how SGD enter the picture, or the details of optimization. The authors have confirmed that each of the four networks is updated separately in their experiments. However, in this implementation, encoder and generator (E and G networks) are updated jointly and share an optimizer. It may be worth revisiting the sequence and separation of optimizers.

## Basic Usage

```#
from alphagan import AlphaGAN

E, G, D, C = ... #torch.nn.Module

model = AlphaGAN(E, G, D, C, lambd=10, latent_dim=32)
if use_gpu:
  model = model.cuda()

X_train, X_valid = ... #torch.utils.data.DataSet

train_loader, valid_loader = ... #torch.utils.data.DataLoader

model.fit(train_loader, valid_loader, n_iter=(2,1,1), n_epochs=4, log_fn=print)

# encode and reconstruct
z_valid, x_recon = model(X_valid[0].unsqueeze(0))

# sample from the generative model
z, x_gen = model(batch_size, mode='sample')
```

Supply any torch.nn.Module encoder, generator, discriminator, and code discriminator at construction and any torch.optim.Optimizer and torch.utils.DataLoader to fit().

## Examples

alphagan/examples/CIFAR.ipynb

### Progress Bars

Install tqdm for progress bars. To get working nested progress bars in jupyter notebooks: `pip install -e git+https://github.com/dvm-shlee/tqdm.git@master#egg=tqdm`
