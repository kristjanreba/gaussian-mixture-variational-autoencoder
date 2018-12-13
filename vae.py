import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from load_mnist import load_mnist


def plot_results(x_train, results):
    x_train = x_train.detach() # remove gradient tracking
    results = results.detach()

    f, a = plt.subplots(2, 10, figsize=(20,4))
    for i in range(10):
        a[0][i].imshow(x_train[i].view(28,28))
        a[1][i].imshow(results[i].view(28,28))
    plt.show()


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        D_in = 28*28
        H_1 = 128
        H_2 = 64
        H_z = 10

        self.encoder = nn.Sequential(
            nn.Linear(D_in, H_1), nn.ReLU(True),
            nn.Linear(H_1, H_2), nn.ReLU(True))

        self.enc_mu = torch.nn.Linear(H_2, H_z)
        self.enc_log_sigma = torch.nn.Linear(H_2, H_z)

        self.decoder = nn.Sequential(
            nn.Linear(H_z, H_2), nn.ReLU(True),
            nn.Linear(H_2, H_1), nn.ReLU(True),
            nn.Linear(H_1, D_in), nn.Tanh())

    def sample_latent(self, x_enc):
        '''
        Sample z from latent space (reparameterization trick)
        return z ~ N(mu, sigma^2)
        '''

        mu = self.enc_mu(x_enc)
        log_sigma = self.enc_log_sigma(x_enc)

        sigma = torch.exp(log_sigma * 0.5)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        return z

    def forward(self, x):
        x_enc = self.encoder(x)
        z = self.sample_latent(x_enc)
        return self.decoder(z)


def get_model():
    model = VAE()
    return model, optim.Adam(model.parameters(), lr=lr)


def loss_batch(model, loss_fn, xb, yb, opt=None):
    '''
    Computes loss for one batch. If we pass the optimizer,
    function performs backpropagation.
    '''
    loss = loss_fn(model(xb), xb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_fn, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_fn, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_fn, xb, yb) for xb, yb in valid_dl])

        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)


def get_data(x_train, y_train, x_valid, y_valid, bs, shuffle=False):
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

    return train_dl, valid_dl


x_train, y_train, x_valid, y_valid = load_mnist()
x_train, y_train, x_valid, y_valid = map(torch.FloatTensor, (x_train, y_train, x_valid, y_valid))

#plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
#plt.show()
#print(x_train.shape)

# Hyper-parameters
D_in = 784
epochs = 10
bs = 64
lr = 1e-3

loss_fn = F.mse_loss
train_dl, valid_dl = get_data(x_train, y_train, x_valid, y_valid, bs, shuffle=False)
model, opt = get_model()
fit(epochs, model, loss_fn, opt, train_dl, valid_dl)

results = model(x_valid)
plot_results(x_valid, results)
