import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

def generate_data(n_samples):
    epsilon = np.random.normal(size=(n_samples))
    x_data = np.random.uniform(-10.5, 10.5, n_samples)
    y_data = 7*np.sin(x_data) + 0.5*x_data + epsilon
    return x_data, y_data


n_samples = 1000
x_data, y_data = generate_data(n_samples)


n_input = 1
n_hidden = 20
n_output = 1

model = nn.Sequential(
    nn.Linear(n_input, n_hidden),
    nn.Tanh(),
    nn.Linear(n_hidden, n_output))

loss_fn = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters())

x_tensor = torch.from_numpy(np.float32(x_data).reshape(n_samples, n_input))
y_tensor = torch.from_numpy(np.float32(y_data).reshape(n_samples, n_input))
x_variable = Variable(x_tensor)
y_variable = Variable(y_tensor, requires_grad=False)


def train():
    for epoch in range(3000):
        y_pred = model(x_variable) # make prediction
        loss = loss_fn(y_pred, y_variable)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 300 == 0: print(epoch, loss.item())

'''
train()


x_test_data = np.linspace(-10, 10, n_samples)

# change data shape, move from numpy to torch
x_test_tensor = torch.from_numpy(np.float32(x_test_data).reshape(n_samples, n_input))
x_test_variable = Variable(x_test_tensor)
y_test_variable = model(x_test_variable)

# move from torch back to numpy
y_test_data = y_test_variable.data.numpy()

# plot the original data and the test data
plt.figure(figsize=(8, 8))
plt.scatter(x_data, y_data, alpha=0.2)
plt.scatter(x_test_data, y_test_data, alpha=0.2)
plt.show()


# plot x against y instead of y against x
#plt.figure(figsize=(8, 8))
#plt.scatter(y_data, x_data, alpha=0.2)
#plt.show()

x_variable.data = y_tensor
y_variable.data = x_tensor

train()
'''
x_test_data = np.linspace(-15, 15, n_samples)
x_test_tensor = torch.from_numpy(np.float32(x_test_data).reshape(n_samples, n_input))
x_test_variable = Variable(x_test_tensor)

y_test_variable = model(x_test_variable)

# move from torch back to numpy
y_test_data = y_test_variable.data.numpy()


class MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussians):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)

    def forward(self, x):
        z_h = self.z_h(x)
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        mu = self.z_mu(z_h)
        sigma = torch.exp(self.z_sigma(z_h))
        return pi, mu, sigma

def gaussian_distribution(y, mu, sigma):
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return  torch.exp(result) * torch.reciprocal(sigma)

def mdn_loss_fn(pi, mu, sigma, y):
    result = gaussian_distribution(y, mu, sigma) * pi
    result = torch.sum(result, dim=1)
    result = -torch.log(result)
    return torch.mean(result)



mdn_model = MDN(n_hidden=20, n_gaussians=10)
optimizer = torch.optim.Adam(mdn_model.parameters())

mdn_x_data = y_data
mdn_y_data = x_data
mdn_x_tensor = y_tensor
mdn_y_tensor = x_tensor

x_variable = Variable(mdn_x_tensor)
y_variable = Variable(mdn_y_tensor)

#c = 1.0 / np.sqrt(2.0 * np.pi)

def train_mdn():
    for epoch in range(20000):
        pi_variable, mu_variable, sigma_variable = mdn_model(x_variable)
        loss = mdn_loss_fn(pi_variable, mu_variable, sigma_variable, y_variable)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0: print(epoch, loss.item())

train_mdn()


pi_variable, mu_variable, sigma_variable = mdn_model(x_test_variable)
pi_data = pi_variable.data.numpy()
mu_data = mu_variable.data.numpy()
sigma_data = sigma_variable.data.numpy()

'''
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8,8))
ax1.plot(x_test_data, pi_data)
ax1.set_title('$\Pi$')
ax2.plot(x_test_data, sigma_data)
ax2.set_title('$\sigma$')
ax3.plot(x_test_data, mu_data)
ax3.set_title('$\mu$')
plt.xlim([-15,15])
plt.show()


plt.figure(figsize=(8, 8), facecolor='white')
for mu_k, sigma_k in zip(mu_data.T, sigma_data.T):
    plt.plot(x_test_data, mu_k)
    plt.fill_between(x_test_data, mu_k-sigma_k, mu_k+sigma_k, alpha=0.1)
plt.scatter(mdn_x_data, mdn_y_data, marker='.', lw=0, alpha=0.2, c='black')
plt.xlim([-10,10])
plt.ylim([-10,10])
plt.show()
'''

def gumbel_sample(x, axis=1):
    z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return (np.log(x) + z).argmax(axis=axis)


k = gumbel_sample(pi_data)
indices = (np.arange(n_samples), k)
rn = np.random.randn(n_samples)
sampled = rn * sigma_data[indices] + mu_data[indices]


plt.figure(figsize=(8, 8))
plt.scatter(mdn_x_data, mdn_y_data, alpha=0.2)
plt.scatter(x_test_data, sampled, alpha=0.2, color='red')
plt.show()
