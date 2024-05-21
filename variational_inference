import numpy as np 
import torch
from torch.distributions import Normal

def gaussian_likelihood(x):
    """ function to compute the likelihood
    of observing a specific data point x
    given the model's parameters and any latent variables z """
    x_std = math.sqrt(torch.var(x))
    x_mean = torch.mean(x)
    coefficient = 1/torch.sqrt(2 * torch.pi * x_std**2)
    exponential = torch.exp ** -1*(x - x_mean)**2 / 2*x_std**2
    return coefficient * exponential

def log_guassian_prior(z):
    """ Define the prior distribution.
    Then, take the log probability of the prior occurring """
    gaussian_prior = Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z)
    return gaussian_prior

def variational_gaussian(theta):
    """ returns the gaussian function """
    mu = theta[:0]
    logvar = theta[:,1]
    return torch.distributions.Normal(mu, torch.exp(logvar))



class VartionalInference:

    def __init__(self, model_likelihood, prior, variational_family):
        self.model_likelihood = model_likelihood
        self.prior = prior
        self.variational_family = variational_family
        self.theta = 

    def elbo(self, x):
        """ Sample latent variables from the variational distribution.
        Then compute log-likelihood and KL divergence """
        z, q_dist = self.variational(self.theta)

        log_likelihood = self.gaussian_likelihood(x, z)
        kl_divergence = q_dist.log_prob(z) - self.prior(z)
        return torch.mean(log_likelihood - kl_divergence)

    def variational(self, theta):
        """ Return the distribution of the latent variables
        where theta is a vector that represents the parameters of the chosen variational distribution"""
        mu = theta[:0]
        logvar = thera[:,1]
        z = self.reparameterize(mu, logvar)
        return z, q_dist = torch.distributions.Normal(mu, torch.exp(logvar))
    
    def reparameterize(mu, logvar):
        """ Reparameterize the lower bound to yield a lower bound estimator 
        that can be obtimized using stochastic gradient methods"""
        std = torch.exp(0.5 * logvar)
        epsilon = np.random.randn_like(std)
        return mu + epsilon * std

    def train(self, data, theta, optimizer, num_epochs):
        optimizer = torch.optim.Adam([self.theta], lr=0.01)

        for epoch in range(num_epochs):
            for x in data:
                loss = -1*elbo(data, theta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

