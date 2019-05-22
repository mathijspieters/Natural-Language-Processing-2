import torch
import numpy as np

from metrics import approx_likelihood


latent_size = 2
batch_size = 3
S = 1

prior = torch.distributions.normal.Normal(torch.zeros(latent_size), 1)

for k in range(S):
    mu = torch.randn((batch_size, latent_size))
    sigma = torch.randn((batch_size, latent_size)).pow(2)

    q = torch.distributions.normal.Normal(mu, sigma)
    z = q.sample()
    q_prob = q.log_prob(z).sum(-1)
    prior_prob = prior.log_prob(z).sum(-1)

    p_prob = torch.randn(3,)

    p_x = p_prob + prior_prob - q_prob

    p_x = torch.logsumexp(p_x) - log(N)
