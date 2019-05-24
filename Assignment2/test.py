import torch
import numpy as np

from metrics import approx_likelihood


latent_size = 2
batch_size = 3
seq_len = 5
vocab_size = 10
S = 1


targets = torch.zeros(seq_len, batch_size).long()

prior = torch.distributions.normal.Normal(torch.zeros(batch_size, latent_size), 1)
mu, sigma = torch.randn((batch_size, latent_size)), torch.randn((batch_size, latent_size))**2
posterior = torch.distributions.normal.Normal(mu, sigma)

likelihood = torch.zeros(S, batch_size)

for k in range(S):
    z = posterior.sample()
    #out = self.decoder(inputs, z, mask.size(0))
    out = torch.randn((seq_len, batch_size, vocab_size))
    out = torch.softmax(out, dim=-1)
    p = out.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze().log().sum(0)
    q = posterior.log_prob(z).sum(-1)
    prior_p = prior.log_prob(z).sum(-1)
    likelihood[k] = (p + prior_p - q)
    print((p + prior_p - q).shape)

likelihood = torch.logsumexp(likelihood, dim=0) - np.log(S)
print(likelihood.size())
