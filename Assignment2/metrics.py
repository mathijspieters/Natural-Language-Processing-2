import torch
import numpy as np

def eval_RNN(out, targets, mask):
    """ Compute perplexity """
    #Collect the probs of the targets.
    out = torch.nn.functional.softmax(out, dim=-1)
    likelihood = out.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze()
    likelihood = (likelihood.log()*mask).sum(dim=0)
    return (-likelihood.sum()/mask.sum()).exp()

def eval_VAE(SentVAE, inputs, targets, S=10):
    prior = torch.distributions.normal.Normal(torch.zeros(inputs.size(0), SentVAE.latent_size), 1)
    mu, sigma = SentVAE.encoder(inputs)
    posterior = torch.distributions.normal.Normal(mu, sigma)
    likelihood = torch.zeros(S, inputs.size(0))

    for k in range(S):
        print("sample")
        z = posterior.sample()
        out = SentVAE.decoder(inputs, z, mask.size(0))
        p = out.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze().log().sum(0)
        q = posterior.log_prob(z).sum(-1)
        prior_p = prior.log_prob(z).sum(-1)
        likelihood[k] = (p + prior_p - q)

    likelihood = (torch.logsumexp(likelihood) - np.log(S)).mean()


    return -likelihood, (-likelihood/inputs.size().sum()).exp()
