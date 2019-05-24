import torch

def KL(mu, sigma):
    loss = -0.5 * torch.sum(1 + sigma.log() - mu.pow(2) - sigma)
    return loss

def ACC(predictions, targets, masks, lengths):
    predictions[masks == 0] = -1
    correct = ((predictions == targets).sum(dim=0).float() / lengths.float()).mean()
    return correct

def ppl(out, targets, mask):
    """ Compute perplexity """
    # Remove the effect of irrelevant tokens.
    out[mask==0] = 1

    #Collect the probs of the targets.
    out = torch.nn.functional.softmax(out, dim=-1)
    likelihood = out.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze()
    likelihood = likelihood.log().sum()
    return -likelihood

def approx_likelihood(SentVAE, inputs, targets, S=10):

    prior = torch.distributions.normal.Normal(torch.zeros(SentVAE.latent_size), 1)
    log_px = 0

    for k in range(S):
        p_z = SentVAE.encoder(inputs)
        print(p_z.size())
