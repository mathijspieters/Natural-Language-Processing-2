import torch
import numpy as np

def KL(mu, sigma):
    loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp(), dim=0)
    return loss

def ACC(predictions, targets, masks, lengths):
    predictions[masks == 0] = -1
    correct = ((predictions == targets).sum(dim=0).float() / lengths.float()).mean()
    return correct

def ppl(out, targets, mask):
    """ Compute perplexity """
    #Collect the probs of the targets.
    out = torch.nn.functional.softmax(out, dim=-1)
    likelihood = out.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze()
    likelihood = likelihood.log().sum()
    return -likelihood

def compute_loss(logits, target, mask):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    losses = losses * mask.float()
    loss = losses.sum() / mask.float().sum()
    loss = loss / logits.shape[0]
    return loss

def approx_likelihood(SentVAE, inputs, targets, S=10):

    prior = torch.distributions.normal.Normal(torch.zeros(SentVAE.latent_size), 1)
    log_px = 0

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
