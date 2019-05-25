import torch
import numpy as np

def KL(mu, sigma):
    loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp(), dim=0)
    return loss

def ACC(predictions, targets, masks, lengths):
    predictions[masks == 0] = -1
    correct = ((predictions == targets).sum(dim=0).float() / lengths.float()).mean()
    return correct

def eval_RNN(out, targets, mask):
    """ Compute perplexity """
    #Collect the probs of the targets.

    logits_flat = out.view(-1, out.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = targets.contiguous().view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*targets.size())
    # mask: (batch, max_len)
    prob = losses * mask.float()

    likelihood = prob.sum(0)
    ppl = prob.sum()

    return -likelihood.mean(), ppl

def eval_VAE(SentVAE, inputs, targets, mask, device, S=10):
    mu, log_sigma = SentVAE.encoder(inputs, mask.sum(0))

    prior = torch.distributions.normal.Normal(torch.zeros(inputs.size(1), SentVAE.latent_size).to(device), 1)
    posterior = torch.distributions.normal.Normal(mu, log_sigma.exp())

    likelihood = torch.zeros(S, mask.size(1)).to(device)
    for k in range(S):
        z = posterior.sample().to(device)
        out = SentVAE.decoder(inputs, z, mask.sum(0))
        
        logits_flat = out.view(-1, out.size(-1))
        # log_probs_flat: (batch * max_len, num_classes)
        log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=1)
        # target_flat: (batch * max_len, 1)
        target_flat = targets.contiguous().view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        losses = losses_flat.view(*targets.size())
        # mask: (batch, max_len)
        p = (losses * mask.float()).sum(0)

        q = posterior.log_prob(z).sum(-1)
        prior_p = prior.log_prob(z).sum(-1)
        likelihood[k] = (p + prior_p - q)

    likelihood = (torch.logsumexp(likelihood, dim=0) - np.log(S))

    return -likelihood.mean(), likelihood.sum()


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
