def ppl(out, targets, mask):
    """ Compute perplexity """
    # Remove the effect of irrelevant tokens.
    out[mask==0] = 1

    #Collect the probs of the targets.
    likelihood = out.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze()
    likelihood = likelihood.prod(dim=0)
    return (-likelihood.log().sum()/mask.sum()).exp()
