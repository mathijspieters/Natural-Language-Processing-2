import torch

def ppl(out, targets, mask):
    """ Compute perplexity """
    # Remove the effect of irrelevant tokens.
    out[mask==0] = 1

    #Collect the probs of the targets.
    out = torch.nn.functional.softmax(out, dim=-1)
    likelihood = out.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze()
    likelihood = likelihood.log().sum(dim=0)
    return (-likelihood.sum()/mask.sum()).exp()
