def ppl(out, targets, lengths):
    """ Compute perplexity """
    # Todo how about padding tokens?
    likelihood = out.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze()
    likelihood = likelihood.prod(dim=0)

    return (-likelihood.log().sum()/lengths.sum()).exp()
