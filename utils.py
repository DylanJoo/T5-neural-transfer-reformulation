import random
import torch
import math
import numpy as np

def kl_weight(anneal_fn, step, k, x0):
    if anneal_fn == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

def kl_loss(logv, mean):
    """
    Args
        logv (`torch.tensor`): mapped batch embeddings with (B L H)
        mean (`torch.tensor`): mapped batch embeddings with (B L H)
    """
    return -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

def random_masking(tokens_lists, masked_token):

    for i, tokens_list in enumerate(tokens_lists):
        tokens = tokens_list.split()
        n_tokens = len(tokens)
        masked = random.sample(range(n_tokens), math.floor(n_tokens * 0.15))

        for j in masked:
            tokens[j] = masked_token

        tokens_list[i] = " ".join(tokens)

    return tokens_lists


