import torch
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from utils import reverse_logcumsumexp, smart_perm


class PlackettLuce(Distribution):
    """
        Plackett-Luce distribution
    """
    arg_constraints = {"logits": constraints.real}

    def __init__(self, logits):
        # last dimension is for scores of plackett luce
        super(PlackettLuce, self).__init__()
        self.logits = logits
        self.size = self.logits.size()

    def sample(self, num_samples):
        # sample permutations using Gumbel-max trick to avoid cycles
        with torch.no_grad():
            logits = self.logits.unsqueeze(0).expand(num_samples, *self.size)
            u = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
            z = self.logits - torch.log(-torch.log(u))
            samples = torch.sort(z, descending=True, dim=-1)[1]
        return samples

    def log_prob(self, samples):
        # samples shape is: num_samples x self.size
        # samples is permutations not permutation matrices
        if samples.ndimension() == self.logits.ndimension():  # then we already expanded logits
            logits = smart_perm(self.logits, samples)
        elif samples.ndimension() > self.logits.ndimension():  # then we need to expand it here
            logits = self.logits.unsqueeze(0).expand(*samples.size())
            logits = smart_perm(logits, samples)
        else:
            raise ValueError("Something wrong with dimensions")
        logp = (logits - reverse_logcumsumexp(logits, dim=-1)).sum(-1)
        return logp
