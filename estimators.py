# based on Sean Robertson's pydrobert-pytorch.estimators


import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import PlackettLuce
from utils import logcumsumexp, reverse_logcumsumexp, smart_perm, make_permutation_matrix
from itertools import permutations


def to_z(logits, u=None):
    '''Samples random noise, then injects it into logits to produce z

    Parameters
    ----------
    logits : torch.Tensor
    u : torch.Tensor (you can specify noise yourself)

    Returns
    -------
    z : torch.Tensor
    '''
    if u is not None:
        assert u.size() == logits.size()
    else:
        u = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
    log_probs = F.log_softmax(logits, dim=-1)
    z = log_probs - torch.log(-torch.log(u))
    z.requires_grad_(True)
    return z


def _to_z_tilde(logits, b, v=None):
    '''Produce posterior z

    Parameters
    ----------
    logits : torch.Tensor
    b : torch.Tensor
    v : torch.Tensor (you can specify noise yourself)

    Returns
    -------
    z : torch.Tensor
    '''
    if v is not None:
        assert v.size() == logits.size()
    else:
        v = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
    b_inv = torch.sort(b, dim=-1)[1]
    log_probs = smart_perm(F.log_softmax(logits, dim=-1), b)
    gumbel = torch.log(-torch.log(v))
    z_tilde = -logcumsumexp(gumbel - reverse_logcumsumexp(log_probs, dim=-1), dim=-1)
    z_tilde = smart_perm(z_tilde, b_inv)
    return z_tilde


def _reattach_z_to_new_logits(logits, z):
    log_probs = F.log_softmax(logits, dim=-1)
    z = z.detach() + log_probs - log_probs.detach()
    return z


def to_b(z):
    '''Converts z to sample using a deterministic mapping

    Parameters
    ----------
    z : torch.Tensor

    Returns
    -------
    b : torch.Tensor
    '''
    b = torch.sort(z, descending=True, dim=-1)[1]
    return b


def reinforce(fb, b, logits, **kwargs):
    r'''Perform REINFORCE gradient estimation

    REINFORCE [williams1992]_, or the score function, has a single-sample
    implementation as

    .. math:: g = f(b) \partial \log Pr(b; logits) / \partial logits

    It is an unbiased estimate of the derivative of the expectation w.r.t
    `logits`.

    Though simple, it is often cited as high variance.

    Parameters
    ----------
    fb : torch.Tensor
    b : torch.Tensor
    logits : torch.Tensor

    Returns
    -------
    g : torch.tensor
        A tensor with the same shape as `logits` representing the estimate
        of ``d fb / d logits``

    Notes
    -----
    It is common (such as in A2C) to include a baseline to minimize the
    variance of the estimate. It's incorporated as `c` in

    .. math:: g = (f(b) - c) \partial \log Pr(b; logits) / \partial logits

    Note that :math:`c_i` should be conditionally independent of :math:`b_i`
    for `g` to be unbiased. You can, however, condition on any preceding
    outputs :math:`b_{i - j}, j > 0` and all of `logits`.

    To get this functionality, simply subtract `c` from `fb` before passing it
    to this method. If `c` is the output of a neural network, a common (but
    sub-optimal) loss function is the mean-squared error between `fb` and `c`.
    '''
    fb = fb.detach()
    b = b.detach()
    log_pb = PlackettLuce(logits=logits).log_prob(b)
    fb = fb.unsqueeze(-1)
    g = fb * torch.autograd.grad([log_pb], [logits], grad_outputs=torch.ones_like(log_pb))[0]
    assert g.size() == logits.size()
    return g


def relax(fb, b, logits, z, c, v=None, **kwargs):
    r'''Perform RELAX gradient estimation

    RELAX [grathwohl2017]_ has a single-sample implementation as

    .. math::

        g = (f(b) - c(\widetilde{z}))
                \partial \log Pr(b; logits) / \partial logits
            + \partial c(z) / \partial logits
            - \partial c(\widetilde{z}) / \partial logits

    where :math:`b = H(z)`, :math:`\widetilde{z} \sim Pr(z|b, logits)`, and `c`
    can be any differentiable function. It is an unbiased estimate of the
    derivative of the expectation w.r.t `logits`.

    `g` is itself differentiable with respect to the parameters of the control
    variate `c`. If the c is trainable, an easy choice for its loss is to
    minimize the variance of `g` via ``(g ** 2).sum().backward()``. Propagating
    directly from `g` should be suitable for most situations. Insofar as the
    loss cannot be directly computed from `g`, setting the argument for
    `components` to true will return a tuple containing the terms of `g`
    instead.

    Parameters
    ----------
    fb : torch.Tensor
    b : torch.Tensor
    logits : torch.Tensor
    z : torch.Tensor
    c : callable
        A module or function that accepts input of the shape of `z` and outputs
        a tensor of the same shape if modelling a Bernoulli, or of shape
        ``z[..., 0]`` (minus the last dimension) if Categorical.

    Returns
    -------
    g : torch.Tensor or tuple
        `g` will be the gradient estimate with respect to `logits`.
    '''

    fb = fb.detach()
    b = b.detach()
    # in all computations (including those made by the user later), we don't
    # want to backpropagate past "logits" into the model. We make a detached
    # copy of logits and rebuild the graph from the detached copy to z
    logits = logits.detach().requires_grad_(True)
    z = _reattach_z_to_new_logits(logits, z)
    z_tilde = _to_z_tilde(logits, b, v)
    c_z = c(z, **kwargs)
    c_z_tilde = c(z_tilde, **kwargs)
    fb = fb.unsqueeze(-1)
    diff = fb - c_z_tilde
    log_pb = PlackettLuce(logits=logits).log_prob(b)
    dlog_pb, = torch.autograd.grad(
        [log_pb],
        [logits],
        grad_outputs=torch.ones_like(log_pb)
    )
    # we need `create_graph` to be true here or backpropagation through the
    # control variate will not include the graphs of the derivative terms
    dc_z, = torch.autograd.grad(
        [c_z],
        [logits],
        create_graph=True,
        retain_graph=True,
        grad_outputs=torch.ones_like(c_z)
    )
    dc_z_tilde, = torch.autograd.grad(
        [c_z_tilde],
        [logits],
        create_graph=True,
        retain_graph=True,
        grad_outputs=torch.ones_like(c_z_tilde)
    )
    g = diff * dlog_pb + dc_z - dc_z_tilde
    assert g.size() == logits.size()
    return g


class ExactGradientEstimator(nn.Module):
    """
        just computing exact gradient using log-derivative trick
        and taking expectation over all possible permutations
    """

    def __init__(self, f, d):
        super(ExactGradientEstimator, self).__init__()
        self.f = f
        self.d = d
        self.pi = torch.tensor(list(permutations(range(self.d))))
        self.f_all = self.f(make_permutation_matrix(self.pi))

    def forward(self, log_theta, **kwargs):
        # return gradients and expectation
        logprob = PlackettLuce(log_theta).log_prob(self.pi)
        loss = (torch.exp(logprob).detach() * self.f_all * logprob).sum()
        d_log_theta = torch.autograd.grad([loss], [log_theta])[0]
        return d_log_theta, (torch.exp(logprob).detach() * self.f_all).sum()
