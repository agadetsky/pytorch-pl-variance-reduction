import torch


def logcumsumexp(x, dim):
    # slow implementation, but ok for now
    if (dim != -1) or (dim != x.ndimension() - 1):
        x = x.transpose(dim, -1)

    out = []
    for i in range(1, x.size(-1) + 1):
        out.append(torch.logsumexp(x[..., :i], dim=-1, keepdim=True))
    out = torch.cat(out, dim=-1)

    if (dim != -1) or (dim != x.ndimension() - 1):
        out = out.transpose(-1, dim)
    return out


def reverse_logcumsumexp(x, dim):
    return torch.flip(logcumsumexp(torch.flip(x, dims=(dim, )), dim), dims=(dim, ))


def make_permutation_matrix(b):
    # permutation matrix P_b with column representation: p_{ij} = 1 if j = b(i)
    return torch.eye(b.size(-1))[b]


def smart_perm(x, permutation):
    assert x.size() == permutation.size()
    if x.ndimension() == 1:
        ret = x[permutation]
    elif x.ndimension() == 2:
        d1, d2 = x.size()
        ret = x[
            torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
            permutation.flatten()
        ].view(d1, d2)
    elif x.ndimension() == 3:
        d1, d2, d3 = x.size()
        ret = x[
            torch.arange(d1).unsqueeze(1).repeat((1, d2 * d3)).flatten(),
            torch.arange(d2).unsqueeze(1).repeat((1, d3)).flatten().unsqueeze(0).repeat((1, d1)).flatten(),
            permutation.flatten()
        ].view(d1, d2, d3)
    else:
        ValueError("Only 3 dimensions maximum")
    return ret


def neuralsortsoft(scores, tau):
    """
    scores:
        tensor of size n_samples x d x 1
    tau:
        float
    returns:
        tensor of size n_samples x d x d (soft permutation matrices)
    """
    d = scores.size(1)
    one = torch.ones((d, 1), dtype=torch.get_default_dtype())
    A_s = torch.abs(scores - scores.permute(0, 2, 1))
    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))
    scaling = (d + 1 - 2 * (torch.arange(d) + 1)).type(torch.get_default_dtype())
    C = torch.matmul(scores, scaling.unsqueeze(0))
    P_max = (C - B).permute(0, 2, 1)

    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)

    return P_hat
