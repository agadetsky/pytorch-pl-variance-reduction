import torch
import torch.nn as nn
from utils import neuralsortsoft


class REBARCritic(nn.Module):
    """
        eta * f(NeuralSort(z, tau))
    """

    def __init__(self, f):
        super(REBARCritic, self).__init__()
        self.f = f
        self.eta = nn.Parameter(torch.ones(1))
        self.log_tau = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        assert (z.ndimension() == 1) or (z.ndimension() == 2)
        if z.ndimension() == 1:  # just one sample
            z = z.unsqueeze(0)
        z = z.unsqueeze(-1)
        tau = torch.exp(self.log_tau)
        return self.eta * self.f(neuralsortsoft(z, tau)).squeeze(0).unsqueeze(-1)


class RELAXCritic(nn.Module):
    """
        f(NeuralSort(z, tau)) + rho_\phi(z)
    """

    def __init__(self, f, d, hidden_dim):
        super(RELAXCritic, self).__init__()
        self.f = f
        self.log_tau = nn.Parameter(torch.zeros(1))
        self.rho = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        assert (z.ndimension() == 1) or (z.ndimension() == 2)
        if z.ndimension() == 1:  # just one sample
            z = z.unsqueeze(0)
        z = z.unsqueeze(-1)
        tau = torch.exp(self.log_tau)
        return self.f(neuralsortsoft(z, tau)).squeeze(0).unsqueeze(-1) + self.rho(z.squeeze())
