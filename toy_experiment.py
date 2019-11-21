# based on Sean Robertson's pydrobert-pytorch.estimators

import argparse
import numpy as np
import torch
from estimators import to_z, to_b, reinforce, relax, ExactGradientEstimator
from critics import REBARCritic, RELAXCritic
from utils import make_permutation_matrix
import os
from itertools import chain
import pickle
from tqdm import tqdm


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=8, help='Dimension of vector to sort')
    parser.add_argument('--eps', type=float, default=0.05, help="small value to add to diagonal")
    parser.add_argument('--estimator', choices=['reinforce', 'rebar', 'relax', 'exact'], default='reinforce')
    parser.add_argument('--hidden', type=int, default=32, help='Dimension of hidden layer for RELAX critic')
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--iters', type=int, default=50000, help='Number of iterations to train')
    parser.add_argument('--precision', choices=["float", "double"], default="double", help="Default tensor type")
    parser.add_argument('--exp_path', type=str, default="./exp_log/", help='Where to save all stuff')
    parser.add_argument('--seed', type=int, default=40, help='Random seed')
    return parser.parse_args(args)


def loss(P, target_P):
    return torch.norm(P - target_P, dim=(-2, -1))**2


def make_target(d, eps):
    target = torch.ones(d, d) / d
    target[torch.arange(d), torch.arange(d)] += eps
    off_diag = torch.ones(d, d) * eps / (d - 1)
    off_diag[torch.arange(d), torch.arange(d)] = 0
    target = target - off_diag
    return target


def run_toy_example(args=None):
    args = _parse_args(args)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = "cuda"
        if args.precision == "float":
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")
    else:
        device = "cpu"
        if args.precision == "float":
            torch.set_default_tensor_type("torch.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.DoubleTensor")
    # this random seed setup to control critics initialization
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)

    target_P = make_target(args.d, args.eps)
    f = lambda P: loss(P, target_P)

    log_theta = torch.zeros(args.d, requires_grad=True)

    if args.estimator == "reinforce":
        estimator = reinforce
        critic = None
        tunable = []
    elif args.estimator == "rebar":
        estimator = relax
        critic = REBARCritic(f)
        tunable = critic.parameters()
    elif args.estimator == "relax":
        estimator = relax
        critic = RELAXCritic(f, args.d, args.hidden)
        tunable = critic.parameters()
    else:
        estimator = ExactGradientEstimator(f, args.d)
        critic = None
        tunable = []

    if args.estimator == "exact":
        history = {
            "mean_objective": [],
            "grad_std": []
        }
    else:
        num_mc_samples = 10
        history = {
            "grad_mean": [],
            "grad_std": [],
            "critic_loss": [],
            "mean_objective": [],
            "mode_objective": []
        }
    optim = torch.optim.Adam(chain([log_theta], tunable), args.lr)
    # this random seed setup to get first point of mean objective equal across estimators
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    for i in tqdm(range(args.iters)):
        if args.estimator == "exact":
            optim.zero_grad()
            d_log_theta, mean_objective = estimator(log_theta)
            log_theta.backward(d_log_theta)
            optim.step()
            if i % 500 == 0:
                history["mean_objective"].append(mean_objective.item())
                history["grad_std"].append(np.zeros(args.d))
        else:
            # save statistics
            if i % 500 == 0:
                u_mc = torch.distributions.utils.clamp_probs(torch.rand(num_mc_samples, args.d))
                v_mc = torch.distributions.utils.clamp_probs(torch.rand(num_mc_samples, args.d))
                log_theta_mc = log_theta.unsqueeze(0).expand(num_mc_samples, args.d)
                z_mc = to_z(log_theta_mc, u_mc)
                b_mc = to_b(z_mc)
                f_b_mc = f(make_permutation_matrix(b_mc))
                mean_objective = f_b_mc.mean().item()
                d_log_theta = estimator(
                    fb=f_b_mc, b=b_mc, logits=log_theta_mc,
                    z=z_mc, v=v_mc, c=critic
                ).detach().cpu().numpy()
                if args.estimator == "reinforce":
                    critic_loss = 0.0
                else:
                    critic_loss = (d_log_theta ** 2).sum(axis=1).mean()

                grad_mean = np.mean(d_log_theta, 0)
                grad_std = np.std(d_log_theta, 0)
                history["grad_mean"].append(grad_mean)
                history["grad_std"].append(grad_std)
                history["mean_objective"].append(mean_objective)
                cur_pi = log_theta.topk(args.d)[1]
                cur_P = make_permutation_matrix(cur_pi)
                mode_objective = f(cur_P).item()
                history["mode_objective"].append(mode_objective)
                history["critic_loss"].append(critic_loss)

            # do optimization
            optim.zero_grad()
            u = torch.distributions.utils.clamp_probs(torch.rand_like(log_theta))
            v = torch.distributions.utils.clamp_probs(torch.rand_like(log_theta))
            z = to_z(log_theta, u)
            b = to_b(z)
            f_b = f(make_permutation_matrix(b))
            d_log_theta = estimator(fb=f_b, b=b, logits=log_theta, z=z, c=critic, v=v)
            if tunable:
                (d_log_theta ** 2).sum().backward()
            log_theta.backward(d_log_theta)
            optim.step()

    for key in history.keys():
        history[key] = np.array(history[key])
    with open(args.exp_path + "history_{0}.pkl".format(args.estimator), "wb") as file:
        pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)
    if tunable:
        torch.save(critic.state_dict(), args.exp_path + "critic_{0}.pt".format(args.estimator))
    torch.save(log_theta.detach().cpu(), args.exp_path + "log_theta_{0}.pt".format(args.estimator))

if __name__ == '__main__':
    run_toy_example()
