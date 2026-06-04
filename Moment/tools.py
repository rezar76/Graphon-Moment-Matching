from math import exp
from functools import lru_cache
import os
import copy
import tempfile
import uuid
import numpy as np
import random
import networkx as nx
import subprocess as sp
from scipy.special import comb
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F
from einops import rearrange
import ot
from tqdm import tqdm
from typing import List
from scipy.integrate import nquad
import schedulefree
import math
import torch.optim as optim

seed = 21
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORCA_DIR = os.path.join(_REPO_ROOT, 'orca')
ORCA_BIN = os.path.join(ORCA_DIR, 'orca')

def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    if not os.path.exists(ORCA_BIN):
        raise FileNotFoundError(
            f'ORCA executable not found at {ORCA_BIN}. Clone '
            'https://github.com/thocevar/orca.git into ./orca and run '
            '`make -C orca` before running MomentNet experiments.'
        )

    # Use a unique id per call to prevent collisions when many calls happen in
    # quick succession (random.random() is seeded module-wide, and 4 decimal
    # digits is too narrow). Also write the orca output to a unique path so
    # concurrent calls do not stomp a shared 'output.txt'.
    unique_id = f"{os.getpid()}-{uuid.uuid4().hex}"
    tmp_file_path = os.path.join(ORCA_DIR, f'tmp-{unique_id}.txt')
    out_file_path = os.path.join(ORCA_DIR, f'out-{unique_id}.txt')

    with open(tmp_file_path, 'w+') as f:
        f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
        for (u, v) in edge_list_reindexed(graph):
            f.write(str(u) + ' ' + str(v) + '\n')

    sp.check_output([ORCA_BIN, '4', tmp_file_path, out_file_path])
    with open(out_file_path, 'r') as file:
        output = file.read()
    output = output.strip()
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
                                  for node_cnts in output.strip('\n').split('\n')])
    for path in (tmp_file_path, out_file_path):
        try:
            os.remove(path)
        except OSError:
            pass

    return node_orbit_counts


_MOTIF_NODE_SIZES = np.array(1 * [2] + 2 * [3] + 6 * [4])
_MOTIF_LOC_MAP = [1, 2, 1, 2, 2, 1, 3, 2, 1]
_MOTIF_REWIRING = np.array([1., 3., 1., 12., 4., 3., 12., 6., 1.])


def count2unique_count(node_orbit_counts):
    """Per-graph unique motif counts (length-9 vector, no normalization by graph size)."""
    map_loc2motif = _MOTIF_LOC_MAP
    node_size = _MOTIF_NODE_SIZES
    non_unique_count = np.zeros(9)
    count_over_nodes = np.sum(node_orbit_counts, axis=0)
    non_unique_count[0] = count_over_nodes[0]
    for i in range(1, 9):
        start_idx = sum(map_loc2motif[:i])
        non_unique_count[i] = sum(count_over_nodes[start_idx: start_idx + map_loc2motif[i]])
    return non_unique_count / node_size


def count2density(node_orbit_counts, graph_size):
    """Per-graph induced motif densities."""
    unique_count = count2unique_count(node_orbit_counts)
    density = np.zeros(9)
    for i in range(9):
        all_possible = comb(graph_size, _MOTIF_NODE_SIZES[i], exact=True)
        density[i] = unique_count[i] / (_MOTIF_REWIRING[i] * all_possible)
    return density


def aggregate_moments(graphs):
    """Aggregate motif densities across a list of graphs of possibly varying sizes.

    Uses the method-of-moments estimator weighted by the number of node-tuples in
    each graph, i.e. density_i = sum_g count_g_i / sum_g (rewiring_i * C(n_g, k_i)).
    This is much more stable than averaging per-graph densities uniformly when
    graph sizes differ, because small graphs are far noisier per motif.
    """
    sizes = _MOTIF_NODE_SIZES
    rewiring = _MOTIF_REWIRING
    total_unique_counts = np.zeros(9)
    total_possible_motifs = np.zeros(9)
    for graph in graphs:
        node_orbit_counts = orca(graph)
        unique_counts = count2unique_count(node_orbit_counts)
        n = graph.number_of_nodes()
        possible = np.array([comb(n, k, exact=True) for k in sizes], dtype=np.float64)
        total_unique_counts += unique_counts
        total_possible_motifs += rewiring * possible
    total_possible_motifs = np.where(total_possible_motifs == 0, 1.0, total_possible_motifs)
    return total_unique_counts / total_possible_motifs


def integrand(x, W, edges, not_edges):
    product = np.ones(x.shape[0])

    for edge in edges:
        i = edge[0]
        j = edge[1]
        product *= W(x[:,i], x[:,j])

    for edge in not_edges:
        i = edge[0]
        j = edge[1]
        product *= (1 - W(x[:,i], x[:,j]))

    return product

def approximate_integral(W, k, edges, not_edges, integrand_fn, num_samples=10000): # Monte-Carlo method
    x = np.random.rand(num_samples, k)  # Generate random points in [0,1]^k
    integral_vals = integrand_fn(x, W, edges, not_edges)
    return integral_vals.mean()


# convert motifs into induced format (this one will preserve the non-induced motifs)
def motifs_to_induced_motifs(Es):
    Es_induced = []
    for num_nodes, Es_k in enumerate(Es):
        #if num_nodes == 0:
        #    print("yes")
        #    continue
        #    Es_induced.append([2, [(0,0)], []])
        #    continue
        num_nodes += 1
        for motif in Es_k:
            #print(motif)
            edges = []
            not_edges = []

            for k in range(num_nodes):
                for kk in range(k+1, num_nodes+1):
                    if (k, kk) in motif:
                        edges.append((k, kk))
                    else:
                        not_edges.append((k, kk))

            Es_induced.append([num_nodes+1, edges, not_edges])
    return Es_induced


def compute_real_moments_induced(W, Es, N_mc=1000000, force_approx=False, print_details=True):
    results = {}
    for motif in Es:
        num_dim = motif[0]
        edges = motif[1]
        not_edges = motif[2]
        if force_approx or num_dim >= 3: # After squares, we go with the MC method to approximate the integral
            integral_val = approximate_integral(W, num_dim, edges, not_edges, integrand_fn=integrand, num_samples=N_mc)
            if print_details:
                print(f"Aproximate integral for {motif}: {integral_val}")
        else:
            # Calculate the exact integral using nquad
            def integrand_wrapper(*args):
                product = 1

                for edge in edges:
                    i = edge[0]
                    j = edge[1]
                    product *= W(args[i], args[j])

                for not_edge in not_edges:
                    i = not_edge[0]
                    j = not_edge[1]
                    product *= (1 - W(args[i], args[j]))


                return product
            ranges = [[0, 1]] * num_dim  # Integration ranges [0,1]
            integral_val, error = nquad(integrand_wrapper, ranges)
            if print_details:
                print(f"Exact integral for {motif}: {integral_val}")
        if str(motif) in results:
            print(f"Warning: motif {motif} already exists in the results")
        results[str(motif)] = integral_val
    results_arr = list(results.values())
    real_moments = torch.tensor(results_arr)
    return real_moments

class Net(nn.Module):
    def __init__(self, hid_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.clamp(self.fc3(x), min=0.0, max=1.0)
        return x


class SiLUMLP(nn.Module):
    def __init__(self, hid_dim=192, num_layers=3):
        super().__init__()
        modules = []
        in_dim = 2
        for _ in range(num_layers):
            modules.append(nn.Linear(in_dim, hid_dim))
            modules.append(nn.SiLU())
            in_dim = hid_dim
        modules.append(nn.Linear(in_dim, 1))
        modules.append(nn.Sigmoid())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


def initialize_output_mean(model, edge_density):
    edge_density = float(np.clip(edge_density, 1e-4, 1 - 1e-4))
    bias = np.log(edge_density / (1 - edge_density))
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        return
    with torch.no_grad():
        linears[-1].weight.mul_(0.05)
        linears[-1].bias.fill_(bias)


def _moment_plan_key(Es):
    return tuple(
        (int(num_dim),
         tuple((int(i), int(j)) for i, j in edges),
         tuple((int(i), int(j)) for i, j in not_edges))
        for num_dim, edges, not_edges in Es
    )


@lru_cache(maxsize=128)
def _cached_moment_plan(plan_key, device_name):
    """Precompute pair/motif incidence tensors for moment estimation.

    The same pair probabilities are reused across motifs. For graphon 9 this
    cuts MLP pair evaluations from 43N per epoch to 6N per epoch.
    """
    device = torch.device(device_name)
    pair_set = set()
    for _, edges, not_edges in plan_key:
        pair_set.update(tuple(sorted(pair)) for pair in edges)
        pair_set.update(tuple(sorted(pair)) for pair in not_edges)
    pairs = sorted(pair_set)

    pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}
    use_mask = torch.zeros((len(plan_key), len(pairs)), dtype=torch.bool, device=device)
    edge_mask = torch.zeros((len(plan_key), len(pairs)), dtype=torch.bool, device=device)

    for motif_idx, (_, edges, not_edges) in enumerate(plan_key):
        for pair in edges:
            idx = pair_to_idx[tuple(sorted(pair))]
            use_mask[motif_idx, idx] = True
            edge_mask[motif_idx, idx] = True
        for pair in not_edges:
            idx = pair_to_idx[tuple(sorted(pair))]
            use_mask[motif_idx, idx] = True

    if pairs:
        pair_i = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=device)
        pair_j = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=device)
    else:
        pair_i = torch.empty(0, dtype=torch.long, device=device)
        pair_j = torch.empty(0, dtype=torch.long, device=device)

    return pair_i, pair_j, use_mask, edge_mask


def estimate_moments_efficient(net, X, Es):
    pair_i, pair_j, use_mask, edge_mask = _cached_moment_plan(
        _moment_plan_key(Es), str(X.device))

    if pair_i.numel() == 0:
        return torch.ones(len(Es), device=X.device, dtype=X.dtype)

    left = X[:, pair_i]
    right = X[:, pair_j]
    pair_inputs = torch.stack((torch.minimum(left, right),
                               torch.maximum(left, right)), dim=-1)
    pair_probs = net(pair_inputs).squeeze(dim=-1)

    pair_terms = torch.where(edge_mask.unsqueeze(0),
                             pair_probs.unsqueeze(1),
                             1 - pair_probs.unsqueeze(1))
    pair_terms = torch.where(use_mask.unsqueeze(0),
                             pair_terms,
                             torch.ones((), device=X.device, dtype=pair_terms.dtype))
    return pair_terms.prod(dim=2).mean(dim=0)

def w_mse(weights, estimate, target):
    return torch.mean((weights * (estimate - target)) ** 2)

def _reinit_module(module):
    """Re-initialize a torch module's parameters in-place.

    Handles plain nn.Linear, the project's LipschitzLinear, and SirenNet's
    Siren layers. For everything else, falls back to xavier_uniform on weights
    and zero on biases."""
    cls_name = module.__class__.__name__
    if cls_name == 'LipschitzLinear':
        # Project-defined layer in Moment.tools.lipmlp; has its own init.
        module.initialize_parameters()
        return True
    if cls_name == 'Siren':
        # SIGL.tools.Siren — its forward calls F.linear with module.weight/bias.
        w0 = getattr(module, 'w0', getattr(getattr(module, 'activation', None), 'w0', 1.))
        module.init_(module.weight.data, module.bias.data if module.bias is not None else None,
                     c=6., w0=w0)
        return True
    if isinstance(module, nn.Linear):
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
            return True
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
            return True
    return False


def reinitialize_model(model, seed=None):
    """Walk the model and re-initialize every parameter-bearing leaf module."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    touched = 0
    for m in model.modules():
        if m is model:
            continue
        if _reinit_module(m):
            touched += 1
    if touched == 0:
        # Fallback: randomize all leaf parameters with xavier/zeros.
        for p in model.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)


def train_momentnet(moment_net, E_list, real_moments, k_max, N, epochs, patience, lr, device,
                    weight_mode=0, eval_N=None, eval_interval=20, ema_alpha=0.1, grad_clip=1.0,
                    min_lr=1e-6, lr_decay_factor=0.5, lr_decay_patience=None, verbose=False,
                    collapse_threshold=0.1, collapse_check_epoch=200, max_collapse_resets=5):
    """Train a moment-matching network with several stability safeguards.

    Stability fixes vs. the original:
      * Always returns a list (never None) so the caller's bare ``losses[-1]``
        check cannot silently spin in an outer ``while True`` loop.
      * Tracks the best model state by an EMA-smoothed loss and a periodic
        large-N evaluation, restoring it before returning. Stops the network
        from drifting on noisy MC gradients.
      * Detects NaN / Inf and breaks instead of poisoning the trained weights.
      * Adds gradient clipping and a ReduceLROnPlateau scheduler so the model
        does not oscillate when the noisy gradient flips sign.
      * Removes the broken stagnation-reset that bare-``return``-ed None.
      * **Collapse auto-reset**: after ``collapse_check_epoch`` warmup epochs,
        if the best eval loss is still above ``collapse_threshold`` (a sign
        the network parked at the constant-prediction plateau, eg loss ~0.26
        for graphon 9 with weighted-MSE), re-initialize the model and try
        again, up to ``max_collapse_resets`` times. This avoids burning a full
        outer-loop attempt on an obviously-bad init.

    Args:
        eval_N: Sample count for the *evaluation* MC pass used for best-model
            tracking and early stopping. Defaults to ``max(N, 4*N)`` clamped at
            200k for memory. Larger -> less noisy decisions.
        eval_interval: Evaluate every ``eval_interval`` epochs. The training MC
            loss is too noisy to drive early stopping by itself.
        ema_alpha: Smoothing factor for the per-epoch training loss EMA.
        grad_clip: Max L2 norm for gradient clipping (None to disable).
        lr_decay_patience: Patience for ReduceLROnPlateau (None -> patience//4).
        collapse_threshold: If best eval loss > this after the warmup window,
            consider the run collapsed and reinit. Default 0.1 (well above the
            'medium' plateau ~0.016 we see on graphon 9).
        collapse_check_epoch: Warmup epochs before the collapse check kicks in.
        max_collapse_resets: How many times we'll reinit before giving up.
    """
    if lr_decay_patience is None:
        lr_decay_patience = max(1, patience // 4)

    if weight_mode == 0:
        denumerator = real_moments.clone()
        for i in range(len(real_moments)):
            if denumerator[i] < 1e-9:
                denumerator[i] = 1
        weights = real_moments.sum() / denumerator
    else:
        weights = torch.ones(len(E_list)).to(device)

    if eval_N is None:
        eval_N = min(max(N, 4 * N), 200000)

    real_moments_dev = real_moments.to(device)
    weights_dev = weights.to(device)

    def _evaluate():
        moment_net.eval()
        with torch.no_grad():
            X_eval = torch.rand(eval_N, k_max, device=device)
            est = estimate_moments_efficient(moment_net, X_eval, E_list)
            eval_loss = w_mse(weights_dev, est, real_moments_dev).item()
        moment_net.train()
        return eval_loss

    losses_all = []
    best_eval_loss_overall = float('inf')
    best_state_overall = None
    collapse_resets = 0

    while True:
        # Per-attempt state
        optimizer = optim.Adam(moment_net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_decay_factor,
            patience=lr_decay_patience, min_lr=min_lr)

        ema_loss = None
        best_eval_loss = _evaluate()
        best_state = {k: v.detach().clone() for k, v in moment_net.state_dict().items()}
        epochs_no_improve = 0
        attempt_losses = []
        collapsed = False

        for epoch in range(epochs):
            X = torch.rand(N, k_max, device=device)
            est_moments = estimate_moments_efficient(moment_net, X, E_list)

            optimizer.zero_grad(set_to_none=True)
            loss = w_mse(weights_dev, est_moments, real_moments_dev)

            if not torch.isfinite(loss):
                if verbose:
                    print(f"Non-finite loss at epoch {epoch}; restoring best and stopping.")
                break

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(moment_net.parameters(), max_norm=grad_clip)
            optimizer.step()

            loss_val = loss.item()
            attempt_losses.append(loss_val)
            ema_loss = loss_val if ema_loss is None else ema_alpha * loss_val + (1 - ema_alpha) * ema_loss

            if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
                eval_loss = _evaluate()
                scheduler.step(eval_loss)

                if eval_loss < best_eval_loss - 1e-8:
                    best_eval_loss = eval_loss
                    best_state = {k: v.detach().clone() for k, v in moment_net.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += eval_interval

                # Collapse detection: after a warmup window, if best eval loss
                # is still far above any reasonable plateau, give up on this
                # init and re-initialize.
                if (epoch + 1 >= collapse_check_epoch
                        and best_eval_loss > collapse_threshold
                        and collapse_resets < max_collapse_resets):
                    collapsed = True
                    if verbose:
                        print(f"  collapse detected at epoch {epoch+1} (best_eval_loss={best_eval_loss:.4f}); resetting init")
                    break

                if epochs_no_improve >= patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch + 1}, best eval loss {best_eval_loss:.6f}')
                    break

        losses_all.extend(attempt_losses)

        # Track the best across attempts in case all attempts collapse.
        if best_eval_loss < best_eval_loss_overall:
            best_eval_loss_overall = best_eval_loss
            best_state_overall = best_state

        if collapsed and collapse_resets < max_collapse_resets:
            collapse_resets += 1
            reinitialize_model(moment_net,
                               seed=int(torch.randint(0, 2**31 - 1, (1,)).item()))
            continue
        break

    if best_state_overall is not None:
        moment_net.load_state_dict(best_state_overall)
    if not losses_all:
        losses_all = [best_eval_loss_overall]
    return losses_all


def _degree_variance(model, anchors, integration):
    x = anchors[:, None].repeat(1, integration.shape[0]).reshape(-1, 1)
    y = integration[None, :].repeat(anchors.shape[0], 1).reshape(-1, 1)
    pairs = torch.cat((torch.minimum(x, y), torch.maximum(x, y)), dim=1)
    degrees = model(pairs).reshape(anchors.shape[0], integration.shape[0]).mean(dim=1)
    return degrees.var(unbiased=False)


def train_momentnet_sobol_degree(moment_net, E_list, real_moments, k_max, N, epochs, lr,
                                 device, seed=0, eval_N=None, eval_interval=50,
                                 warmup_weight=2.0, warmup_end=1200, min_var_weight=0.0,
                                 weight_mode=0, grad_clip=1.0, weight_decay=1e-6,
                                 eta_min=2e-5, weight_floor=None, weight_cap=None,
                                 verbose=False):
    """Train with fixed Sobol samples and validation-moment checkpointing.

    This variant is used for graphon 9 because random MC sampling made the
    moment objective noisy enough to choose unstable checkpoints. Selection is
    still based only on held-out moment estimates, not on GW or the true graphon.
    """
    if weight_mode == 0:
        if weight_floor is None:
            denumerator = real_moments.clone()
            for i in range(len(real_moments)):
                if denumerator[i] < 1e-9:
                    denumerator[i] = 1
            weights = real_moments.sum() / denumerator
        else:
            weights = real_moments.sum() / torch.clamp(real_moments, min=weight_floor)
            if weight_cap is not None:
                weights = torch.clamp(weights, max=weight_cap)
    else:
        weights = torch.ones(len(E_list), device=device)

    if eval_N is None:
        eval_N = max(N, 2 * N)

    target = real_moments.to(device)
    weights = weights.to(device)
    initialize_output_mean(moment_net, float(target[0].detach().cpu()))

    train_x = torch.quasirandom.SobolEngine(k_max, scramble=True, seed=seed).draw(N).to(device)
    eval_x = torch.quasirandom.SobolEngine(k_max, scramble=True, seed=seed + 1).draw(eval_N).to(device)
    anchors = torch.quasirandom.SobolEngine(1, scramble=True, seed=seed + 2).draw(96).squeeze(-1).to(device)
    integration = torch.quasirandom.SobolEngine(1, scramble=True, seed=seed + 3).draw(256).squeeze(-1).to(device)

    target_degree_variance = None
    if len(target) >= 3:
        target_degree_variance = target[1] + target[2] - target[0].square()
        target_degree_variance = torch.clamp(target_degree_variance, min=1e-8)

    optimizer = optim.AdamW(moment_net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    best_eval_loss = float('inf')
    best_state = None
    best_epoch = 0
    eval_losses = []

    for epoch in range(1, epochs + 1):
        estimate = estimate_moments_efficient(moment_net, train_x, E_list)
        base_loss = w_mse(weights, estimate, target)
        loss = base_loss

        if target_degree_variance is not None and (warmup_weight > 0 or min_var_weight > 0):
            model_degree_variance = _degree_variance(moment_net, anchors, integration)
            var_loss = ((model_degree_variance - target_degree_variance)
                        / target_degree_variance).square()
            var_weight = max(min_var_weight, warmup_weight * (1 - epoch / warmup_end))
            loss = loss + var_weight * var_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(moment_net.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if epoch % eval_interval == 0 or epoch == epochs:
            moment_net.eval()
            with torch.no_grad():
                eval_estimate = estimate_moments_efficient(moment_net, eval_x, E_list)
                eval_loss = w_mse(weights, eval_estimate, target).item()
            moment_net.train()
            eval_losses.append(eval_loss)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone()
                              for k, v in moment_net.state_dict().items()}

            if verbose and epoch % 100 == 0:
                print(f'epoch={epoch:04d} loss={loss.item():.8f} '
                      f'base={base_loss.item():.8f} eval={eval_loss:.8f} '
                      f'best_eval={best_eval_loss:.8f}@{best_epoch}',
                      flush=True)

    if best_state is not None:
        moment_net.load_state_dict(best_state)
    if not eval_losses:
        eval_losses = [best_eval_loss]

    return eval_losses, {
        'best_eval_loss': best_eval_loss,
        'best_epoch': best_epoch,
        'seed': seed,
        'lr': lr,
    }


def gw_distance(graphon: np.ndarray, estimation: np.ndarray) -> float:
    p = np.ones((graphon.shape[0],)) / graphon.shape[0]
    q = np.ones((estimation.shape[0],)) / estimation.shape[0]
    loss_fun = 'square_loss'
    dw2 = ot.gromov.gromov_wasserstein2(graphon, estimation, p, q, loss_fun, log=False, armijo=False)
    return np.sqrt(dw2)

def comp_GW_loss(net, W, resolution=1000):
    # Generate data for the plot
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)

    # Network output
    inputs = torch.tensor(np.stack((X.flatten(), Y.flatten()), axis=1), dtype=torch.float32)
    Z_net = net(inputs).cpu().detach().numpy().reshape(X.shape)
    Z_sym = np.copy(Z_net)

    # Copy lower triangle to upper triangle (excluding the diagonal)
    #i_lower = np.tril_indices(Z_net.shape[0], -1)
    #Z_sym[i_lower] = Z_sym.T[i_lower]

    # Copy upper triangle to lower triangle (excluding the diagonal)
    i_upper = np.triu_indices(Z_net.shape[0], 1)
    Z_sym[i_upper] = Z_sym.T[i_upper]

    # Real graphon
    Z_real = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
          if X[i,j] < Y[i,j]:
            xx = X[i,j]
            yy = Y[i,j]
          else:
            xx = Y[i,j]
            yy = X[i,j]
          Z_real[i, j] = W(xx, yy)


    # set the diagonal of Z_sym equal to Z_real
    np.fill_diagonal(Z_real, 0)
    np.fill_diagonal(Z_sym, 0)
    #np.fill_diagonal(Z_sym, np.diag(Z_real))

    # replace 1 with 0.8 in Z_sym
    
    #print("GW Distance  = " + str(gw_distance(Z_sym, Z_real)))
    return gw_distance(Z_sym, Z_real)

class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        # compute lipschitz constant of initial weight to initialize self.c
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max() # just a rough initialization

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)

class lipmlp(torch.nn.Module):
    def __init__(self, dims):
        """
        dim[0]: input dim
        dim[1:-1]: hidden dims
        dim[-1]: out dim

        assume len(dims) >= 3
        """
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for ii in range(len(dims)-2):
            self.layers.append(LipschitzLinear(dims[ii], dims[ii+1]))

        self.layer_output = LipschitzLinear(dims[-2], dims[-1])
        self.relu = torch.nn.ReLU()

    def get_lipschitz_loss(self):
        loss_lipc = 1.0
        for ii in range(len(self.layers)):
            loss_lipc = loss_lipc * self.layers[ii].get_lipschitz_constant()
        loss_lipc = loss_lipc *  self.layer_output.get_lipschitz_constant()
        return loss_lipc

    def forward(self, x):
        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            x = self.relu(x)
        x = self.layer_output(x)
        return torch.clamp(x, min=0.0, max=1.0)
