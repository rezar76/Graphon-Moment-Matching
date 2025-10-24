from math import exp
import os
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
ORCA_DIR = '../orca/'

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
    tmp_file_path = os.path.join(ORCA_DIR, f'tmp-{random.random():.4f}.txt')
    f = open(tmp_file_path, 'w+')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()


    output = sp.check_output(["../orca/orca",'4', tmp_file_path, 'output.txt'])
    with open('output.txt', 'r') as file:
        output = file.read()
    output = output.strip()
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
                                  for node_cnts in output.strip('\n').split('\n')])
    try:
        os.remove(tmp_file_path)
    except OSError:
        pass

    return node_orbit_counts


def count2density(node_orbit_counts, graph_size):


    all_possible_motifs = {}
    for size in [2, 3, 4]:
        all_possible_motifs[size] = comb(graph_size, size, exact=True)

    map_loc2motif = [1, 2, 1, 2, 2, 1, 3, 2, 1]
    node_size = np.array(1*[2] + 2*[3] + 6*[4])
    rewiring_normalizer = [ 1.,  3.,  1., 12.,  4.,  3., 12.,  6.,  1.]
    non_unique_count = np.zeros(9)
    density = np.zeros(9)

    # summing over nodes
    count_over_nodes = np.sum(node_orbit_counts, axis=0)
    non_unique_count[0] = count_over_nodes[0]
    for i in range(1, 9):
      start_idx = sum(map_loc2motif[:i])
      non_unique_count[i] = sum(count_over_nodes[start_idx: start_idx+map_loc2motif[i]])

    unique_count = non_unique_count / node_size

    for i in range(9):
      density[i] = unique_count[i] / (rewiring_normalizer[i] * all_possible_motifs[node_size[i]])
    return density


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
        #x = torch.sigmoid(self.fc3(x)) # Output between 0 and 1
        x = torch.clamp(self.fc3(x), 0, 1)
        return x

def estimate_moments_efficient(net, X, Es):
    N = X.shape[0]
    est_moments = torch.ones((N, len(Es)), device=X.device)
    for e, E in enumerate(Es):
        num_dim = E[0]
        edges = torch.tensor(E[1], device=X.device)
        not_edges = torch.tensor(E[2], device=X.device)

        all_edges = torch.cat([edges, not_edges], dim=0)
        is_not_edge = torch.cat([torch.zeros(edges.shape[0]), torch.ones(not_edges.shape[0])]).bool().to(X.device)

        i_indices = all_edges[:, 0].int()
        j_indices = all_edges[:, 1].int()

        selected_X = X[:, i_indices, None]  # Shape (N, num_edges + num_not_edges, 1)
        selected_X = torch.cat((selected_X, X[:, j_indices, None]), dim=2)  # Shape (N, num_edges + num_not_edges, 2)

        sorted_X, _ = torch.sort(selected_X, dim=2)
        #print(sorted_X.shape)

        outputs = net(sorted_X).squeeze(dim=-1)
        

        outputs = torch.where(is_not_edge[None, :], 1 - outputs, outputs)

        est_moments[:, e] = torch.prod(outputs, dim=1)

    return est_moments.mean(0)

def w_mse(weights, estimate, target):
    return torch.mean((weights * (estimate - target)) ** 2)

def train_momentnet(moment_net, E_list, real_moments, k_max, N, epochs, patience, lr, device, weight_mode = 0):
    optimizer = optim.Adam(moment_net.parameters(), lr=lr)
    #optimizer = optim.SGD(moment_net.parameters(), lr=lr)
    #optimizer = schedulefree.AdamWScheduleFree(moment_net.parameters(), lr=lr, betas = (0.95, 0.9))
    # check if the optimizer has train method
    #optimizer.train()
    # Training loop with early stopping
    best_loss = float('inf')
    epochs_no_improve = 0

    if weight_mode == 0:
        denumerator = real_moments.clone()
        for i in range(len(real_moments)):
            if denumerator[i] < 1e-9:
                denumerator[i] = 1
        weights = real_moments.sum() / (denumerator) # The ones with more nodes are less common, so we give a larger weight
    else:
        weights = torch.ones(len(E_list)).to(device)
    
    



    losses = []
    reset = 0
    for epoch in range(epochs):

        X = torch.rand(N, k_max).to(device)
        est_moments = estimate_moments_efficient(moment_net, X, E_list)

        optimizer.zero_grad()
        # moments loss
        loss = w_mse(weights, est_moments, real_moments)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
        # Early stopping check
        if loss < best_loss:
            best_loss = loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch < 200:
            if epoch > 1:
                if np.abs(losses[-1] - losses[-2]) < 1e-9:
                    reset += 1

            if reset == 2:
                return

        if epochs_no_improve == patience:
            #print(f'Early stopping triggered at epoch {epoch}')
            break


    #print('Finished Training')
    return losses


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
        x = torch.clamp(x, 0, 1)
        return x