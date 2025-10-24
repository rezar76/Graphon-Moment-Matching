import sys
import os 
sys.path.append('../')
import pickle
import numpy as np
import torch
import time
import pandas as pd
from GraphTools.utils import general_graphon, compare_centrality_measures
from Moment.tools import motifs_to_induced_motifs, orca, count2density, lipmlp, train_momentnet
import networkx as nx
from SIGL.tools import comp_GW_loss
from tqdm import tqdm
from SIGL.tools import SirenNet
def train_Moment(dataset_name, graphon_idx):
    """
    Train the MomentNet model using the dataset and return GW loss, centrality NMSE averages, and standard deviations.

    Args:
        dataset_name (str): Name of the dataset file in the dataset folder.
        graphon_idx (int): Index of the graphon.

    Returns:
        tuple: Avg GW loss, Std GW loss, Avg and Std NMSE for each centrality measure.
    """
    # Load dataset
    dataset_path = os.path.join('../dataset', dataset_name)
    with open(dataset_path, 'rb') as f:
        graphs_inr = pickle.load(f)

    # Determine n_trials based on the length of the loaded dataset
    n_trials = len(graphs_inr)

    # Default parameters
    epochs = 7000
    patience = 600
    lr = 1e-3
    N = 30000
    hid_dim = 64
    num_motifs = 9
    weight_mode = 0
    model_name = "MLP"


    # extract dataset index
    dataset_idx = int(dataset_name.split('_')[1].split('.')[0])

    if dataset_idx == 0:
        hid_dim = 128
        lr = 3.6e-4
        N = 11000
        num_layers = 3
    elif dataset_idx == 1:
        hid_dim = 128
        lr = 2.1e-4
        N = 7000
        num_layers = 5
    elif dataset_idx == 2:
        hid_dim = 128
        lr = 8.6e-4
        N = 3000
        num_layers = 3
    elif dataset_idx == 3:
        hid_dim = 96
        lr = 4.1e-4
        N = 17000
        num_layers = 2
    elif dataset_idx == 4:
        hid_dim = 112
        lr = 6.5e-4
        N = 9000
        num_layers = 2
    elif dataset_idx == 5:
        hid_dim = 112
        lr = 2e-5
        N = 11000
        num_layers = 2
    elif dataset_idx == 11:
        hid_dim = 128
        lr = 9.26e-05
        N = 20000
        num_layers = 6
        model_name = "SirenNet"
        w0 = 9.0
        w0_initial = 30.0
        patience = 1000

    elif dataset_idx == 12:
        hid_dim = 128
        lr = 2.5e-4
        N = 11000
        num_layers = 5
        model_name = "SirenNet"
        w0 = 1.784
        w0_initial = 17.2525
        patience = 1000

    


    # Generate true graphon
    if graphon_idx == 11:
        sbm_split = np.array([0.5, 0.5])
        sbm_param = np.array([[0, 0.8], [0.8, 0]])
        true_graphon = general_graphon(graphon_idx, sbm_split, sbm_param)
    else:
        true_graphon = general_graphon(graphon_idx, None, None)

    errors = np.zeros(n_trials)
    centrality_results = []

    Es = [[[(0, 1)]], [[(0, 1), (1, 2)], [(0, 1), (0, 2), (1, 2)]], [[(0, 1), (1, 2), (2, 3)], [(0, 1), (0, 2), (0, 3)], [(0, 1), (0, 2), (1, 3), (2, 3)], [(0, 1), (0, 2), (0, 3), (1, 2)],
                                                                 [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)], [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]],
      [[(0, 1), (1, 2), (2, 3), (3, 4)], [(0, 1), (0, 2), (0, 3), (3, 4)], [(0, 1), (0, 2), (0, 3), (0, 4)], [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4)], [(0, 1), (0, 2), (1, 2), (0, 3), (3, 4)],
       [(0, 1), (0, 2), (0, 3), (0, 4), (3, 4)], [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)], [(0, 1), (0, 2), (0, 3), (2, 4), (3, 4)], [(0, 1), (0, 2), (0, 3), (2, 4), (3, 4), (0, 4)], [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)] ,
       [(0, 1), (0, 2), (0, 3), (2, 4), (3, 4), (2, 3)], [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)], [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)], [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (1, 4), (2, 4)],
       [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4)], [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4)], [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (1, 4), (3, 4)], [(0, 1), (0, 2), (0, 3), (1, 3), (1, 2), (2, 3), (2, 4), (3, 4)],
       [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 4), (3, 4)], [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (1, 3), (2, 4), (3, 4)], [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
       ]]

    induced_list = motifs_to_induced_motifs(Es)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for trial in tqdm(range(n_trials)):
        graphs = graphs_inr[trial]

        # compute the moments with graphs
        estimated_densities = np.zeros(9)
        for graph in graphs:
            node_orbit_counts = orca(graph)
            density = count2density(node_orbit_counts, graph.number_of_nodes())
            estimated_densities += density
        real_moments = estimated_densities / len(graphs)
        real_moments = torch.tensor(real_moments).to(device)

        while True:
            if model_name == "SirenNet":
                hid_dim = num_layers*[96]
                model = SirenNet(2, hid_dim, 1, num_layers=num_layers, w0=w0, w0_initial=w0_initial).train().to(device)
            else:
                layers = [2] + num_layers * [hid_dim] + [1]
                model = lipmlp(layers).train().to(device)
            # convert networkx to adjacency matrix
            losses = train_momentnet(model, induced_list[:num_motifs], real_moments, 4, N, epochs, patience, lr, device, weight_mode)

            try:
                if losses[-1] < 0.01:
                    break
            except:
                continue

        # Compare centrality measures
        # move the model to CPU
        trained_inr = model.cpu()
        centrality_nmse = compare_centrality_measures(trained_inr, graphon_idx)
        centrality_results.append(centrality_nmse)

        # Replace error_trial computation with GW loss using comp_GW_loss
        error_trial = comp_GW_loss(trained_inr, true_graphon)

        errors[trial] = error_trial

       

    # Remove NaN values
    errors = errors[~np.isnan(errors)]

    avg_gw = np.round(np.mean(errors), 4)
    std_gw = np.round(np.std(errors), 4)

    # Compute average and standard deviation for centrality NMSEs
    centrality_keys = centrality_results[0].keys()
    avg_centrality = {key: np.round(np.mean([res[key] for res in centrality_results]), 4) for key in centrality_keys}
    std_centrality = {key: np.round(np.std([res[key] for res in centrality_results]), 4) for key in centrality_keys}

    print(f"Graphon {graphon_idx}: Avg GW distance = {avg_gw}, Std = {std_gw}")
    print(f"Centrality NMSE Averages: {avg_centrality}")
    print(f"Centrality NMSE Standard Deviations: {std_centrality}")

    return avg_gw, std_gw, avg_centrality, std_centrality, trained_inr