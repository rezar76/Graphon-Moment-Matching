import os
import pickle
import numpy as np
import torch
import time
import pandas as pd
from SIGL.tools import *
from GraphTools.utils import general_graphon, compare_centrality_measures
import networkx as nx
from SIGL.tools import comp_GW_loss
from tqdm import tqdm
def train_SIGL(dataset_name, graphon_idx):
    """
    Train the SIGL model using the dataset and return GW loss, centrality NMSE averages, and standard deviations.

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
    alpha = 0.5
    n_G = 10
    offset = 0
    Res = 1000
    n_epochs = 100
    epoch_show = 20
    gnn_dim_hidden = [8, 8, 8]
    inr_dim_hidden = [20, 20]
    batch_size = 1024
    w0 = 10
    lr = 0.01
    hsize = 4

    # Generate true graphon
    if graphon_idx == 11:
        sbm_split = np.array([0.5, 0.5])
        sbm_param = np.array([[0, 0.8], [0.8, 0]])
        true_graphon = general_graphon(graphon_idx, sbm_split, sbm_param)
    else:
        true_graphon = general_graphon(graphon_idx, None, None)

    errors = np.zeros(n_trials)
    all_times = np.zeros(n_trials)
    centrality_results = []

    for trial in tqdm(range(n_trials)):
        start_time = time.time()
        graphs = graphs_inr[trial]
        # convert networkx to adjacency matrix
        graphs = [nx.to_numpy_array(graph) for graph in graphs]
        model_SIGL, _ = coords_prediction(inr_dim_hidden, gnn_dim_hidden, n_epochs, epoch_show, w0, graphs, lr)
        X_all, y_all, w_all = graph2XY(graphs, model_SIGL)
        trained_inr = train_graphon(inr_dim_hidden, w0, X_all, y_all, w_all, n_epochs, epoch_show, lr, batch_size)
        #predictions_graphon = get_graphon(Res, trained_inr)

        # Compare centrality measures
        # move the model to CPU
        trained_inr = trained_inr.cpu()
        centrality_nmse = compare_centrality_measures(trained_inr, graphon_idx)
        centrality_results.append(centrality_nmse)

        end_time = time.time()
        # Replace error_trial computation with GW loss using comp_GW_loss
        error_trial = comp_GW_loss(trained_inr, true_graphon)

        errors[trial] = error_trial
        all_times[trial] = end_time - start_time

       

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

    return avg_gw, std_gw, avg_centrality, std_centrality