import sys
import os
sys.path.append('../')
import pickle
import random
import numpy as np
import torch
from GraphTools.utils import general_graphon, compare_centrality_measures
from Moment.tools import (motifs_to_induced_motifs, aggregate_moments, Net,
                          SiLUMLP, lipmlp, train_momentnet,
                          train_momentnet_sobol_degree, comp_GW_loss)
from tqdm import tqdm
def train_Moment(dataset_name, graphon_idx, evaluate_centrality=False,
                 trial_indices=None):
    """
    Train the MomentNet model using the dataset and return GW loss, centrality NMSE averages, and standard deviations.

    Args:
        dataset_name (str): Name of the dataset file in the dataset folder.
        graphon_idx (int): Index of the graphon.
        evaluate_centrality (bool): Whether to compute the slower centrality metrics.
        trial_indices (list[int] | None): Optional subset of trials to run.

    Returns:
        tuple: Avg GW loss, Std GW loss, Avg and Std NMSE for each centrality measure.
    """
    # Load dataset
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(repo_root, 'dataset', dataset_name)
    with open(dataset_path, 'rb') as f:
        graphs_inr = pickle.load(f)

    # Determine trials based on the length of the loaded dataset.
    if trial_indices is None:
        trial_indices = list(range(len(graphs_inr)))
    else:
        trial_indices = list(trial_indices)
    n_trials = len(trial_indices)

    # Default parameters
    epochs = 7000
    patience = 600
    lr = 1e-3
    N = 30000
    hid_dim = 64
    num_layers = 3
    num_motifs = 9
    weight_mode = 0
    model_name = "LipMLP"
    use_sobol_zero_weights = False
    zero_weight_floor = None
    zero_weight_cap = None


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
    elif dataset_idx == 9:
        hid_dim = 192
        lr = 2e-3
        N = 90000
        epochs = 3200
        num_layers = 3
        model_name = "SiLUMLP"
        graphon9_eval_N = 180000
    elif dataset_idx == 11:
        hid_dim = 128
        lr = 1e-3
        N = 60000
        epochs = 1500
        num_layers = 6
        model_name = "SirenNet"
        w0 = 1.0
        w0_initial = 5.0
        patience = 1000
        use_sobol_zero_weights = True
        zero_weight_floor = 1e-3
        zero_weight_cap = 100.0

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

    max_retries = 6
    convergence_threshold = 0.01
    if use_sobol_zero_weights:
        max_retries = 1

    trained_inr = None
    for error_idx, trial in enumerate(tqdm(trial_indices)):
        graphs = graphs_inr[trial]

        # Aggregate motif moments across the (varying-sized) graphs in this
        # graphon instance using a count-weighted estimator. This is much more
        # stable than the previous uniform-average-of-per-graph-densities
        # because small graphs are far noisier per motif.
        real_moments_np = aggregate_moments(graphs)
        real_moments = torch.tensor(real_moments_np, dtype=torch.float32).to(device)

        best_losses = None
        best_model = None
        best_final_loss = float('inf')

        if model_name == "SiLUMLP":
            graphon9_attempts = [
                (9001 + trial, 2e-3),
                (9200 + trial, 2e-3),
            ]
            attempt_iter = list(enumerate(graphon9_attempts))
        else:
            attempt_iter = [(attempt, None) for attempt in range(max_retries)]

        for attempt, attempt_config in attempt_iter:
            # Re-seed per (trial, attempt) so that retries explore a different
            # init / MC trajectory rather than repeating the same trajectory.
            if model_name == "SiLUMLP":
                attempt_seed, attempt_lr = attempt_config
            elif use_sobol_zero_weights:
                attempt_seed = 11001 + attempt
                attempt_lr = lr
            else:
                attempt_seed = 1000 * graphon_idx + 100 * trial + attempt + 1
                attempt_lr = lr
            torch.manual_seed(attempt_seed)
            torch.cuda.manual_seed_all(attempt_seed)
            np.random.seed(attempt_seed)
            random.seed(attempt_seed)

            if model_name == "SirenNet":
                from SIGL.tools import SirenNet
                hid_dim_list = num_layers * [96]
                model = SirenNet(2, hid_dim_list, 1, num_layers=num_layers,
                                 w0=w0, w0_initial=w0_initial).train().to(device)
            elif model_name == "MLP":
                model = Net(hid_dim).train().to(device)
            elif model_name == "SiLUMLP":
                model = SiLUMLP(hid_dim, num_layers).train().to(device)
            else:
                layers = [2] + num_layers * [hid_dim] + [1]
                model = lipmlp(layers).train().to(device)

            if model_name == "SiLUMLP":
                losses, train_info = train_momentnet_sobol_degree(
                    model, induced_list[:num_motifs], real_moments,
                    4, N, epochs, attempt_lr, device, seed=attempt_seed,
                    eval_N=graphon9_eval_N, eval_interval=50, warmup_weight=2.0,
                    warmup_end=1200, min_var_weight=0.0, weight_mode=weight_mode)
                final_loss = train_info['best_eval_loss']
                print(f"Graphon {graphon_idx}, trial {trial}, attempt {attempt}: "
                      f"validation moment loss = {final_loss:.8f}, "
                      f"seed = {attempt_seed}, lr = {attempt_lr}", flush=True)
            elif use_sobol_zero_weights:
                losses, train_info = train_momentnet_sobol_degree(
                    model, induced_list[:num_motifs], real_moments,
                    4, N, epochs, attempt_lr, device, seed=attempt_seed,
                    eval_N=120000, eval_interval=50, warmup_weight=0.0,
                    warmup_end=1, min_var_weight=0.0, weight_mode=weight_mode,
                    weight_floor=zero_weight_floor, weight_cap=zero_weight_cap)
                final_loss = train_info['best_eval_loss']
                print(f"Graphon {graphon_idx}, trial {trial}, attempt {attempt}: "
                      f"zero-aware validation moment loss = {final_loss:.8f}, "
                      f"seed = {attempt_seed}, lr = {attempt_lr}", flush=True)
            else:
                losses = train_momentnet(model, induced_list[:num_motifs], real_moments,
                                         4, N, epochs, patience, attempt_lr, device, weight_mode)
                final_loss = losses[-1] if losses else float('inf')

            if final_loss < best_final_loss:
                best_final_loss = final_loss
                best_losses = losses
                best_model = model

            if (model_name != "SiLUMLP" and not use_sobol_zero_weights
                    and final_loss < convergence_threshold):
                break

        # Use the best of the attempts (handles cases where the threshold is
        # unreachable — e.g. very large weights — without spinning forever).
        model = best_model
        losses = best_losses

        # Compare centrality measures
        # move the model to CPU
        trained_inr = model.cpu()
        if evaluate_centrality:
            centrality_nmse = compare_centrality_measures(trained_inr, graphon_idx)
            centrality_results.append(centrality_nmse)

        # Replace error_trial computation with GW loss using comp_GW_loss
        error_trial = comp_GW_loss(trained_inr, true_graphon)

        errors[error_idx] = error_trial
        print(f"Graphon {graphon_idx}, trial {trial}: GW distance = {error_trial}", flush=True)

       

    # Remove NaN values
    errors = errors[~np.isnan(errors)]

    avg_gw = np.round(np.mean(errors), 4)
    std_gw = np.round(np.std(errors), 4)

    # Compute average and standard deviation for centrality NMSEs
    if evaluate_centrality:
        centrality_keys = centrality_results[0].keys()
        avg_centrality = {key: np.round(np.mean([res[key] for res in centrality_results]), 4) for key in centrality_keys}
        std_centrality = {key: np.round(np.std([res[key] for res in centrality_results]), 4) for key in centrality_keys}
    else:
        avg_centrality = {}
        std_centrality = {}

    print(f"Graphon {graphon_idx}: Avg GW distance = {avg_gw}, Std = {std_gw}")
    if evaluate_centrality:
        print(f"Centrality NMSE Averages: {avg_centrality}")
        print(f"Centrality NMSE Standard Deviations: {std_centrality}")

    return avg_gw, std_gw, avg_centrality, std_centrality, trained_inr
