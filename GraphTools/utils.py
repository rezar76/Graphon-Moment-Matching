import numpy as np
import networkx as nx
from typing import List
import torch
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import quad
import math

def simulate_graphs(w: np.ndarray, seed_gsize: int=123, seed_edge: int=123, num_graphs: int = 10,
                           num_nodes: int = 200, graph_size: str = 'fixed', offset: int = 0) -> List[np.ndarray]:
    """
    Simulate graphs based on a graphon
    :param w: graphon function
    :param num_graphs: the number of simulated graphs
    :param num_nodes: the number of nodes per graph
    :param graph_size: fix each graph size as num_nodes or sample the size randomly as num_nodes * (0.5 + uniform)
    :return:
        graphs: a list of binary adjacency matrices
    """
    graphs = []

    if graph_size == 'vary':
        numbers = np.linspace(50 + offset, 300 + offset, num_graphs).astype(int).tolist()
    else:  # fixed size
        numbers = [num_nodes for _ in range(num_graphs)]

    np.random.seed(seed_edge)  # Add random seed for reproducibility
    for n in range(num_graphs):
        node_locs = np.random.rand(numbers[n])
        adj_graph = np.zeros((numbers[n], numbers[n]))
        for i in range(numbers[n]):
            for j in range(i + 1, numbers[n]):
                adj_graph[i, j] = w(node_locs[i], node_locs[j])
                adj_graph[j, i] = adj_graph[i, j]
        noise = np.random.rand(adj_graph.shape[0], adj_graph.shape[1])
        adj_graph -= noise
        np.fill_diagonal(adj_graph, 0)
        sampled_graph = (adj_graph > 0).astype('float')
        sampled_graph = np.triu(sampled_graph) + np.triu(sampled_graph, 1).T
        G = nx.from_numpy_array(sampled_graph)
        graphs.append(G)

    return graphs

def simulate_graphs_sorted(w: np.ndarray, seed_gsize: int=123, seed_edge: int=123, num_graphs: int = 10,
                           num_nodes: int = 200, graph_size: str = 'fixed', offset: int = 0) -> List[np.ndarray]:
    """
    Simulate graphs based on a graphon
    :param w: graphon function
    :param num_graphs: the number of simulated graphs
    :param num_nodes: the number of nodes per graph
    :param graph_size: fix each graph size as num_nodes or sample the size randomly as num_nodes * (0.5 + uniform)
    :return:
        graphs: a list of binary adjacency matrices
    """
    graphs = []

    if graph_size == 'vary':
        numbers = np.linspace(50 + offset, 300 + offset, num_graphs).astype(int).tolist()
    else:  # fixed size
        numbers = [num_nodes for _ in range(num_graphs)]

    np.random.seed(seed_edge)  # Add random seed for reproducibility
    for n in range(num_graphs):
        node_locs = np.sort(np.random.rand(numbers[n]))
        adj_graph = np.zeros((numbers[n], numbers[n]))
        for i in range(numbers[n]):
            for j in range(i + 1, numbers[n]):
                adj_graph[i, j] = w(node_locs[i], node_locs[j])
                adj_graph[j, i] = adj_graph[i, j]
        noise = np.random.rand(adj_graph.shape[0], adj_graph.shape[1])
        adj_graph -= noise
        np.fill_diagonal(adj_graph, 0)
        sampled_graph = (adj_graph > 0).astype('float')
        sampled_graph = np.triu(sampled_graph) + np.triu(sampled_graph, 1).T
        G = nx.from_numpy_array(sampled_graph)
        graphs.append(G)

    return graphs

def general_graphon(model_type, sbm_split, sbm_param):


  if model_type == 0:
    return lambda x,y : x*y
  elif model_type == 1:
    return lambda x,y : np.exp(-(x**(0.7) + y**(0.7)))
  elif model_type == 2:
    return lambda x,y : 0.25*(x**2 + y**2 + np.sqrt(x) + np.sqrt(y))
  elif model_type == 3:
    return lambda x,y : 0.5*(x+y)
  elif model_type == 4:
    return lambda x,y : (1 + np.exp(-2*(x**2+y**2)))**(-1)
  elif model_type == 5:
    return lambda x,y : (1 + np.exp(-np.maximum(x,y)**2 - np.minimum(x,y)**4))**(-1)
  elif model_type == 6:
    return lambda x,y : np.exp(-np.maximum(x,y)**(0.75))
  elif model_type == 7:
    return lambda x,y : np.exp(-0.5*(np.minimum(x,y)+np.sqrt(x)+np.sqrt(y)))
  elif model_type == 8:
    return lambda x,y : np.log(1+np.maximum(x,y))
  elif model_type == 9:
    return lambda x,y : np.abs(x-y)
  elif model_type == 10:
    return lambda x,y : 1-np.abs(x-y)
  elif model_type == 12:
      return lambda u, v: 0.5 + 0.1 * math.cos(math.pi * u) * math.cos(math.pi * v)
  elif model_type == 13:
      return lambda x, y: 0.2 + 0.6 * x * y
  elif model_type == 11:
    def sbm_graphon(x, y):
      i = (x >= sbm_split[0]).astype(int)
      j = (y >= sbm_split[1]).astype(int)
      return sbm_param[i,j]
    return sbm_graphon
  



def normalized_rmse(y_true, y_pred, norm_type='range'):
    """
    Compute the Normalized Root Mean Square Error (NRMSE) between predictions and true values.

    Parameters:
    -----------
    y_true : array-like
        Array of true values.
    y_pred : array-like
        Array of predicted values.
    norm_type : str, optional (default='range')
        The normalization method. Options:
         - 'range': Normalize by the range (max - min) of y_true.
         - 'mean': Normalize by the mean of y_true.
         - 'std': Normalize by the standard deviation of y_true.

    Returns:
    --------
    float
        The normalized RMSE value.

    Raises:
    -------
    ValueError:
        If an unknown normalization type is provided.
    """
    # Convert inputs to numpy arrays for consistency
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the RMSE
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # Select the normalization factor based on user choice
    if norm_type == 'range':
        norm_factor = np.max(y_true) - np.min(y_true)
        if norm_factor == 0:
            raise ValueError("Normalization by range is not possible because max and min of y_true are equal.")
    elif norm_type == 'mean':
        norm_factor = np.mean(y_true)
        if norm_factor == 0:
            raise ValueError("Normalization by mean is not possible because the mean of y_true is zero.")
    elif norm_type == 'std':
        norm_factor = np.std(y_true)
        if norm_factor == 0:
            raise ValueError("Normalization by standard deviation is not possible because the std of y_true is zero.")
    else:
        raise ValueError("Unknown norm_type. Use 'range', 'mean', or 'std'.")
    
    return rmse / norm_factor

def get_grid(resolution):
    """
    Create a uniform grid in [0, 1] with the given resolution.
    
    Returns:
        x: numpy.ndarray, grid points
        dx: float, spacing between points
    """
    x = np.linspace(0, 1, resolution)
    dx = x[1] - x[0] if resolution > 1 else 1.0
    return x, dx

def get_kernel_matrix(W, x, symmetrize=False, torch_dtype=torch.float32):
    """
    Build the kernel matrix for the graphon function W evaluated at grid points x.
    
    Parameters:
        W : callable or torch.nn.Module
            Graphon function or a neural network that takes a pair (x,y) as input.
        x : numpy.ndarray
            1D grid array.
        symmetrize : bool
            If True, symmetrize the resulting matrix (recommended for torch modules).
        torch_dtype : torch.dtype
            Data type to use when converting inputs (only applies if W is a torch module).
    
    Returns:
        W_matrix : numpy.ndarray
            The kernel matrix.
    """
    resolution = len(x)
    if isinstance(W, torch.nn.Module):
        xx, yy = np.meshgrid(x, x)
        inputs = torch.tensor(np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch_dtype)
        W_matrix = W(inputs).detach().numpy().reshape(resolution, resolution)
        if symmetrize:
            i_upper = np.triu_indices(resolution, 1)
            W_matrix[i_upper] = W_matrix.T[i_upper]
            np.fill_diagonal(W_matrix, 0)
    else:
        W_matrix = np.array([[W(xi, xj) for xj in x] for xi in x])
    return W_matrix

def compute_degree_centrality_graphon(W, resolution=1000):
    """
    Compute the degree centrality for each node in [0,1] for a graphon.
    
    This function first generates the full graphon matrix on a uniform grid 
    of (resolution x resolution) over [0,1] x [0,1] and then computes the 
    degree centrality for each node by averaging over the corresponding row.
    
    Parameters:
        W : callable or torch.nn.Module
            Graphon function or neural network which takes two inputs (x and y).
        resolution : int, optional
            Number of evaluation points along each dimension (default is 1000).
            
    Returns:
        degree_centrality : numpy.ndarray
            An array of shape (resolution,) containing the degree centrality 
            for each node (grid point in [0,1]).
    """
    x, dx = get_grid(resolution)
    symmetrize = isinstance(W, torch.nn.Module)
    W_matrix = get_kernel_matrix(W, x, symmetrize=symmetrize)
    degree_centrality = np.mean(W_matrix, axis=1)
    return degree_centrality

def compute_eigenvector_centrality_graphon(W, resolution=50):
    """
    Compute the top eigenfunction (eigenvector centrality) of the integral operator defined by W.
    
    Parameters:
        W : callable or torch.nn.Module
            Graphon function or neural network.
        resolution : int
            Number of grid points.
    
    Returns:
        x : numpy.ndarray
            Grid points.
        phi : numpy.ndarray
            Normalized top eigenfunction.
    """
    x, dx = get_grid(resolution)
    symmetrize = isinstance(W, torch.nn.Module)
    W_matrix = get_kernel_matrix(W, x, symmetrize=symmetrize)
    
    # Apply quadrature weight to simulate the integral operator
    W_op = W_matrix * dx
    eigenvalues, eigenvectors = eigh(W_op)
    
    # Select the top eigenfunction and ensure consistent sign
    phi = eigenvectors[:, -1]
    if np.sum(phi) < 0:
        phi = -phi
    phi = phi / np.sqrt(np.sum(phi**2) * dx)
    return x, phi

def compute_katz_centrality_graphon(W, alpha=None, resolution=50):
    """
    Compute the Katz centrality for the graphon using an operator inversion method.
    
    Parameters:
        W : callable or torch.nn.Module
            Graphon function or neural network.
        alpha : float, optional
            Damping factor. If None, set to 0.85 divided by the operator norm.
        resolution : int
            Number of grid points.
    
    Returns:
        x : numpy.ndarray
            Grid points.
        c_katz : numpy.ndarray
            Katz centrality vector.
        op_norm : float
            Operator norm of the graphon.
    """
    x, dx = get_grid(resolution)
    symmetrize = isinstance(W, torch.nn.Module)
    W_matrix = get_kernel_matrix(W, x, symmetrize=symmetrize)
    op_norm = np.linalg.norm(W_matrix, ord=2)
    
    if alpha is None:
        alpha = 0.85 / op_norm
    elif alpha >= 1 / op_norm:
        raise ValueError(f"alpha must be less than 1/||W|| = {1/op_norm}")
    
    M_alpha = np.eye(resolution) - alpha * W_matrix * dx
    c_katz = np.linalg.solve(M_alpha, np.ones(resolution))
    c_katz = c_katz / np.linalg.norm(c_katz)
    return x, c_katz, op_norm

import numpy as np
import torch


def compute_pagerank_centrality_graphon(W, beta=0.85, resolution=50):
    """
    Compute the PageRank centrality for a general graphon W(x, y).

    Parameters:
        W : callable or torch.nn.Module
            Graphon function or neural network.
        beta : float
            Damping factor.
        resolution : int
            Number of grid points.

    Returns:
        x : numpy.ndarray
            Grid points.
        c_pr : numpy.ndarray
            PageRank centrality vector.
    """
    x, dx = get_grid(resolution)
    symmetrize = isinstance(W, torch.nn.Module)
    W_matrix = get_kernel_matrix(W, x, symmetrize=symmetrize)

    # Compute degree and pseudo-inverse
    c_d = np.sum(W_matrix, axis=1) * dx  # integral approximation
    c_d_inv = np.where(c_d != 0, 1 / c_d, 0)

    # Construct L_beta = I - beta * W @ diag(c_d_inv) * dx
    D_inv = np.diag(c_d_inv)
    L_beta = np.eye(resolution) - beta * W_matrix @ D_inv * dx

    # Solve the system: (I - beta W D^dagger dx) f = 1
    rhs = np.ones(resolution)
    c_pr = (1 - beta) * np.linalg.solve(L_beta, rhs)

    return x, c_pr

import matplotlib.gridspec as gridspec

def compare_centrality_measures_plot(trained_model, input_graphs, graph_size, graphon_idx, alpha=None, beta=0.85):
    """
    Compare degree, eigenvector, Katz, and PageRank centralities between a trained graphon
    and a list of input graphs. Plots the results for visual comparison and displays the 
    analytic (LaTeX) mathematical expressions and graphon type in a dedicated area below the plots.
    
    Parameters:
        trained_model : callable or torch.nn.Module
            The graphon model (or function).
        input_graphs : list of networkx.Graph
            Graphs on which to compute traditional centrality measures.
        graph_size : int
            Number of nodes in each input graph.
        graphon_idx : int
            Index specifying the type of graphon. For each index, a dictionary with the analytic 
            centrality functions should be available. Currently, index 0 corresponds to an xy graphon.
        alpha : float, optional
            Damping factor for Katz centrality (overridden in this routine based on op norm).
        beta : float
            Damping factor for PageRank centrality.
    
    Returns:
        alpha : float
            The computed alpha value used for Katz centrality.
    """
    # Switch model to eval if it is a torch module.
    if isinstance(trained_model, torch.nn.Module):
        trained_model.eval()

    resolution =  graph_size
    x, _ = get_grid(resolution)

    # Define analytic functions and LaTeX strings.
    analytic_funcs = {}
    if graphon_idx == 0:
        analytic_funcs = {
            'degree': lambda x: x / 2,
            'eigenvector': lambda x: (np.sqrt(3) * x) / (np.linalg.norm(np.sqrt(3) * x)
                                                        if np.linalg.norm(np.sqrt(3) * x) != 0 else 1),
            'katz': lambda x, alpha: (1 + (3 * alpha * x) / (6 - 2 * alpha)) /
                        (np.linalg.norm(1 + (3 * alpha * x) / (6 - 2 * alpha))
                         if np.linalg.norm(1 + (3 * alpha * x) / (6 - 2 * alpha)) != 0 else 1),
            'pagerank': lambda x, beta: ((1 - beta) + 2 * beta * x) /
                        (np.linalg.norm((1 - beta) + 2 * beta * x)
                         if np.linalg.norm((1 - beta) + 2 * beta * x) != 0 else 1)
        }
        latex_degree = r'$C_{deg}(x)=\frac{x}{2}$'
        latex_eigen  = r'$C_{eig}(x)=\frac{\sqrt{3}\,x}{\|\sqrt{3}\,x\|}$'
        latex_katz   = r'$C_{katz}(x)=\frac{1+\frac{3\alpha x}{6-2\alpha}}{\left\|1+\frac{3\alpha x}{6-2\alpha}\right\|}$'
        latex_pr     = r'$C_{pr}(x)=\frac{(1-\beta)+2\beta x}{\left\|(1-\beta)+2\beta x\right\|}$'
    elif graphon_idx == 1:
        analytic_funcs['degree'] = lambda x: quad(lambda y: np.exp(-y**0.7), 0, 1)[0] * np.exp(-x**0.7)
        analytic_funcs['eigenvector'] = lambda x: np.exp(-x**0.7) / np.sqrt(quad(lambda y: np.exp(-2*y**0.7), 0, 1)[0])
        analytic_funcs['katz'] = lambda x, alpha: 1 + (alpha * quad(lambda y: np.exp(-y**0.7), 0, 1)[0] * np.exp(-x**0.7)) / (
            1 - alpha * quad(lambda y: np.exp(-2*y**0.7), 0, 1)[0])
        analytic_funcs['pagerank'] = lambda x, beta: (1 - beta) + beta / quad(lambda z: np.exp(-z**0.7), 0, 1)[0] * np.exp(-x**0.7)
        latex_degree = r'$C_{deg}(x) = 0.7492\,e^{-x^{0.7}}$'
        latex_eigen = r'$C_{eig}(x) = \frac{e^{-x^{0.7}}}{\sqrt{0.473}}$'
        latex_katz = r'$C_{katz}(x,\alpha) = 1 + \frac{0.7492\,\alpha\,e^{-x^{0.7}}}{1-0.473\,\alpha}$'
        latex_pr = r"$C_{pr}(x,\beta) = (1-\beta) + \frac{\beta}{0.7492}\,e^{-x^{0.7}}$"
    else:
        latex_degree = latex_eigen = latex_katz = latex_pr = ""

    # --- Compute predicted graphon centralities (without reordering) ---
    c_d_graphon = np.array(compute_degree_centrality_graphon(trained_model, resolution))
    
    x_eig, c_e_graphon = compute_eigenvector_centrality_graphon(trained_model, resolution)
    c_e_graphon = c_e_graphon / np.linalg.norm(c_e_graphon)
    
    symmetrize = isinstance(trained_model, torch.nn.Module)
    W_matrix = get_kernel_matrix(trained_model, x, symmetrize=symmetrize)
    op_norm = np.linalg.norm(W_matrix, ord=2)
    print("Operator norm of graphon:", op_norm)
    alpha = 0.85 / op_norm  # override any provided alpha
    
    x_katz, c_k_graphon, _ = compute_katz_centrality_graphon(trained_model, alpha=None, resolution=resolution)
    x_pr, c_pr_graphon = compute_pagerank_centrality_graphon(trained_model, beta, resolution=resolution)
    c_pr_graphon = c_pr_graphon / np.linalg.norm(c_pr_graphon)
    
    # --- Compute analytic centrality values only once ---
    if graphon_idx in [0, 1]:
        c_d_analytic = analytic_funcs['degree'](x)
        c_e_analytic = analytic_funcs['eigenvector'](x)
        # Normalize eigenvector analytic centrality.
        norm_e = np.linalg.norm(c_e_analytic)
        if norm_e != 0:
            c_e_analytic = c_e_analytic / norm_e
        
        c_k_analytic = analytic_funcs['katz'](x, alpha)
        # For graphon_idx==1, the analytic Katz is not normalized inside the lambda.
        if graphon_idx == 1:
            norm_k = np.linalg.norm(c_k_analytic)
            if norm_k != 0:
                c_k_analytic = c_k_analytic / norm_k
        
        c_pr_analytic = analytic_funcs['pagerank'](x, beta)
        norm_pr = np.linalg.norm(c_pr_analytic)
        if norm_pr != 0:
            c_pr_analytic = c_pr_analytic / norm_pr

        # --- Flip predicted graphon centralities based on Pearson correlation ---
        if np.corrcoef(c_d_graphon, c_d_analytic)[0, 1] < 0:
            c_d_graphon = c_d_graphon[::-1]
        if np.corrcoef(c_e_graphon, c_e_analytic)[0, 1] < 0:
            c_e_graphon = c_e_graphon[::-1]
        if np.corrcoef(c_k_graphon, c_k_analytic)[0, 1] < 0:
            c_k_graphon = c_k_graphon[::-1]
        if np.corrcoef(c_pr_graphon, c_pr_analytic)[0, 1] < 0:
            c_pr_graphon = c_pr_graphon[::-1]
    else:
        c_d_analytic = c_e_analytic = c_k_analytic = c_pr_analytic = None

    # Define graphon type label.
    if graphon_idx == 0:
        graphon_type = "xy"
    elif graphon_idx == 1:
        graphon_type = "exp^(-(x^0.7+y^0.7))"
    else:
        graphon_type = "Unknown"

    # --- Create figure using GridSpec ---
    fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.3])
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{mathpazo}"
    })
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[2, :])

    # Plot graphon centrality curves (blue lines).
    ax1.plot(x, c_d_graphon, 'b-', label='MomentNet Degree')
    ax2.plot(x_eig, c_e_graphon, 'b-', label='MomentNet Eigenvector')
    ax3.plot(x_katz, c_k_graphon, 'b-', label='MomentNet Katz')
    ax4.plot(x_pr, c_pr_graphon, 'b-', label='MomentNet PageRank')




    # Overlay input graphs' centrality estimates (red dots).
    done = False
    for graph in input_graphs:
        # Degree centrality from networkx.
        c_d = list(nx.degree_centrality(graph).values())
        
        # Eigenvector centrality.
        adj_matrix = nx.to_numpy_array(graph)
        eigvals, eigvecs = np.linalg.eig(adj_matrix)
        idx = np.argmax(eigvals.real)
        v1 = np.abs(eigvecs[:, idx].real)
        c_e = (v1 / np.linalg.norm(v1)).tolist()
        
        # Katz centrality.
        try:
            lambda_max = np.max(np.abs(np.linalg.eigvals(adj_matrix)))
            alpha_graph = (0.85 / lambda_max) / graph_size
            c_k = list(nx.katz_centrality_numpy(graph, alpha=alpha_graph, beta=1.0).values())
            c_k = (np.array(c_k) / np.linalg.norm(c_k)).tolist()
        except Exception as e:
            print("Warning: Katz centrality computation failed for a graph:", e)
            c_k = None
        
        # PageRank centrality.
        c_pr = list(nx.pagerank(graph, alpha=beta).values())
        c_pr = np.array(c_pr) / np.linalg.norm(c_pr)
        x_graph = np.linspace(0, 1, len(c_d))
        
        if not done:
            ax1.plot(x_graph, c_d, 'r.', alpha=0.1, label='Graph Degree')
            ax2.plot(x_graph, c_e, 'r.', alpha=0.1, label='Graph Eigenvector')
            if c_k is not None:
                ax3.plot(x_graph, c_k, 'r.', alpha=0.1, label='Graph Katz')
            ax4.plot(x_graph, c_pr, 'r.', alpha=0.1, label='Graph PageRank')
            done = True
        else:
            ax1.plot(x_graph, c_d, 'r.', alpha=0.1)
            ax2.plot(x_graph, c_e, 'r.', alpha=0.1)
            if c_k is not None:
                ax3.plot(x_graph, c_k, 'r.', alpha=0.1)
            ax4.plot(x_graph, c_pr, 'r.', alpha=0.1)

    # Plot analytic curves (dashed black lines), if available.
    if c_d_analytic is not None:
        ax1.plot(x, c_d_analytic, 'k--', label='Analytic Degree')
    if c_e_analytic is not None:
        ax2.plot(x, c_e_analytic, 'k--', label='Analytic Eigenvector')
    if c_k_analytic is not None:
        ax3.plot(x, c_k_analytic, 'k--', label='Analytic Katz')
    if c_pr_analytic is not None:
        ax4.plot(x, c_pr_analytic, 'k--', label='Analytic PageRank')

    # Compute MSE losses using a normalized RMSE function.
    if c_d_analytic is not None:
        mse_deg = normalized_rmse(c_d_analytic, c_d_graphon, norm_type='range')
        mse_eig = normalized_rmse(c_e_analytic, c_e_graphon, norm_type='range')
        mse_katz = normalized_rmse(c_k_analytic, c_k_graphon, norm_type='range')
        mse_pr = normalized_rmse(c_pr_analytic, c_pr_graphon, norm_type='range')
    else:
        mse_deg = mse_eig = mse_katz = mse_pr = np.nan

    # Set titles, labels, legends and grid.
    ax1.set_title('Degree Centrality', fontsize=18) # Adjust 18 to your desired size
    ax2.set_title('Eigenvector Centrality', fontsize=18)
    ax3.set_title(f'Katz Centrality ($\\alpha$ = {alpha * op_norm:.4f})', fontsize=18)
    ax4.set_title(f'PageRank Centrality ($\\beta$ = {beta})', fontsize=18)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Latent Variable', fontsize=15) # Adjust 18 to your desired size
        # Set ylabel with specific fontsize
        ax.set_ylabel('Centrality Value', fontsize=15) #
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.legend(fontsize=14)
        ax.grid(True)
    
    # Annotation with analytic formulas and MSE information.
    ax_text.axis("off")
    analytical_info = (
        f"Graphon type: {graphon_type}\n"
        f"Analytic centrality formulas:\n"
        f"Degree: {latex_degree}\n"
        f"Eigenvector: {latex_eigen}\n"
        f"Katz: {latex_katz}\n"
        f"PageRank: {latex_pr}"
    )
    mse_info = (
         f"\n\nMSE between Analytic and Graphon Estimates:\n"
         f"Degree: {mse_deg:.4e}\n"
         f"Eigenvector: {mse_eig:.4e}\n"
         f"Katz: {mse_katz:.4e}\n"
         f"PageRank: {mse_pr:.4e}"
    )

    plt.tight_layout()
    # save the figure
    fig.savefig(f'results/centrality_comparison_graphon_{graphon_idx}.png', dpi=300)
    plt.show()
    
    # Switch model back to train mode if applicable.
    if isinstance(trained_model, torch.nn.Module):
        trained_model.train()
    
    return alpha


def compare_centrality_measures(trained_model, graphon_idx, alpha=None, beta=0.85):
    """
    Compare degree, eigenvector, Katz, and PageRank centralities between a trained graphon
    and the analytical graphon. Returns the NMSE between the graphon centrality and analytical results.

    Parameters:
        trained_model : callable or torch.nn.Module
            The graphon model (or function).
        graphon_idx : int
            Index specifying the type of graphon.
        alpha : float, optional
            Damping factor for Katz centrality (overridden in this routine based on op norm).
        beta : float
            Damping factor for PageRank centrality.

    Returns:
        dict: NMSE values for each centrality measure.
    """
    # Switch model to eval if it is a torch module.
    if isinstance(trained_model, torch.nn.Module):
        trained_model.eval()

    resolution = 1000
    x, _ = get_grid(resolution)

    # Generate the true graphon
    if graphon_idx == 11:
        sbm_split = np.array([0.5, 0.5])
        sbm_param = np.array([[0, 0.8], [0.8, 0]])
        true_graphon = general_graphon(graphon_idx, sbm_split, sbm_param)
    else:
        true_graphon = general_graphon(graphon_idx, None, None)

    # --- Compute predicted graphon centralities ---
    c_d_graphon = np.array(compute_degree_centrality_graphon(trained_model, resolution))
    x_eig, c_e_graphon = compute_eigenvector_centrality_graphon(trained_model, resolution)
    c_e_graphon = c_e_graphon / np.linalg.norm(c_e_graphon)

    symmetrize = isinstance(trained_model, torch.nn.Module)
    W_matrix = get_kernel_matrix(trained_model, x, symmetrize=symmetrize)
    op_norm = np.linalg.norm(W_matrix, ord=2)
    alpha = 0.85 / op_norm  # override any provided alpha

    x_katz, c_k_graphon, _ = compute_katz_centrality_graphon(trained_model, alpha=None, resolution=resolution)
    x_pr, c_pr_graphon = compute_pagerank_centrality_graphon(trained_model, beta, resolution=resolution)
    c_pr_graphon = c_pr_graphon / np.linalg.norm(c_pr_graphon)

    # --- Compute analytical centralities ---
    c_d_analytic = np.array(compute_degree_centrality_graphon(true_graphon, resolution))
    x_eig, c_e_analytic = compute_eigenvector_centrality_graphon(true_graphon, resolution)
    c_e_analytic = c_e_analytic / np.linalg.norm(c_e_analytic)

    x_katz, c_k_analytic, _ = compute_katz_centrality_graphon(true_graphon, alpha=None, resolution=resolution)
    x_pr, c_pr_analytic = compute_pagerank_centrality_graphon(true_graphon, beta, resolution=resolution)
    c_pr_analytic = c_pr_analytic / np.linalg.norm(c_pr_analytic)

    # --- Flip predicted graphon centralities based on Pearson correlation ---
    if np.corrcoef(c_d_graphon, c_d_analytic)[0, 1] < 0:
        c_d_graphon = c_d_graphon[::-1]
    if np.corrcoef(c_e_graphon, c_e_analytic)[0, 1] < 0:
        c_e_graphon = c_e_graphon[::-1]
    if np.corrcoef(c_k_graphon, c_k_analytic)[0, 1] < 0:
        c_k_graphon = c_k_graphon[::-1]
    if np.corrcoef(c_pr_graphon, c_pr_analytic)[0, 1] < 0:
        c_pr_graphon = c_pr_graphon[::-1]

    # Compute NMSE values
    if graphon_idx == 11:
        nmse_deg = normalized_rmse(c_d_analytic, c_d_graphon, norm_type='max')
        nmse_eig = normalized_rmse(c_e_analytic, c_e_graphon, norm_type='max')
        nmse_katz = normalized_rmse(c_k_analytic, c_k_graphon, norm_type='max')
        nmse_pr = normalized_rmse(c_pr_analytic, c_pr_graphon, norm_type='max')
    else:
        nmse_deg = normalized_rmse(c_d_analytic, c_d_graphon, norm_type='range')
        nmse_eig = normalized_rmse(c_e_analytic, c_e_graphon, norm_type='range')
        nmse_katz = normalized_rmse(c_k_analytic, c_k_graphon, norm_type='range')
        nmse_pr = normalized_rmse(c_pr_analytic, c_pr_graphon, norm_type='range')

    nmse_results = {
        'degree': nmse_deg,
        'eigenvector': nmse_eig,
        'katz': nmse_katz,
        'pagerank': nmse_pr
    }

    # Switch model back to train mode if applicable.
    if isinstance(trained_model, torch.nn.Module):
        trained_model.train()

    return nmse_results