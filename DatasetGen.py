import os
import pickle
from GraphTools.utils import simulate_graphs, general_graphon
import numpy as np

# Ensure dataset folder exists
os.makedirs('dataset', exist_ok=True)
os.makedirs('dataset/scalability', exist_ok=True)

# change this value to generate different datasets
# id = 1 : Synthetic dataset for MomentNet Evaluation 
# id = 2 : MomentNet Scalability Dataset Generator
# id = 3 : Graphon 12 $(0.5 + 0.1cos(\pi v)cos(\pi u))$
dataset_id = 2

if dataset_id == 1:
    # Generate graphs for each graphon
    for graphon_idx in range(13):
        if graphon_idx == 11:
            sbm_split = [0.5, 0.5]
            sbm_param = np.array([[0, 0.8], [0.8, 0]])
            graphon = general_graphon(graphon_idx, sbm_split, sbm_param)
        elif graphon_idx == 12:
            sbm_split = [0.5, 0.5]
            sbm_param = np.array([[0.8, 0], [0, 0.8]])
            graphon = general_graphon(graphon_idx, sbm_split, sbm_param)
        else:
            graphon = general_graphon(graphon_idx, None, None)

        # Generate 10 sets of 10 graphs
        all_graphs = []
        for set_idx in range(10):
            graphs = simulate_graphs(graphon, num_graphs=10, graph_size='vary')
            all_graphs.append(graphs)
            
        # Save all graphs for this graphon index to dataset folder
        file_path = f'dataset/graphon_{graphon_idx}.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(all_graphs, f)
elif dataset_id == 2:
    W = general_graphon(5, [0.5, 0.5], np.array([[0, 0.0], [0.0, 0]]))

    # Loop over node sizes from 50 to 1000 with step size 50
    for num_nodes in range(10, 1201, 20):
        # Generate 10 graphs for the current node size
        graphs = simulate_graphs(W, num_graphs=10, num_nodes=num_nodes, graph_size='fixed', seed_edge=num_nodes)
        
        # Save the entire list of graphs to one file
        filename = f"dataset/scalability/graphs_{num_nodes}_single.gpickle"
        with open(filename, "wb") as f:
            pickle.dump(graphs, f)
        print(f"Saved 10 graphs with {num_nodes} nodes in {filename}")
elif dataset_id == 3:
    W = general_graphon(12, [0.5, 0.5], np.array([[0, 0.0], [0.0, 0]]))

    # Loop over node sizes from 50 to 1000 with step size 50
    for num_nodes in range(10, 1201, 20):
        # Generate 10 graphs for the current node size
        graphs = simulate_graphs(W, num_graphs=10, num_nodes=num_nodes, graph_size='fixed', seed_edge=num_nodes)
        
        # Save the entire list of graphs to one file
        filename = f"dataset/scalability_12/graphs_{num_nodes}_single.gpickle"
        with open(filename, "wb") as f:
            pickle.dump(graphs, f)
        print(f"Saved 10 graphs with {num_nodes} nodes in {filename}")