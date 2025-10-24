import sys
import os 
sys.path.append('../')
import pandas as pd
from Moment.trainMoment import train_Moment
from GraphTools.utils import general_graphon, compare_centrality_measures_plot
import pickle

# Ensure result folder exists
os.makedirs('result', exist_ok=True)

# Graphon Idx (Based on the table 7 in the paper)
graphon_idx = 0
W = general_graphon(0, [], [])
dataset_name = f'graphon_{graphon_idx}.pkl'
results = train_Moment(dataset_name, graphon_idx)

model = results[-1]  # Get the trained model from the results
dataset_path = os.path.join('../dataset', dataset_name)
with open(dataset_path, 'rb') as f:
    graphs = pickle.load(f)
compare_centrality_measures_plot(model, graphs, 100, 0, alpha=0.85, beta=0.9) # this function is designed for graphon 0 and 1! you might need to change it for other graphons, or use compare_centrality_measures to get the error estimate only



