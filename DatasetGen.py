import argparse
import os
import pickle

import numpy as np

from GraphTools.utils import general_graphon, simulate_graphs


def parse_graphons(value):
    if value.lower() == 'all':
        return list(range(13))
    graphons = []
    for item in value.replace(',', ' ').split():
        graphons.append(int(item))
    return sorted(dict.fromkeys(graphons))


def generate_graphon_benchmark(graphons, num_sets, num_graphs):
    os.makedirs('dataset', exist_ok=True)
    for graphon_idx in graphons:
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

        all_graphs = []
        for set_idx in range(num_sets):
            graphs = simulate_graphs(graphon, num_graphs=num_graphs, graph_size='vary',
                                     seed_edge=123 + set_idx)
            all_graphs.append(graphs)

        file_path = f'dataset/graphon_{graphon_idx}.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(all_graphs, f)
        print(f'Saved {file_path}', flush=True)


def generate_scalability(graphon_idx, output_dir, min_nodes, max_nodes, step, num_graphs):
    os.makedirs(output_dir, exist_ok=True)
    graphon = general_graphon(
        graphon_idx, [0.5, 0.5], np.array([[0, 0.0], [0.0, 0]]))

    for num_nodes in range(min_nodes, max_nodes + 1, step):
        graphs = simulate_graphs(
            graphon, num_graphs=num_graphs, num_nodes=num_nodes,
            graph_size='fixed', seed_edge=num_nodes)
        filename = f'{output_dir}/graphs_{num_nodes}_single.gpickle'
        with open(filename, "wb") as f:
            pickle.dump(graphs, f)
        print(f'Saved {num_graphs} graphs with {num_nodes} nodes in {filename}',
              flush=True)


def main():
    parser = argparse.ArgumentParser(description='Regenerate MomentNet datasets.')
    parser.add_argument('--dataset-id', type=int, default=1, choices=[1, 2, 3],
                        help=('1: graphon benchmark, 2: scalability benchmark, '
                              '3: graphon 12 scalability benchmark.'))
    parser.add_argument('--graphons', default='all',
                        help='Graphons for dataset-id 1, e.g. "0,1,9" or "all".')
    parser.add_argument('--num-sets', type=int, default=10)
    parser.add_argument('--num-graphs', type=int, default=10)
    parser.add_argument('--min-nodes', type=int, default=10)
    parser.add_argument('--max-nodes', type=int, default=1200)
    parser.add_argument('--step', type=int, default=20)
    args = parser.parse_args()

    if args.dataset_id == 1:
        generate_graphon_benchmark(
            parse_graphons(args.graphons), args.num_sets, args.num_graphs)
    elif args.dataset_id == 2:
        generate_scalability(
            5, 'dataset/scalability', args.min_nodes, args.max_nodes,
            args.step, args.num_graphs)
    elif args.dataset_id == 3:
        generate_scalability(
            12, 'dataset/scalability_12', args.min_nodes, args.max_nodes,
            args.step, args.num_graphs)


if __name__ == '__main__':
    main()
