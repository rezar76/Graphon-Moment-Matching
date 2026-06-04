import argparse
import os
import pickle
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

GRAPHON9_ES = [[[(0, 1)]],
               [[(0, 1), (1, 2)],
                [(0, 1), (0, 2), (1, 2)]],
               [[(0, 1), (1, 2), (2, 3)],
                [(0, 1), (0, 2), (0, 3)],
                [(0, 1), (0, 2), (1, 3), (2, 3)],
                [(0, 1), (0, 2), (0, 3), (1, 2)],
                [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)],
                [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]]]


def discover_graphons(dataset_dir):
    indices = []
    for path in dataset_dir.glob('graphon_*.pkl'):
        try:
            indices.append(int(path.stem.split('_')[1]))
        except (IndexError, ValueError):
            continue
    return sorted(indices)


def parse_graphons(value, available):
    if value.lower() == 'all':
        return available
    requested = [int(item) for item in value.replace(',', ' ').split()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f'Missing dataset files for graphons: {missing}')
    return requested


def parse_gpus(value):
    return [item.strip() for item in value.replace(',', ' ').split()
            if item.strip()]


def auto_gpus():
    try:
        out = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=index,name,memory.total',
            '--format=csv,noheader,nounits',
        ], text=True)
    except (OSError, subprocess.CalledProcessError):
        return '0'

    ids = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 3:
            continue
        idx, name, total_mem = parts[0], parts[1], parts[2]
        try:
            mem_mb = int(total_mem)
        except ValueError:
            mem_mb = 0
        if 'A100' in name or mem_mb >= 20000:
            ids.append(idx)
    return ','.join(ids) if ids else '0'


def dataset_trial_count(graphon_idx):
    dataset_path = REPO_ROOT / 'dataset' / f'graphon_{graphon_idx}.pkl'
    with open(dataset_path, 'rb') as f:
        return len(pickle.load(f))


def write_results(rows, path):
    if rows:
        pd.DataFrame(rows).sort_values('graphon_idx').to_csv(path, index=False)


def train_selected_trials(task):
    gpu = task['gpu']
    if gpu.lower() != 'cpu':
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    from Moment.trainMoment import train_Moment

    graphon_idx = task['graphon_idx']
    trial = task['trial']
    start = time.time()
    avg_gw, std_gw, avg_cent, std_cent, _ = train_Moment(
        f'graphon_{graphon_idx}.pkl', graphon_idx,
        evaluate_centrality=task['centrality'], trial_indices=[trial])
    row = {
        'graphon_idx': graphon_idx,
        'trial': trial,
        'gpu': gpu,
        'avg_gw': float(avg_gw),
        'std_gw': float(std_gw),
        'elapsed_sec': round(time.time() - start, 2),
    }
    row.update({f'avg_{key}': value for key, value in avg_cent.items()})
    row.update({f'std_{key}': value for key, value in std_cent.items()})
    return row


def graphon9_targets(trials):
    from Moment.tools import aggregate_moments, motifs_to_induced_motifs

    motifs = motifs_to_induced_motifs(GRAPHON9_ES)
    with open(REPO_ROOT / 'dataset' / 'graphon_9.pkl', 'rb') as f:
        graph_trials = pickle.load(f)
    return {
        trial: aggregate_moments(graph_trials[trial])[:len(motifs)]
        for trial in trials
    }


def train_graphon9_attempt(task):
    gpu = task['gpu']
    if gpu.lower() != 'cpu':
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    import torch
    from GraphTools.utils import general_graphon
    from Moment.tools import (SiLUMLP, comp_GW_loss, motifs_to_induced_motifs,
                              train_momentnet_sobol_degree)

    seed = int(task['seed'])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    motifs = motifs_to_induced_motifs(GRAPHON9_ES)
    target = torch.tensor(task['target_np'], dtype=torch.float32, device=device)

    model = SiLUMLP(192, 3).train().to(device)
    start = time.time()
    _, train_info = train_momentnet_sobol_degree(
        model, motifs, target, 4, 90000, 3200, task['lr'], device,
        seed=seed, eval_N=180000, eval_interval=50, warmup_weight=2.0,
        warmup_end=1200, min_var_weight=0.0, weight_mode=0)

    trained = model.cpu()
    gw = comp_GW_loss(trained, general_graphon(9, None, None))
    row = {
        'trial': task['trial'],
        'attempt': task['attempt'],
        'seed': seed,
        'lr': task['lr'],
        'gpu': gpu,
        'best_eval': train_info['best_eval_loss'],
        'best_epoch': train_info['best_epoch'],
        'gw': gw,
        'elapsed_sec': round(time.time() - start, 2),
    }

    if task['centrality']:
        from GraphTools.utils import compare_centrality_measures
        centrality = compare_centrality_measures(trained, 9)
        row.update({f'avg_{key}': value for key, value in centrality.items()})

    return row


def aggregate_trial_rows(graphon_idx, trial_rows, elapsed):
    df = pd.DataFrame(trial_rows).sort_values('trial')
    row = {
        'graphon_idx': graphon_idx,
        'avg_gw': round(float(df['avg_gw'].mean()), 4),
        'std_gw': round(float(df['avg_gw'].std(ddof=0)), 4),
        'elapsed_sec': round(elapsed, 2),
    }

    for prefix in ('avg_', 'std_'):
        for col in [c for c in df.columns if c.startswith(prefix)]:
            if col in ('avg_gw', 'std_gw'):
                continue
            values = df[col].dropna()
            if len(values) == 0:
                continue
            if prefix == 'avg_':
                row[col] = round(float(values.mean()), 4)
            else:
                base = col.removeprefix('std_')
                avg_col = f'avg_{base}'
                if avg_col in df:
                    row[col] = round(float(df[avg_col].std(ddof=0)), 4)
                else:
                    row[col] = round(float(values.mean()), 4)
    return row


def run_graphon_parallel(graphon_idx, args):
    trials = list(range(dataset_trial_count(graphon_idx)))
    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise ValueError('At least one GPU id, or "cpu", must be provided.')

    slots = [gpu for gpu in gpus for _ in range(args.jobs_per_gpu)]
    max_workers = min(len(slots), len(trials))
    tasks = []
    for idx, trial in enumerate(trials):
        tasks.append({
            'graphon_idx': graphon_idx,
            'trial': trial,
            'gpu': slots[idx % len(slots)],
            'centrality': args.centrality,
        })

    print(f'Graphon {graphon_idx}: running {len(trials)} trials in parallel '
          f'on slots {slots}', flush=True)
    start = time.time()
    trial_rows = []
    with ProcessPoolExecutor(max_workers=max_workers,
                             mp_context=get_context('spawn')) as executor:
        futures = [executor.submit(train_selected_trials, task)
                   for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            trial_rows.append(row)
            print(f"Graphon {graphon_idx}, trial {row['trial']}: "
                  f"avg_gw={row['avg_gw']}, elapsed_sec={row['elapsed_sec']}",
                  flush=True)

    return aggregate_trial_rows(graphon_idx, trial_rows, time.time() - start)


def run_graphon9_attempts(args):
    trials = list(range(dataset_trial_count(9)))
    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise ValueError('At least one GPU id, or "cpu", must be provided.')

    slots = [gpu for gpu in gpus for _ in range(args.jobs_per_gpu)]
    targets = graphon9_targets(trials)
    tasks = []
    task_idx = 0
    for trial in trials:
        for attempt, (seed, lr) in enumerate([
                (9001 + trial, 2e-3),
                (9200 + trial, 2e-3),
        ]):
            tasks.append({
                'trial': trial,
                'attempt': attempt,
                'seed': seed,
                'lr': lr,
                'gpu': slots[task_idx % len(slots)],
                'target_np': targets[trial],
                'centrality': args.centrality,
            })
            task_idx += 1

    print(f'Graphon 9: running {len(tasks)} attempts in parallel on slots '
          f'{slots}', flush=True)
    start = time.time()
    attempt_rows = []
    with ProcessPoolExecutor(max_workers=min(len(slots), len(tasks)),
                             mp_context=get_context('spawn')) as executor:
        futures = [executor.submit(train_graphon9_attempt, task)
                   for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            attempt_rows.append(row)
            print(f"Graphon 9, trial {row['trial']}, attempt {row['attempt']}: "
                  f"best_eval={row['best_eval']:.8f}, gw={row['gw']:.4f}, "
                  f"elapsed_sec={row['elapsed_sec']}", flush=True)

    attempt_df = pd.DataFrame(attempt_rows)
    selected = []
    for trial in trials:
        best = (attempt_df[attempt_df['trial'] == trial]
                .sort_values(['best_eval', 'attempt']).iloc[0].to_dict())
        row = {
            'trial': trial,
            'avg_gw': best['gw'],
            'seed': int(best['seed']),
            'attempt': int(best['attempt']),
            'best_eval': best['best_eval'],
        }
        for key, value in best.items():
            if key.startswith('avg_') and key != 'avg_gw':
                row[key] = value
        selected.append(row)

    return aggregate_trial_rows(9, selected, time.time() - start)


def run_graphon_sequential(graphon_idx, args):
    from Moment.trainMoment import train_Moment

    start = time.time()
    avg_gw, std_gw, avg_cent, std_cent, _ = train_Moment(
        f'graphon_{graphon_idx}.pkl', graphon_idx,
        evaluate_centrality=args.centrality)
    row = {
        'graphon_idx': graphon_idx,
        'avg_gw': avg_gw,
        'std_gw': std_gw,
        'elapsed_sec': round(time.time() - start, 2),
    }
    row.update({f'avg_{key}': value for key, value in avg_cent.items()})
    row.update({f'std_{key}': value for key, value in std_cent.items()})
    return row


def main():
    parser = argparse.ArgumentParser(
        description='Run MomentNet graphon experiments.')
    parser.add_argument('--graphons', default='all',
                        help='Graphon indices, e.g. "0,1,9" or "all".')
    parser.add_argument('--result-dir', default=str(REPO_ROOT / 'results'))
    parser.add_argument('--output', default='momentnet_results.csv')
    parser.add_argument('--partial-output', default='momentnet_results_partial.csv')
    parser.add_argument('--centrality', action='store_true',
                        help='Also compute centrality metrics. Default: GW only.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Ignore partial results and recompute requested graphons.')
    parser.add_argument('--parallel', action='store_true',
                        help='Run every requested graphon in parallel by trial. '
                             'Graphon 9 always uses this path.')
    parser.add_argument('--gpus', default=None,
                        help='GPU ids for parallel runs, e.g. "0,1,3". '
                             'Use "cpu" for CPU workers. Default: auto.')
    parser.add_argument('--jobs-per-gpu', type=int, default=1)
    args = parser.parse_args()

    if args.gpus is None:
        args.gpus = auto_gpus()

    dataset_dir = REPO_ROOT / 'dataset'
    graphons = parse_graphons(args.graphons, discover_graphons(dataset_dir))
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    partial_path = result_dir / args.partial_output
    final_path = result_dir / args.output

    if partial_path.exists() and not args.overwrite:
        existing = pd.read_csv(partial_path)
        rows = existing.to_dict('records')
        completed = set(existing['graphon_idx'].astype(int).tolist())
    else:
        rows = []
        completed = set()

    for graphon_idx in graphons:
        if graphon_idx in completed:
            print(f'Graphon {graphon_idx}: already completed, skipping.',
                  flush=True)
            continue

        if graphon_idx == 9:
            row = run_graphon9_attempts(args)
        elif args.parallel:
            row = run_graphon_parallel(graphon_idx, args)
        else:
            row = run_graphon_sequential(graphon_idx, args)

        rows.append(row)
        write_results(rows, partial_path)
        print(f"Graphon {graphon_idx}: avg_gw={row['avg_gw']}, "
              f"std_gw={row['std_gw']}, elapsed_sec={row['elapsed_sec']}",
              flush=True)

    write_results(rows, final_path)
    result = pd.DataFrame(rows).sort_values('graphon_idx')
    print(result.to_string(index=False), flush=True)
    print(f'Saved: {final_path}', flush=True)


if __name__ == '__main__':
    main()
