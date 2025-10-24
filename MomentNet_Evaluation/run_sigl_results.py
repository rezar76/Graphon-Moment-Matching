
import sys
import os 
sys.path.append('../')

import pandas as pd
from SIGL.trainSIGL import train_SIGL

# Ensure result folder exists
os.makedirs('result', exist_ok=True)

# Initialize results list
results = []

# Check if partial results exist
results_file = 'result/sigl_results_partial.csv'
if os.path.exists(results_file):
    results_df = pd.read_csv(results_file)
    results = results_df.to_dict('records')
    completed_graphons = set([r['graphon_idx'] for r in results])
else:
    completed_graphons = set()

# Loop over all datasets
for graphon_idx in range(12):
    if graphon_idx in completed_graphons:
        continue

    dataset_name = f'graphon_{graphon_idx}.pkl'
    result = train_SIGL(dataset_name, graphon_idx)

    # Append results to list
    results.append({
        'graphon_idx': graphon_idx,
        'avg_gw': result[0],
        'std_gw': result[1],
        **{f'avg_{key}': value for key, value in result[2].items()},
        **{f'std_{key}': value for key, value in result[3].items()}
    })

    # Save partial results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)

# Save final results to CSV
final_results_file = 'result/sigl_results.csv'
results_df.to_csv(final_results_file, index=False)

# Print results as a table
print(results_df)