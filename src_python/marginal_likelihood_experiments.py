import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import jax.numpy as jnp
import jax
from itertools import permutations
from hmmlearn import hmm
from functools import partial
from itertools import product
from hmm_mcmc import *
from marginal_likelihood import *
import os
from datetime import datetime
import argparse

# Set up the argument parser
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SIS experiments')
    parser.add_argument('--sis_iters', type=int, default=5_000,
                        help='Number of SIS iterations')
    parser.add_argument('--project_dir', type=str, default='/home/danmoss/Documents/HMMs',
                        help='Project directory')
    parser.add_argument('--data_suffix', type=str, default='Experiments/categorical_mix_data',
                        help='Path to data dir, appended to project_dir')
    parser.add_argument('--output_suffix', type=str, default='Experiments/categorical_mix_results',
                    help='Path to sis output dir, appended to project_dir')
    parser.add_argument('--data_seed', type=int, default=0)
    parser.add_argument('--dataset_sizes_to_fit', type=int, nargs='+', default=[250, 500, 1000, 2000, 4000],
                        help='List of dataset sizes to fit (space-separated)')
    parser.add_argument('--is_mixture', action='store_true', default=False, help='flag for mixture, else default to HMM')
    args = parser.parse_args()


sis_iters = args.sis_iters
data_filepath = os.path.join(args.project_dir, args.data_suffix, f"data_seed_{args.data_seed}.csv")

data = pd.read_csv(data_filepath)
data_idx_list = [int(col_name.split("_")[1]) for col_name in data.columns]

# with open(f'{args.project_dir}/src_python/mixture_params.pkl', 'rb') as f:
#     params_df = pickle.load(f) # To recover the true number of states and bins

dataset_sizes_to_fit = args.dataset_sizes_to_fit # Partial sizes of dataset
max_obs = len(data)
assert max_obs >= max(dataset_sizes_to_fit) # Make sure we have enough data
counter = 0

if args.is_mixture:
    sis_estimator_function = sis_estimator_mixture
else:
    sis_estimator_function = sis_estimator_hmm

for data_key in data_idx_list: # 32 datasets
    dataset = data.filter(regex=f"dataset_{data_key}").to_numpy()
    if args.is_mixture:
        assert dataset.shape == (max_obs,3) # 3D mixture
    else:
        if len(dataset.shape) > 1: # Reshaping as required
            assert dataset.shape[1] == 1
            assert len(dataset.shape) == 2
            dataset = dataset.flatten()
        assert dataset.shape == (max_obs,) # 1D HMM
    obs = jnp.array(dataset,dtype=jnp.float16)
    with open(os.path.join(
        args.project_dir,args.data_suffix,f"categorical_{'mix' if args.is_mixture else 'hmm'}_params_{data_key}.pkl"
        ), 'rb') as f:
        param_dict = pickle.load(f)
    true_states = param_dict['states']
    num_bins = param_dict['emissions'].shape[1]
    assert num_bins == true_states + 1 # The number of bins is the number of states + 1 in our experiments

    # Only fit to the true number of states, or the true number +- 1
    if true_states == 2:
        states_to_fit = [2,3]
    else:
        states_to_fit = [true_states-1,
                         true_states,
                         true_states+1]
        
    # Run the SIS for each combination of the above
    for fitted_states in states_to_fit: # Loop over this first, as it's a static argument which requires recompilation
        for dataset_size in dataset_sizes_to_fit: # 20-30 combos

            # File name
            output_file_name = f'dataset_{data_key}_samples_{dataset_size}_fitted_states_{fitted_states}.pkl'
            output_file_path = os.path.join(args.project_dir, args.output_suffix, f"seed_{args.data_seed}", output_file_name)

            # Compute if not done already
            if not os.path.exists(output_file_path):
                counter +=1
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Performing SIS procedure (data key {data_key}), total count {counter}")
                sis_out = sis_estimator_function(obs[:dataset_size], # Only fit to part of the data
                                                iters=sis_iters,
                                                num_bins=num_bins,
                                                num_states=fitted_states,
                                                single_latent_weight_prior=1.0, # Potentially experiment with differences here
                                                single_bin_weight_prior=1.0)
                
                # Saving the results
                with open(output_file_path,'wb') as f:
                    pickle.dump(sis_out, f)