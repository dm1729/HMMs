import numpy as np
from hmmlearn import hmm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorly as tl
from tensorly.decomposition import parafac
from itertools import product

def extract_entries(vector, T):
    """
    Extracts the first 3 entries, leaves a gap of length "T", then takes the next 3, and so on.
    
    Parameters:
    vector (numpy array): The input vector of length N
    T (int): The length of the gap between groups of 3 entries to be extracted
    
    Returns:
    numpy array: A matrix where each row is a group of 3 entries extracted from the vector
    """
    if T is None:
        T = 0
    else:
        assert T >= 0, "T must be a non-negative integer"
    N = len(vector)
    # Calculate the number of complete groups of 3 entries that can be extracted
    n_groups = (N + T) // (3 + T)
    # Initialize an empty list to store the extracted entries
    extracted_entries = []
    for i in range(n_groups):
        start_index = i * (3 + T)
        end_index = start_index + 3
        extracted_entries.append(vector[start_index:end_index])
    # Convert the list of extracted entries into a numpy array and return it
    return np.array(extracted_entries)


def joint_dist(A, B, pi, length=3):
    """
    Compute the true tensor for the joint probability distribution over observed symbols.
    
    Parameters:
    A (numpy array): Transition probability matrix of shape (n_states, n_states)
    B (numpy array): Emission probability matrix of shape (n_states, n_symbols)
    pi (numpy array): Initial state distribution of shape (n_states,)
    n_samples (int): Number of samples (length of observation sequence)
    
    Returns:
    numpy array: True tensor of shape (n_symbols, n_symbols, n_symbols) 
    """
    n_states, n_symbols = B.shape
    tensor_shape = tuple(n_symbols for _ in range(length))
    joint_dist = np.zeros(tensor_shape)
    
    # Iterate over all possible observation sequences
    for obs_seq in product(range(n_symbols), repeat=length):
        # Sum over all possible hidden state sequences
        for state_seq in product(range(n_states), repeat=length):
            prob = pi[state_seq[0]] * B[state_seq[0], obs_seq[0]]
            for t in range(1, length):
                prob *= A[state_seq[t-1], state_seq[t]] * B[state_seq[t], obs_seq[t]]
            joint_dist[obs_seq] += prob
            
    return joint_dist


def sample_tensor(n_states,
                  n_symbols,
                  n_samples,
                  same_chain=False,
                  thin = None,
                  seed=42):
    """
    Generates a tensor representing counts of sequences from a Hidden Markov Model (HMM).
    
    Parameters
    ----------
    n_states : int
        The number of states in the HMM.
    n_symbols : int
        The number of possible emission symbols per state.
    n_samples : int, optional
        The number of samples.
    n_experiments : int, optional
        The number of experiments to generate, by default 10_000.
    seed : int, optional
        Seed for the random number generator, by default 42.
    
    Returns
    -------
    numpy.ndarray
        A tensor of shape (n_symbols, n_symbols, n_symbols), normalized to represent probabilities.
    numpy.ndarray
        The true tensor of shape (n_symbols, n_symbols, n_symbols), normalized to represent probabilities.
    int
        The effective sample size, which is the number of triplets used for the empirical tensor.
    """ 

    np.random.seed(seed)

    # Transition probability matrix
    A = np.random.dirichlet(np.ones(n_states), size=n_states)

    # Emission probability matrix
    B = np.random.dirichlet(np.ones(n_symbols), size=n_states)

    # Initial state distribution
    eigvals, eigvecs = np.linalg.eig(A.T)
    pi = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    pi =  ( pi / pi.sum()  ). reshape(n_states)

    # Create HMM instance and set parameters
    model = hmm.CategoricalHMM(n_components=n_states)
    model.startprob_ = pi
    model.transmat_ = A
    model.emissionprob_ = B

    # Generate samples
    if same_chain:
        X, _ = model.sample(n_samples)
        Obs_all = extract_entries(X.flatten(), thin)
        eff_sample_size = Obs_all.shape[0]
    else:
        eff_sample_size = n_samples // 3
        obs_ls = []
        for _ in range(eff_sample_size):
            X, _ = model.sample(3)
            obs_ls.append(X)
        Obs_all = np.concatenate(obs_ls,axis=1).T

    df = pd.DataFrame(Obs_all, columns=[f'sample_{i}' for i in range(3)])

    for col in df.columns:
        df[col] = pd.Categorical(df[col], categories=range(n_symbols))

    counts = df.groupby([f'sample_{sample_idx}' for sample_idx in range(3)]).size().reset_index(name='counts')

    tensor = counts.pivot_table(index='sample_0', columns=[f'sample_{sample_idx}' for sample_idx in range(1,3)], values='counts', fill_value=0).values

    # Reshape to the desired shape and normalise
    tensor = tensor.reshape(tuple(n_symbols for _ in range(3))) / eff_sample_size

    # True tensor
    true_tensor = joint_dist(A, B, pi)

    return tensor, true_tensor, eff_sample_size

def plot_reconstruction_error(tensor, eff_sample_size, n_states=None,
                              true_tensor = None,r_max=None, one_chain=False,
                              plot_dir = '/home/danmoss/Documents/HMMs/Plots'):
    """
    Plots the reconstruction error of a tensor against the rank of approximation using CP decomposition.
    
    Parameters
    ----------
    tensor : numpy.ndarray
        The emprical tensor to approximate.
    eff_sample_size : int
        The number of 3-samples used to generate the tensor.
    n_states : int
        The number of states in the original HMM.
    true_tensor : numpy.ndarray, optional
        The true tensor, by default None.
    r_max : int, optional
        Custom maximum rank for the CP decomposition, optional.
    plot_dir : str, optional
        The directory where to save the plot, by default '/home/danmoss/Documents/HMMs/Plots'.
    """
    n_symbols = tensor.shape[0]
    if r_max is None:
        r_max = min( n_symbols + max(n_symbols,5) , n_symbols**2)
    l1_errors = []
    l2_errors = []
    eff_par_size = n_symbols**3
    est_stat_error = (1/2)*np.log10(eff_par_size/eff_sample_size)


    for r in range(r_max+1):
        if r == 0:
            l1_errors.append(eff_par_size*mean_absolute_error(tensor.flatten(), np.zeros(tensor.shape).flatten()))
            l2_errors.append(np.sqrt(eff_par_size)*mean_squared_error(tensor.flatten(), np.zeros(tensor.shape).flatten(),squared=False))
        else:    
            # Perform CP decomposition
            factors = parafac(tensor, rank=r)
            
            # Reconstruct the tensor
            tensor_hat = tl.cp_to_tensor(factors)
            
            # Calculate approximation error
            l1_error = eff_par_size*mean_absolute_error(tensor.flatten(), tensor_hat.flatten())
            l2_error = np.sqrt(eff_par_size)*mean_squared_error(tensor.flatten(), tensor_hat.flatten(),squared=False)
            l1_errors.append(l1_error)
            l2_errors.append(l2_error)

    # transform to numpy array
    l1_errors = np.array(l1_errors)
    l2_errors = np.array(l2_errors)
    # log-scale
    l1_errors = np.log10(l1_errors).clip(min=-10)
    l2_errors = np.log10(l2_errors).clip(min=-10)

    # Plot approximation error vs. rank
    plt.figure(figsize=(12, 6))
    plt.plot(range(r_max+1), l1_errors, marker='o', linestyle=':', markersize=5, color='C0', label='L1 reconstruction error')
    plt.plot(range(r_max+1), l2_errors, marker='o', linestyle=':', markersize=5, color='C2', label='L2 reconstruction error')
    plt.hlines(est_stat_error, 0, r_max, linestyle='--', color='C3', label='Est. statistical error')

    # Optional oracle lines
    if n_states is not None:
        plt.vlines(n_states, np.minimum(l1_errors.min(),l2_errors.min()) , 0, linestyle='-', color='black', label='True rank')
    if true_tensor is not None:
        true_stat_error_l1 = eff_par_size*mean_absolute_error(tensor.flatten(), true_tensor.flatten())
        true_stat_error_l2 = np.sqrt(eff_par_size)*mean_squared_error(tensor.flatten(), true_tensor.flatten(),squared=False)
        plt.hlines(np.log10(true_stat_error_l1), 0, r_max, linestyle='-', color='C0', label='True statistical L1 error')
        plt.hlines(np.log10(true_stat_error_l2), 0, r_max, linestyle='-', color='C2', label='True statistical L2 error')

    # Make legend
    plt.legend()

    # Annotations and save
    title_suffix = 'one chain' if one_chain else 'iid'
    file_suffix = 'one_chain' if one_chain else 'repeated_samples'
    plt.title(f'Reconstruction error vs. Rank: {n_states} states, {n_symbols} bins, {eff_sample_size:.1e} triplets, {title_suffix}')
    plt.xlabel('Rank')
    plt.ylabel('Approximation Error, log-scale')
    plt.grid(True)
    plt.savefig(f'{plot_dir}/reconstruction_error_vs_rank_{n_states}_states_{n_symbols}_bins_{eff_sample_size}_triplets_{file_suffix}.png',
                dpi=300, bbox_inches='tight', pad_inches=0.1)