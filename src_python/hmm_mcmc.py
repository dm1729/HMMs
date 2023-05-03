import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
from hmm_helpers import *
import jax.random as random

def prior_set(num_states, num_bins=None, single_trans_row_prior=None, single_emission_prior=None, set_mix_prior=False):
    """
    Sets prior parameters for binned HMM sampling.
    
    Args:
    - num_states (int): number of hidden states in the HMM.
    - num_bins (int): number of bins in the binned observation data. If None, no emission prior is set.
    - single_trans_row_prior (array): a 1D array of size num_states specifying the prior for each row of the transition matrix. If None, the prior is set to a uniform distribution.
    - single_emission_prior (array): a 1D array of size num_bins specifying the prior for each emission probability vector. If None, the prior is set to a uniform distribution.
    - set_mix_prior (bool): if True, returns the mix_weight_prior instead of the trans_mat_prior.
    
    Returns:
    - dict: a dictionary containing either trans_mat_prior and emission_prior or mix_weight_prior and emission_prior.
    """
    if single_trans_row_prior is None:
        single_trans_row_prior = jnp.ones(num_states)
    if num_bins is None:
        emission_prior = None
    else:
        if single_emission_prior is None:
            single_emission_prior = jnp.ones(num_bins)
        emission_prior = jnp.transpose(jnp.tile(single_emission_prior, (num_states,1))) # num_states x num_bins
    if set_mix_prior:
        return {"mix_weight_prior": single_trans_row_prior, "emission_prior": emission_prior}
    else:
        trans_mat_prior = jnp.tile(single_trans_row_prior, (num_states, 1)) # num_states x num_states
        return {"trans_mat_prior": trans_mat_prior, "emission_prior": emission_prior}


def sample_trans_mat(hidden_states, trans_mat_prior, key):
    """
    Samples a transition matrix from the conditional distribution given
    a sequence of hidden states and a prior over transition matrices.

    Args:
        hidden_states: An array of integers of shape `(n_samples,)` representing the
        sequence of hidden states.
        trans_mat_prior: A 2D array of shape `(n_states, n_states)` representing the
        prior over transition matrices. Each row of the matrix is a Dirichlet prior
        over the transitions from a given state.
        key: Optional `jax.random.PRNGKey`. If provided, used for random number generation.

    Returns:
        A 2D array of shape `(n_states, n_states)` representing the sampled transition matrix.
    """
    num_states = trans_mat_prior.shape[0] # recover the number of distinct states
    transition_count = jnp.zeros((num_states, num_states), dtype=jnp.int32)
    sample_size = hidden_states.shape[0]
    for sample_idx in range(sample_size - 1):
        trans_mat_idx = (hidden_states[sample_idx], hidden_states[1+sample_idx])
        transition_count = transition_count.at[trans_mat_idx].add(1)
    # Counts according to transition
    trans_mat_post = trans_mat_prior + transition_count # New Dirichlet weights
    trans_mat_draw = jnp.zeros((num_states, num_states))
    for i in range(num_states):
        key, subkey = random.split(key)
        trans_mat_draw = trans_mat_draw.at[i, :].set(random.dirichlet(subkey, trans_mat_post[i, :]))
    # draws Q from newly updated Dirichlet weights
    return trans_mat_draw

def sample_emission_weights(hidden_states, obs_data, emission_prior, key):
    """
    Samples emission weights from the conditional distribution given
    a sequence of hidden states and observed data, and a prior over emission weights.

    Args:
        hidden_states: An array of integers of shape `(n_samples,)` representing the
        sequence of hidden states.
        obs_data: An array of integers of shape `(n_samples,)` representing the observed data.
        emission_prior: A 2D array of shape `(n_states, n_bins)` representing the
        prior over emission weights. Each row of the matrix is a Dirichlet prior
        over the emissions from a given state.
        key: Optional `jax.random.PRNGKey`. If provided, used for random number generation.

    Returns:
        A 2D array of shape `(n_states, n_bins)` representing the sampled emission weights.
    """
    num_states, num_bins = emission_prior.shape
    emission_count = jnp.zeros((num_states, num_bins), dtype=jnp.int32)
    for i in range(num_states):
        for j in range(num_bins):
            emission_count = emission_count.at[i, j].add(jnp.dot( (hidden_states == i) , (obs_data == j) ) )
    emission_post = emission_prior + emission_count
    emissions_draw = jnp.zeros((num_states, num_bins))
    for i in range(num_states):
        key, subkey = random.split(key)
        emissions_draw = emissions_draw.at[i, :].set(random.dirichlet(subkey, emission_post[i, :]))
    return emissions_draw


def sample_hidden_states(obs_data, trans_mat, emission_mat, key=None):
    """
    Samples a sequence of hidden states given observations, a transition matrix, and an
    emission matrix.

    Args:
        obs_data: An array of integers of shape `(n_samples,)` representing the sequence of
        observations.
        trans_mat: A 2D array of shape `(n_states, n_states)` representing the transition
        probabilities between states.
        emission_mat: A 2D array of shape `(n_states, n_bins)` representing the emission
        probabilities for each state.
        key: Optional `jax.random.PRNGKey`. If provided, used for random number generation.

    Returns:
        A tuple containing two elements:
        - An array of integers of shape `(n_samples,)` representing the sequence of
        hidden states.
        - A scalar representing the log-likelihood of the generated sequence of hidden states.
    """
    binned_emission_func=compute_emission_probs_multinomial
    binned_emission_kwargs={'emission_mat':emission_mat}
    forward, backward, log_likelihood = forward_backward(obs_data=obs_data, trans_mat=trans_mat,
                                                         emission_func=binned_emission_func,
                                                         emission_kwargs=binned_emission_kwargs)
    cond_prob_kwargs={'forward':forward, 'backward':backward}

    # Probability of being in each state at each time point given data
    # The length of the first dimension is t
    cond_prob = conditional_probability(**cond_prob_kwargs)
    # Joint probability of being in state i at time t and state j at time t+1, given data
    # Consequently the length of the first dimension is t-1
    joint_cond_probs = joint_conditional_probabilities(obs_data=obs_data,trans_mat=trans_mat,
                                                       emission_func=binned_emission_func,
                                                       emission_kwargs=binned_emission_kwargs,
                                                       cond_prob_kwargs=cond_prob_kwargs)
    
    num_obs = len(obs_data)
    broadcast_cond_probs = jnp.expand_dims(cond_prob[:-1,:], axis=2)
    transition_probs = joint_cond_probs / broadcast_cond_probs

    def scan_fun(carry, t):
        key, prev_state = carry
        key, subkey = random.split(key)
        state = random.categorical(subkey, logits=jnp.log(transition_probs[t-1,prev_state,:]))
        return (key, state), state
    
    key, subkey = random.split(key)
    init_state = random.categorical(subkey, logits=jnp.log(cond_prob[0,:]))
    init_carry = (key, init_state)
    _, hidden_states_draw_next = jax.lax.scan(scan_fun, init_carry, jnp.arange(1, num_obs))
    hidden_states_draw = jnp.append(jnp.array([init_state]), hidden_states_draw_next, axis=0)

    return hidden_states_draw, log_likelihood


def binned_prior_sampler(obs_data,
                         num_states,
                         num_bins,
                         num_its,
                         trans_mat_dir_par=1,
                         emission_dir_par=1,
                         hidden_states_init=None,
                         seed=0):
    """
    This function performs a Gibbs sampler for a Hidden Markov Model (HMM) with binned observations.
    It iteratively samples the transition matrix, emission weights, and hidden states and
    accumulates the draws. It also tracks the maximum likelihood estimate of the hidden states.

    Args:
        obs_data (array-like): Observed binned data.
        num_states (int): Number of hidden states in the HMM.
        num_bins (int): Number of bins for the observed data.
        num_its (int): Number of iterations to run the Gibbs sampler.
        trans_mat_dir_par (float, optional): Dirichlet prior parameter for the transition matrix. Defaults to 1.
        emission_dir_par (float, optional): Dirichlet prior parameter for the emission weights. Defaults to 1.
        hidden_states_init (array-like, optional): Initial hidden states to use. If not provided, they will be randomly initialized.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        dict: A dictionary containing the following keys:
            - "trans_mat_draws": Transition matrix draws.
            - "emission_weight_draws": Emission weight draws.
            - "log_like_draws": Log likelihood draws.
            - "hidden_states_MLE": Maximum likelihood estimate of the hidden states.
    """
    key = random.PRNGKey(seed)
    prior_params = prior_set(num_states, num_bins, jnp.full(num_states, trans_mat_dir_par), jnp.full(num_bins, emission_dir_par))
    trans_mat_prior = prior_params['trans_mat_prior']
    emission_prior = prior_params['emission_prior']
    num_obs = len(obs_data)
    
    if hidden_states_init is None:
        key, subkey = random.split(key)
        hidden_states = random.randint(subkey, (num_obs,), 0, num_states)
    else:
        hidden_states = hidden_states_init

    def scan_fun(carry, _):
        (key, hidden_states_prev, max_log_like_prev), hidden_states_MLE_prev = carry

        key, subkey = random.split(key)
        trans_mat_draw = sample_trans_mat(hidden_states_prev, trans_mat_prior, subkey)

        key, subkey = random.split(key)
        emission_draw = sample_emission_weights(hidden_states_prev, obs_data, emission_prior, subkey)

        key, subkey = random.split(key)
        hidden_states_draw, log_like_draw = sample_hidden_states(obs_data, trans_mat_draw, emission_draw, subkey)

        if log_like_draw > max_log_like_prev:
            hidden_states_MLE = hidden_states_draw
            max_log_like = log_like_draw
        else:
            hidden_states_MLE = hidden_states_MLE_prev
            max_log_like = max_log_like_prev

        return ((key, hidden_states_draw, max_log_like), hidden_states_MLE), (trans_mat_draw,emission_draw,log_like_draw)
    
    key, subkey = random.split(key)
    init_carry = (key, hidden_states, float('-inf'), hidden_states)

    (_, hidden_states_MLE), draws = jax.lax.scan(scan_fun, init_carry, jnp.arange(num_its))

    trans_mat_draws, emission_draws, log_like_draws = zip(*draws)

    return {
        "trans_mat_draws": jnp.stack(trans_mat_draws),
        "emission_weight_draws": jnp.stack(emission_draws),
        "log_like_draws": jnp.stack(log_like_draws),
        "hidden_states_MLE": hidden_states_MLE
    }



def uniformly_bin(obs_data, num_bins, link=None):
    """
    Bin observed data into uniformly spaced bins.

    Parameters:
    obs_data (jax.numpy.ndarray): Observed data to be binned.
    num_bins (int): Number of bins to create.
    link (callable): Optional link function to apply to data before binning.

    Returns:
    jax.numpy.ndarray: Binned data.
    """
    if link is not None:
        obs_data = link(obs_data)
    binned_data = jnp.floor(num_bins * obs_data).astype(int)
    binned_data = jnp.clip(binned_data, 0, num_bins - 1)
    return binned_data


@partial(vmap, in_axes=(0, None, None))
def truncated_inv_logit(x , lower: float, upper: float):
    """
    Applies a version of the inverse logit map to the input tensor `x` where the output is linearly
    interpolated between `lower` and `upper` bounds. This function is vectorized using `jax.vmap`.

    Args:
        x: Input tensor of shape `(n_samples,)`
        lower: A scalar float representing the lower bound for the linear interpolation.
        upper: A scalar float representing the upper bound for the linear interpolation.

    Returns:
        A tensor of shape `(n_samples,)` with the truncated inverse logit applied to each element.
        The output is linearly interpolated between the `lower` and `upper` bounds.
    """
    assert lower <= upper
    def logit_part(x):
        return 1 / (1 + jnp.exp(-x))
    def linear_part(x):
        return logit_part(lower) + ((x - lower) / (upper - lower)) * (
                                    logit_part(upper) - logit_part(lower))
    if lower == upper:
        y = logit_part(x)
    else:
        y = jnp.where((x < lower) | (x > upper), logit_part(x), linear_part(x))                                  
    return y