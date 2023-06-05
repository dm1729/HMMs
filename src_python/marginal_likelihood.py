import jax.numpy as jnp
import jax.random as random
import jax
from functools import partial
from jax import vmap, lax, random, jit
import jax.numpy as jnp

def sis_estimator_hmm(obs,
                      iters,
                      num_bins,
                      num_states,
                      single_bin_weight_prior=1.,
                      single_latent_weight_prior=1.,
                      seed=0):
    """
    Implements the Sequential Importance Sampling (SIS) estimator for the marginal likelihood
    in a discrete HIdden Markov Model (HMM)
    (Hairault et al. 2022 https://arxiv.org/abs/2205.05416)
    
    Parameters
    ----------
    obs : array_like
        The observations from the HMM.
    iters : int
        The number of iterations to perform.
    num_bins : int
        The number of bins for the observations.
    num_states : int
        The number of states in the HMM.
    single_bin_weight_prior : float, optional
        The prior weight for each bin. Default is 1.
    single_latent_weight_prior : float, optional
        The prior weight for each latent state. Default is 1.
    seed : int, optional
        The seed for the random number generator. Default is 0.
        
    Returns
    -------
    log_evidence_weights : array_like, shape (iters,)
        The estimated log evidence from the SIS, one for each iteration.
    """
    def scan_body(carry, obs_datum): # Used for scan across observations within sis iter
        log_evidence_weights, partial_obs_old, key, prev_latent, latent_bool_arr_old, data_idx = carry
        partial_obs_new = partial_obs_old.at[data_idx].set(obs_datum)
        latent_bool_arr_ones = latent_bool_arr_old.at[:,data_idx].set(1)
        log_gamma = log_gamma_coefficient_hmm(prev_latent,
                                              partial_obs_old,
                                              partial_obs_new,
                                              latent_bool_arr_old=latent_bool_arr_old,
                                              latent_bool_arr_new=latent_bool_arr_ones,
                                              num_bins=num_bins,
                                              num_states=num_states,
                                              num_obs=num_obs,
                                              bin_weight_prior_par=bin_weight_prior_par,
                                              latent_prior_par=latent_prior_par)
        
        key, subkey = random.split(key)
        new_latent = random.categorical(key=subkey, logits=log_gamma)
        latent_bool_arr_new = latent_bool_arr_old.at[new_latent,data_idx].set(1)
        new_log_evidence_weights = log_evidence_weights + jax.scipy.special.logsumexp(log_gamma)
        return (new_log_evidence_weights, partial_obs_new, key, new_latent, latent_bool_arr_new, data_idx+1), None

    bin_weight_prior_par = jnp.repeat(single_bin_weight_prior,num_bins*num_states).reshape(num_states,num_bins)
    latent_prior_par = jnp.repeat(single_latent_weight_prior,num_states**2).reshape(num_states,num_states)
    num_obs = obs.shape[0]

    def single_sis_iter(key): # Single iteration over which to vmap
        latent_bool_arr = jnp.zeros((num_states,num_obs), dtype=jnp.float16)
        init_state = random.categorical(key, logits=jnp.log(latent_prior_par[1,:]))
        latent_bool_arr = latent_bool_arr.at[init_state,0].set(1)
        partial_obs = -jnp.ones_like(obs, dtype=jnp.float16)
        partial_obs = partial_obs.at[0].set(obs[0])
        init_log_evidence_weight = baseline_log_evidence_hmm(partial_obs, latent_bool_arr, num_states, num_bins, bin_weight_prior_par)[init_state]
        init_carry = (init_log_evidence_weight, partial_obs, key, init_state, latent_bool_arr , 1)
        final_carry, _ = lax.scan(scan_body, init_carry, jax.lax.dynamic_slice_in_dim(obs, 1, num_obs-1, axis=0) )
        log_evidence_weight = final_carry[0]
        return log_evidence_weight

    key = random.PRNGKey(seed)
    keys = random.split(key, iters)
    log_evidence_weights = vmap(single_sis_iter)(keys)
    return log_evidence_weights

def log_gamma_coefficient_hmm(prev_latent,
                              partial_obs_old,
                              partial_obs_new,
                              latent_bool_arr_old,
                              latent_bool_arr_new,
                              num_bins,
                              num_states,
                              num_obs,
                              bin_weight_prior_par,
                              latent_prior_par):
    """
    Computes the log gamma coefficient for a Hidden Markov Model (HMM).
    
    Parameters
    ----------
    prev_latent : array_like
        The previous latent states.
    partial_obs_old : array_like
        The old partial observations.
    partial_obs_new : array_like
        The new partial observations.
    latent_bool_arr_old : array_like
        The old boolean array of latents (the reference)
    latent_bool_arr_new : array_like
        The new boolean array of latents (the 'proposal')
    num_bins : int
        The number of bins for the observations.
    num_states : int
        The number of states in the HMM.
    num_obs : int
        The number of observations.
    bin_weight_prior_par : float
        The prior parameter for the bin weights.
    latent_prior_par : float
        The prior parameter for the latent weights.
        
    Returns
    -------
    log_gamma : array_like
        The log gamma coefficient.
    """
    log_evidence_old = baseline_log_evidence_hmm(partial_obs_old, latent_bool_arr_old, num_states,num_bins, bin_weight_prior_par) # TODO: Modify this function, to do across states
    log_evidence_new = baseline_log_evidence_hmm(partial_obs_new, latent_bool_arr_new, num_states,num_bins, bin_weight_prior_par)
    log_posterior_latent_weight = jnp.log(
        post_latent_weight_hmm(latent_bool_arr_old, num_states,num_obs, latent_prior_par, start_state=prev_latent)
        )
    log_evidence_ratio = log_evidence_new - log_evidence_old # Ratio of m(C_k)) quantities in Hairault et al.
    # Vector of length num_states, with log_gamma_k for each state

    return log_evidence_ratio + log_posterior_latent_weight # log gamma_k in Hairault et al.


@partial(jit, static_argnums=(2,3))
def baseline_log_evidence_hmm(obs,latent_bool_arr, num_states, num_bins, bin_weight_prior_par):
    """
    Calculate the "baseline log evidence" for an HMM.
    
    Parameters
    ----------
    obs : array_like
        The observations.
    latent_bool_arr : array_like
        Boolean array indicating the latent state for each observation.
    num_states : int
        The number of states in the HMM.
    num_bins : int
        The number of bins for the observations.
    bin_weight_prior_par : float
        The prior parameter for the bin weights.
        
    Returns
    -------
    log_evidence : array_like
        The baseline log evidence - vector of m(C_k) as in Hairault et al. (2022)
    """
    # Stack the observations, so that each row is a state, and each column is an observation
    # In each column, the value is the obs if the row is the active latent, and -1 otherwise
    # E.g. if obs = [0,1,2] and latent_bool_arr = [[1,0,1],[0,1,0]] (two states), then stacked_obs = [[0,-1,2],[-1,1,-1]]
    stacked_obs = jnp.multiply(1+obs, latent_bool_arr) - 1

    # Count the number of observations in each bin for each state, then discard the "-1" values
    bin_counts_all = multi_bincount(stacked_obs, length=num_bins+1) # Includes "-1" values
    bin_counts = lax.dynamic_slice(bin_counts_all,(0,1),(num_states,num_bins)) # Discard "-1" values

    bin_weight_posterior_par = bin_weight_prior_par + bin_counts
    log_evidence = (jnp.sum(lax.lgamma(bin_weight_posterior_par),axis=1) - lax.lgamma(jnp.sum(bin_weight_posterior_par,axis=1))
                    + lax.lgamma(jnp.sum(bin_weight_prior_par,axis=1)) - jnp.sum(lax.lgamma(bin_weight_prior_par),axis=1)
                    )
    return log_evidence

@partial(jit,static_argnums=(1,2))
def post_latent_weight_hmm(latent_bool_arr, num_states,num_obs, latent_prior_par, start_state):
    """
    Calculate the posterior latent weight for an HMM.
    Uses Dirichlet conjugacy to update parameter based on observed transitions.
    
    Parameters
    ----------
    latent_bool_arr : array_like
        Boolean array indicating the latent state for each observation.
    num_states : int
        The number of states in the HMM.
    num_obs : int
        The number of observations.
    latent_prior_par : float
        The prior parameter for the latent weights.
    start_state : int
        The initial state.
        
    Returns
    -------
    float
        The posterior latent weight.
    """ 
    transition_counts_from_start_state = transition_count(latent_bool_arr=latent_bool_arr, num_states=num_states,num_obs=num_obs)[start_state,]
    latent_post_par = latent_prior_par[start_state,] + transition_counts_from_start_state
    return latent_post_par / jnp.sum(latent_post_par)


@partial(jit, static_argnums=(1,2))
def transition_count(latent_bool_arr, num_states, num_obs):
    """
    Compute the transition count for a sequence of latent states.
    
    Parameters
    ----------
    latent_bool_arr : array_like
        Boolean array indicating the latent state for each observation.
    num_states : int
        The number of states in the HMM.
    num_obs : int
        The number of observations.
        
    Returns
    -------
    transition_count_mat : array_like
        The transition count matrix.
    """
    start_states = lax.dynamic_slice( latent_bool_arr, (0,0), (num_states,num_obs-1) )
    end_states = lax.dynamic_slice( jnp.roll(latent_bool_arr,-1,axis=1), (0,0), (num_states,num_obs-1) )
    transition_count_mat = jnp.matmul ( start_states, jnp.transpose(end_states) )
    return transition_count_mat


@partial(jax.jit, static_argnums=(1,))
def multi_bincount(arr,length):
    """
    Compute the bincount for each row of an array.
    Used to compute the bincounts for each latent state in parallel.
    
    Parameters
    ----------
    arr : array_like
        The array to compute the bincount for.
    length : int
        The number of bins.
        
    Returns
    -------
    bincount : array_like
        The bincount for each row of the array.
    """
    return jax.vmap(partial( jax.numpy.bincount, length=length))(jnp.int16(arr))

def log_marginal_likelihood_iid(obs, num_bins, single_bin_weight_prior=1.):
    """
    Computes the marginal likelihood for an IID model (i.e. one state) which is available in closed form.

    Parameters
    ----------
    obs : array_like
        The observations.
    num_bins : int
        The number of bins for the observations.
    bin_weight_prior_par : float
        The prior parameter for the bin weights.
    
    Returns
    -------
    float
        The marginal likelihood.
    """
    bin_weight_prior_par = single_bin_weight_prior * jnp.ones(num_bins,dtype=jnp.float32)
    bin_weight_posterior_par = bin_weight_prior_par + jnp.bincount(obs, length=num_bins)
    log_evidence = (jnp.sum(lax.lgamma(bin_weight_posterior_par)) - lax.lgamma(jnp.sum(bin_weight_posterior_par))
                    + lax.lgamma(jnp.sum(bin_weight_prior_par)) - jnp.sum(lax.lgamma(bin_weight_prior_par))
                    )
    return log_evidence


# def gamma_coefficient_mixture(idx, state, obs, latents, num_bins, num_states, bin_weight_prior_par, latent_prior_par):
#     assert idx <= (len(latents) + 1)
#     assert idx >= 2

#     obs_up_to_idx = jnp.asarray(obs[:, :idx - 1])
#     eff_obs = jnp.asarray(obs_up_to_idx[:, latents[:idx - 1] == state])

#     log_evidence_new = baseline_log_evidence_mixture(jnp.column_stack([eff_obs, obs[:, idx]]), num_bins, bin_weight_prior_par)
#     log_evidence_old = baseline_log_evidence_mixture(eff_obs, num_bins, bin_weight_prior_par)
#     posterior_latent_weight = post_latent_weight_mixture(state, latents[:idx-1], num_states, latent_prior_par)
#     evidence_ratio = jnp.exp(log_evidence_new - log_evidence_old)

#     return evidence_ratio * posterior_latent_weight

# def baseline_log_evidence_mixture(obs, num_bins, bin_weight_prior_par):
#     """
#     Calculate the "baseline log evidence" for a mixture model.
#     For given observations, calculates the joint likelihood of these given a shared latent state.
#     Amounts to taking the log ratio of posterior and prior normalising constants.

#     Args:
#         obs (jnp.ndarray): Observations.
#         num_bins (int): Number of bins.
#         bin_weight_prior_par (float): Bin weight prior parameter.

#     Returns:
#         float: Log evidence.
#     """
#     if obs.shape[0] == 0:
#         return 0

#     obs_dim = obs.shape[0]
#     log_evidence = 0
#     for dim in range(obs_dim):
#         bin_counts = jnp.bincount(obs[dim, :], length=num_bins)
#         bin_weight_posterior_par = bin_weight_prior_par + bin_counts
#         log_evidence += (jnp.sum(lax.lgamma(bin_weight_posterior_par)) - lax.lgamma(jnp.sum(bin_weight_posterior_par))
#                          + lax.lgamma(jnp.sum(bin_weight_prior_par)) - jnp.sum(lax.lgamma(bin_weight_prior_par))
#                          )
#     return log_evidence

# def post_latent_weight_mixture(state, latents, num_states, latent_prior_par):
#     """
#     Calculate the posterior latent weight for a mixture model.

#     Args:
#         state (int): State index.
#         latents (jnp.ndarray): Latent states.
#         num_states (int): Number of states.
#         latent_prior_par (float): Latent prior parameter.

#     Returns:
#         float: Posterior latent weight.
#     """
#     state_counts = jnp.bincount(latents, length=num_states)
#     latent_post_par = latent_prior_par + state_counts
#     return latent_post_par[state] / jnp.sum(latent_post_par)

def log_exponential_mean(x: jnp.array):
    """
    Compute the log of the mean of exponentials of input elements.
    """
    return jax.scipy.special.logsumexp(x) - jnp.log(len(x))
