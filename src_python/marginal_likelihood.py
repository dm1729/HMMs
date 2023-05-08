import jax.numpy as jnp
import jax.random as random
import jax
from functools import partial
from jax import vmap, lax, random, jit
import jax.numpy as jnp

def sis_estimator_hmm(obs,
                      num_bins,
                      iters,
                      num_hidden_states,
                      single_bin_weight_prior=1.,
                      single_latent_weight_prior=1.,
                      seed=0):
    
    def scan_body(carry, data_idx): # Used for scan across observations within sis iter
        key, latents, log_evidence_weights = carry
        # def gamma_body(state, log_gamma_accumulated): # Used to compute the gamma across states
        #     log_gamma_val = log_gamma_coefficient_hmm(data_idx, state, obs, latents[:data_idx],
        #                                       num_bins, num_hidden_states, bin_weight_prior_par,
        #                                       latent_prior_par)
        #     return log_gamma_accumulated.at[state].set(log_gamma_val)

        # log_gamma = jax.lax.fori_loop(lower=0,upper=num_hidden_states,
        #                           body_fun=gamma_body,
        #                           init_val=jnp.ones(num_hidden_states,dtype=jnp.float32))
        
        # Function to compute log_gamma for a single state
        def log_gamma_for_state(state):
            return log_gamma_coefficient_hmm(data_idx, state, obs, latents[:data_idx],
                                             num_bins, num_hidden_states, bin_weight_prior_par,
                                             latent_prior_par)

        # Use vmap to compute log_gamma for all states in parallel
        log_gamma = vmap(log_gamma_for_state)(jnp.arange(num_hidden_states))
        
        key, subkey = random.split(key)
        new_latent = random.categorical(key=subkey, logits=log_gamma)
        new_latents = latents.at[data_idx].set(new_latent)
        new_log_evidence_weights = log_evidence_weights + jax.scipy.special.logsumexp(log_gamma)
        return (key, new_latents, new_log_evidence_weights), None

    bin_weight_prior_par = jnp.repeat(single_bin_weight_prior,num_bins)
    latent_prior_par = jnp.repeat(single_latent_weight_prior,num_hidden_states)
    num_obs = obs.shape[0]

    def single_sis_iter(key): # Single iteration over which to vmap
        latents = jnp.zeros(num_obs, dtype=jnp.float32)
        latents = latents.at[0].set(random.categorical(key, logits=jnp.log(latent_prior_par)))
        init_log_evidence_weight = baseline_log_evidence_hmm(obs[0], num_bins, bin_weight_prior_par)
        key = random.PRNGKey(seed)

        init_carry = (key, latents, init_log_evidence_weight)
        final_carry, _ = lax.scan(scan_body, init_carry, jnp.arange(1, num_obs))
        log_evidence_weight = final_carry[2]
        return log_evidence_weight

    key = random.PRNGKey(seed)
    keys = random.split(key, iters)
    log_evidence_weights = vmap(single_sis_iter)(keys)
    return {'evidence': log_evidence_weights }



def log_gamma_coefficient_hmm(idx, state, obs, latents, num_bins, num_states, bin_weight_prior_par, latent_prior_par):
    assert idx <= (len(latents) + 1)
    assert idx >= 1
    
    state_filter = latents[:idx - 1] == state
    obs_up_to_idx = obs[:idx - 1]
    prev_state_obs = obs_up_to_idx[state_filter] # Prev. obs associated to given state
    new_state_obs = jnp.concatenate([prev_state_obs, jnp.array([obs[idx]])]) # Include new obs in state to calc prob.

    log_evidence_new = baseline_log_evidence_hmm(new_state_obs, num_bins, bin_weight_prior_par)
    log_evidence_prev = baseline_log_evidence_hmm(prev_state_obs, num_bins, bin_weight_prior_par)
    log_posterior_latent_weight = jnp.log(
        post_latent_weight_hmm(state, latents[:idx-1], num_states, latent_prior_par, start_state=latents[idx-1])
        )
    log_evidence_ratio = log_evidence_new - log_evidence_prev # Ratio of m(C_k)) quantities in Hairault et al.

    return log_evidence_ratio + log_posterior_latent_weight # log gamma_k in Hairault et al.


def baseline_log_evidence_hmm(obs, num_bins, bin_weight_prior_par):
    """
    Calculate the "baseline log evidence" for an HMM.
    For given observations, calculates the joint likelihood of these given a shared latent state.
    Amounts to taking the log ratio of posterior and prior normalising constants.

    Args:
        obs (jnp.ndarray): Observations.
        num_bins (int): Number of bins.
        bin_weight_prior_par (float): Bin weight prior parameter.

    Returns:
        float: Log evidence.
    """
    if obs.size == 0:
        return 0

    bin_counts = jnp.bincount(obs, length=num_bins)
    bin_weight_posterior_par = bin_weight_prior_par + bin_counts
    log_evidence = (jnp.sum(lax.lgamma(bin_weight_posterior_par)) - lax.lgamma(jnp.sum(bin_weight_posterior_par))
                    + lax.lgamma(jnp.sum(bin_weight_prior_par)) - jnp.sum(lax.lgamma(bin_weight_prior_par))
                    )
    return log_evidence

def post_latent_weight_hmm(state, latents, num_states, latent_prior_par, start_state):
    """
    Calculate the posterior latent weight for an HMM.

    Args:
        state (int): State index.
        latents (jnp.ndarray): Latent states.
        num_states (int): Number of states.
        latent_prior_par (float): Latent prior parameter.
        start_state (int): Start state index.

    Returns:
        float: Posterior latent weight.
    """ 
    transition_counts_from_start_state = transition_count(latent_states=latents, num_states=num_states)[start_state,]
    latent_post_par = latent_prior_par + transition_counts_from_start_state
    return latent_post_par[state] / jnp.sum(latent_post_par)


def gamma_coefficient_mixture(idx, state, obs, latents, num_bins, num_states, bin_weight_prior_par, latent_prior_par):
    assert idx <= (len(latents) + 1)
    assert idx >= 2

    obs_up_to_idx = jnp.asarray(obs[:, :idx - 1])
    eff_obs = jnp.asarray(obs_up_to_idx[:, latents[:idx - 1] == state])

    log_evidence_new = baseline_log_evidence_mixture(jnp.column_stack([eff_obs, obs[:, idx]]), num_bins, bin_weight_prior_par)
    log_evidence_old = baseline_log_evidence_mixture(eff_obs, num_bins, bin_weight_prior_par)
    posterior_latent_weight = post_latent_weight_mixture(state, latents[:idx-1], num_states, latent_prior_par)
    evidence_ratio = jnp.exp(log_evidence_new - log_evidence_old)

    return evidence_ratio * posterior_latent_weight

def baseline_log_evidence_mixture(obs, num_bins, bin_weight_prior_par):
    """
    Calculate the "baseline log evidence" for a mixture model.
    For given observations, calculates the joint likelihood of these given a shared latent state.
    Amounts to taking the log ratio of posterior and prior normalising constants.

    Args:
        obs (jnp.ndarray): Observations.
        num_bins (int): Number of bins.
        bin_weight_prior_par (float): Bin weight prior parameter.

    Returns:
        float: Log evidence.
    """
    if obs.shape[0] == 0:
        return 0

    obs_dim = obs.shape[0]
    log_evidence = 0
    for dim in range(obs_dim):
        bin_counts = jnp.bincount(obs[dim, :], length=num_bins)
        bin_weight_posterior_par = bin_weight_prior_par + bin_counts
        log_evidence += (jnp.sum(lax.lgamma(bin_weight_posterior_par)) - lax.lgamma(jnp.sum(bin_weight_posterior_par))
                         + lax.lgamma(jnp.sum(bin_weight_prior_par)) - jnp.sum(lax.lgamma(bin_weight_prior_par))
                         )
    return log_evidence

def post_latent_weight_mixture(state, latents, num_states, latent_prior_par):
    """
    Calculate the posterior latent weight for a mixture model.

    Args:
        state (int): State index.
        latents (jnp.ndarray): Latent states.
        num_states (int): Number of states.
        latent_prior_par (float): Latent prior parameter.

    Returns:
        float: Posterior latent weight.
    """
    state_counts = jnp.bincount(latents, length=num_states)
    latent_post_par = latent_prior_par + state_counts
    return latent_post_par[state] / jnp.sum(latent_post_par)

@jit
def transition_count(latent_states, num_states):
    transition_count_mat = jnp.array([
    jnp.bincount(latent_states[:-1] * num_states + latent_states[1:],
                 length=num_states * num_states)]).reshape(num_states, num_states)
    return transition_count_mat


def log_exponential_mean(x: jnp.array):
    """
    Compute the log of the mean of exponentials of input elements.
    """
    return jax.scipy.special.logsumexp(x) - jnp.log(len(x))