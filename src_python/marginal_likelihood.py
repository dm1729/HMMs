import jax.numpy as jnp
from jax import lax
import jax.random as random
import jax

def sis_estimator(obs, num_bins, iters, num_hidden_states,
                  bin_weight_prior_par=None, latent_prior_par=None,
                  is_mixture=True, output_latents=False, verbose=False):
    if bin_weight_prior_par is None:
        bin_weight_prior_par = jnp.ones(num_bins)
    if latent_prior_par is None:
        latent_prior_par = jnp.ones(num_hidden_states)
    assert num_hidden_states == len(latent_prior_par), (
        "incomptatible latent prior parameter"
    )
    assert num_bins == len(bin_weight_prior_par), (
        "incomptatible bin weight prior parameter"
    )
    if is_mixture:
        num_obs = obs.shape[1]
        obs_dim = obs.shape[0]
        assert obs_dim >= 3, "obs_dim must exceed 3 for identifiability"
    else:
        obs = jnp.asarray([obs])
        obs = jnp.resize(obs, (min(obs.shape[0], obs.shape[1]), max(obs.shape[0], obs.shape[1])))
        num_obs = obs.shape[1]

    log_evidence_weights = jnp.zeros(iters)
    
    latents_ls = None
    if output_latents:
        latents_ls = jnp.zeros((iters, num_obs))

    for sis_iter in range(iters):
        if (sis_iter % 10 == 0) & verbose:
            print(f'{sis_iter} Iteration number')
        latents = jnp.zeros(num_obs)
        key = random.PRNGKey(0)
        latents[0] = random.categorical(key, logits=jnp.log(latent_prior_par))
        log_evidence_weights[sis_iter] = baseline_log_evidence(obs[:, 0], num_bins,
                                                               bin_weight_prior_par,
                                                               is_mixture=is_mixture)

        for data_idx in range(1, num_obs):
            gamma = jnp.zeros(num_hidden_states)
            for state in range(num_hidden_states):
                gamma[state] = gamma_coefficient(data_idx, state, obs, latents[:data_idx],
                                                 num_bins, num_hidden_states, bin_weight_prior_par,
                                                 latent_prior_par, is_mixture)
            if jnp.sum(gamma) == 0:
                print("Error: sum of gamma is equal to zero")
            key, subkey = random.split(key)
            latents[data_idx] = random.categorical(key = subkey, logits=jnp.log(gamma))
            log_evidence_weights[sis_iter] += jnp.log(jnp.sum(gamma))

        if output_latents:
            latents_ls[sis_iter, :] = latents

    if output_latents:
        return {'evidence': log_evidence_weights, 'latents': latent_ls}
    else:
        return {'evidence': log_evidence_weights }


def gamma_coefficient(idx, state, obs, latents, num_bins, num_states, bin_weight_prior_par, latent_prior_par, is_mixture=True):
    assert idx <= (len(latents) + 1)
    assert idx >= 2
    if is_mixture:
        obs_up_to_idx = jnp.asarray(obs[:, :idx - 1])
        eff_obs = jnp.asarray(obs_up_to_idx[:, latents[:idx - 1] == state])
    else:
        obs_up_to_idx = jnp.asarray(obs[:, :idx - 1]).T
        eff_obs = jnp.asarray(obs_up_to_idx[:, latents[:idx - 1] == state]).T

    log_evidence_new = baseline_log_evidence(jnp.column_stack([eff_obs, obs[:, idx]]), num_bins, bin_weight_prior_par, is_mixture)
    log_evidence_old = baseline_log_evidence(eff_obs, num_bins, bin_weight_prior_par, is_mixture)
    posterior_latent_weight = post_latent_weight(state, latents[:idx-1], num_states, latent_prior_par, start_state=latents[idx-1], is_mixture=is_mixture)
    evidence_ratio = jnp.exp(log_evidence_new - log_evidence_old)
    return evidence_ratio * posterior_latent_weight

def transition_count(latent_states, num_states):
    if len(latent_states) <= 1:
        return jnp.zeros((num_states, num_states))
    sample_size = len(latent_states)
    start_states = latent_states[1:(sample_size-1)]
    end_states = latent_states[2:sample_size]
    transition_count_mat = jnp.zeros((num_states, num_states))
    for i in range(sample_size-1):
        transition_count_mat[start_states[i], end_states[i]] += 1
    return transition_count_mat


def baseline_log_evidence(obs, num_bins, bin_weight_prior_par, is_mixture=True):
    if obs.shape[0] == 0:
        return 0

    if is_mixture:
        obs_dim = obs.shape[0]
        log_evidence = 0
        for dim in range(obs_dim):
            bin_counts = jnp.bincount(obs[dim, :], minlength=num_bins)
            bin_weight_posterior_par = bin_weight_prior_par + bin_counts
            log_evidence += (jnp.sum(lax.lgamma(bin_weight_posterior_par)) - lax.lgamma(jnp.sum(bin_weight_posterior_par))
                             + lax.lgamma(jnp.sum(bin_weight_prior_par)) - jnp.sum(lax.lgamma(bin_weight_prior_par))
                             )
    else:
        bin_counts = jnp.bincount(obs, minlength=num_bins)
        bin_weight_posterior_par = bin_weight_prior_par + bin_counts
        log_evidence = (jnp.sum(lax.lgamma(bin_weight_posterior_par)) - lax.lgamma(jnp.sum(bin_weight_posterior_par))
                        + lax.lgamma(jnp.sum(bin_weight_prior_par)) - jnp.sum(lax.lgamma(bin_weight_prior_par))
                        )
    return log_evidence


def post_latent_weight(state, latents, num_states, latent_prior_par, start_state = None, is_mixture = True):
    if is_mixture:
        state_counts = jnp.bincount(latents, minlength=num_states)
        latent_post_par = latent_prior_par + state_counts
        return latent_post_par[state] / jnp.sum(latent_post_par)
    else:
        # need non-null start state
        if start_state is None:
            raise ValueError("When is_mixture=False, a non-null start_state must be provided")
        transition_counts_from_start_state = transition_count(latent_states = latents, num_states = num_states)[start_state,]
        latent_post_par = latent_prior_par + transition_counts_from_start_state
        return latent_post_par[state] / jnp.sum(latent_post_par)

def log_exponential_mean(x: jnp.array):
    return jax.scipy.special.logsumexp(x) - jnp.log(len(x))