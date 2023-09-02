import jax.numpy as jnp
from jax import random
import jax
from functools import partial


def simulate_hmm(num_obs,transition_matrix, means=None, variances=None,
                 emission_probabilities=None, initial_distribution=None,
                 seed=0):
    n_states = transition_matrix.shape[0]
    # Check if shapes of inputs are consistent
    if transition_matrix.shape[0] != n_states:
        raise ValueError(f"Number of rows in transition matrix should be {n_states}")
    if transition_matrix.shape[1] != n_states:
        raise ValueError(f"Number of columns in transition matrix should be {n_states}")
    if means is not None and variances is not None:
        if len(means) != n_states or len(variances) != n_states:
            raise ValueError(f"Number of means and variances should be {n_states}")
    if emission_probabilities is not None:
        if emission_probabilities.shape[0] != n_states:
            raise ValueError(f"Number of rows in observed_categories should be {n_states}")
    # Compute stationary distribution if no initial distribution is provided
    if initial_distribution is None:
        with jax.default_device(jax.devices("cpu")[0]): # eig only implemented on cpu backend
            leading_eigenvector = jnp.linalg.eig(transition_matrix.T)[1][:, 0]
        initial_distribution = jnp.abs(leading_eigenvector) / jnp.sum(jnp.abs(leading_eigenvector))
    # Set random seed
    key = random.PRNGKey(seed)
    hidden_chain = [random.categorical(key, initial_distribution)]
    observations = []
    key, subkey = random.split(key)
    if means is not None and variances is not None:
        observations.append(random.normal(subkey, means[hidden_chain[0]], variances[hidden_chain[0]]))
    elif emission_probabilities is not None:
        observations.append(random.categorical(subkey, emission_probabilities[hidden_chain[0]]))
    else:
        raise ValueError("Either means and variances or observed_categories should be provided")
    for i in range(1, num_obs):
        key, subkey = random.split(key)
        hidden_chain.append(random.categorical(subkey, transition_matrix[hidden_chain[i - 1]]))
        key, subkey = random.split(key)
        if means is not None and variances is not None:
            observations.append(random.normal(subkey, means[hidden_chain[i]], variances[hidden_chain[i]]))
        elif emission_probabilities is not None:
            observations.append(random.categorical(subkey, emission_probabilities[hidden_chain[i]]))
    return jnp.array(hidden_chain), jnp.array(observations)


# def simulate_mixture(num_obs, num_components, component_probs,
#                      means=None, variances=None, emission_probabilities=None, seed=0,mixture_dim=3):
#     # Check that inputs have consistent shapes
#     if means is not None and variances is not None:
#         if means.shape[0] != variances.shape[0]:
#             raise ValueError("means and variances should have the same number of rows")
#         if means.shape[1] != variances.shape[1]:
#             raise ValueError("means and variances should have the same number of columns")
#     if emission_probabilities is not None:
#         if emission_probabilities.shape[0] != num_components:
#             raise ValueError("emission_probabilities should have the same number of rows as number of components")

#     # Set random seed
#     key = random.PRNGKey(seed)
#     latents = []
#     observations = [ [] for _ in range(mixture_dim) ]
#     key, subkey = random.split(key)
#     for _ in range(num_obs):
#         key, subkey = random.split(key)
#         latent = random.categorical(subkey, component_probs)
#         latents.append(latent)
#         key, subkey = random.split(key)
#         if means is not None and variances is not None:
#             observations.append(random.normal(subkey, means[latent], variances[latent]))
#         elif emission_probabilities is not None:
#             observations.append(random.categorical(subkey, emission_probabilities[latent]))
#     return jnp.array(latents), jnp.array(observations)


# def simulate_mixture(num_obs, num_components, component_probs,
#                      means=None, variances=None, emission_probabilities=None, seed=0,obs_dim=3):
#     """
#     Generates data from a mixture model.

#     Parameters:
#     - num_obs: int, number of observations
#     - num_components: int, number of mixture components
#     - component_probs: array_like, mixing probabilities for each component
#     - means, variances: (optional) parameters for Gaussian emissions
#     - emission_probabilities: (optional) parameters for categorical emissions
#     - seed: int, random seed

#     Returns:
#     - jnp.array(latents), jnp.array(observations)
#     """
#     # Set random seed
#     key = random.PRNGKey(seed)
    
#     # Generate latent states
#     key, subkey = random.split(key)
#     latents = random.choice(subkey, num_components, shape=(num_obs,), p=component_probs)
    
#     # Generate oservations

#     observations = []
#     for i in range(num_obs):
#         key, *subkeys = random.split(key, num=obs_dim+1)
#         if means is not None and variances is not None:
#             obs_i = jnp.stack([random.normal(subkeys[j], loc=means[latents[i], j], scale=variances[latents[i], j]) for j in range(obs_dim)])
#         elif emission_probabilities is not None:
#             obs_i = jnp.stack([random.choice(subkeys[j], emission_probabilities.shape[1], p=emission_probabilities[latents[i], :, j]) for j in range(obs_dim)])
#         observations.append(obs_i)
#
#     return jnp.array(latents), jnp.array(observations)

def simulate_mixture(num_obs, num_components, component_probs,
                     means=None, variances=None, emission_probabilities=None, seed=0,obs_dim=3):
    """
    Generates data from a mixture model.

    Parameters:
    - num_obs: int, number of observations
    - num_components: int, number of mixture components
    - component_probs: array_like, mixing probabilities for each component
    - means, variances: (optional) parameters for Gaussian emissions
    - emission_probabilities: (optional) parameters for categorical emissions
    - seed: int, random seed

    Returns:
    - jnp.array(latents), jnp.array(observations)
    """
    # Set random seed
    key = random.PRNGKey(seed)
    
    # Generate latent states
    key, subkey = random.split(key)
    latents = random.choice(subkey, num_components, shape=(num_obs,), p=component_probs)
    
    # Generate oservations
    key, *obs_keys = random.split(key, num=num_obs+1)
    obs_keys = jnp.array(obs_keys) # required for vmapping
    num_bins = emission_probabilities.shape[1]
    
    @partial(jax.jit, static_argnums=(3,4))
    def generate_gaussian_obs(latent, key, means, variances, obs_dim):
        dim_keys = random.split(key, num=obs_dim)
        single_obs = jnp.stack([random.normal(dim_keys[j], loc=means[latent, j], scale=variances[latent, j]) for j in range(obs_dim)])
        return single_obs
    @partial(jax.jit, static_argnums=(3,4))
    def generate_categorical_obs(latent, key, emission_probabilities, num_bins, obs_dim):
        dim_keys = random.split(key, num=obs_dim)
        single_obs = jnp.stack([random.choice(dim_keys[j],
                                         num_bins,
                                         p=emission_probabilities[latent, :, j]) for j in range(obs_dim)])
        return single_obs
    
    if means is not None and variances is not None:
        vmap_gaussian = jax.vmap(partial(generate_gaussian_obs,
                                         means=means, variances=variances, obs_dim=obs_dim))
        observations = vmap_gaussian(latents, obs_keys)
    elif emission_probabilities is not None:
        vmap_categorical = jax.vmap(partial(generate_categorical_obs,
                                            emission_probabilities=emission_probabilities,num_bins=num_bins, obs_dim=obs_dim))
        observations = vmap_categorical(latents, obs_keys)
    else:
        raise ValueError("Either means and variances or emission_probabilities should be provided")


    return jnp.array(latents), jnp.array(observations)
