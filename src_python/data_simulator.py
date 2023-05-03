import jax.numpy as jnp
from jax import random
import jax


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


def simulate_mixture(num_obs, num_components, component_probs,
                     means=None, variances=None, observed_categories=None, seed=0):
    # Check that inputs have consistent shapes
    if means is not None and variances is not None:
        if means.shape[0] != variances.shape[0]:
            raise ValueError("means and variances should have the same number of rows")
        if means.shape[1] != variances.shape[1]:
            raise ValueError("means and variances should have the same number of columns")
    if observed_categories is not None:
        if observed_categories.shape[0] != num_components:
            raise ValueError("observed_categories should have the same number of rows as number of components")

    # Set random seed
    key = random.PRNGKey(seed)
    latents = []
    observations = []
    key, subkey = random.split(key)
    for i in range(num_obs):
        key, subkey = random.split(key)
        latent = random.categorical(subkey, component_probs)
        latents.append(latent)
        key, subkey = random.split(key)
        if means is not None and variances is not None:
            observations.append(random.normal(subkey, means[latent], variances[latent]))
        elif observed_categories is not None:
            observations.append(random.categorical(subkey, observed_categories[latent]))
    return latents, observations
