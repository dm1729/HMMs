import jax.numpy as jnp
from jax import vmap


def uniformly_bin(obs_data, num_bins, link=None):
    if link is not None:
        obs_data = link(obs_data)
    binned_data = jnp.floor(num_bins * obs_data)  # so binned_data is in 0, 1, ..., num_bins - 1, i.e. range(num_bins)
    binned_data[binned_data == num_bins] = num_bins - 1
    return binned_data


def truncated_inv_logit_base(x, lower=-2, upper=2):
    assert lower <= upper
    if lower == upper:
        y = 1 / (1 + jnp.exp(-x))
    elif x < lower:
        y = 1 / (1 + jnp.exp(-x))
    elif x > upper:
        y = 1 / (1 + jnp.exp(-x))
    else:
        y = 1 / (1 + jnp.exp(-lower)) + ((x - lower) / (upper - lower)) * (
                1 / (1 + jnp.exp(-upper)) - 1 / (1 + jnp.exp(-lower)))
    return y


truncated_inv_logit = vmap(truncated_inv_logit_base, in_axes=(None, 0, 0))
