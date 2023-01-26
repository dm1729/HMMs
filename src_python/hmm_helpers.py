# Intended to provide functionality of RHmm package for computation of the forward and backward probabilities
# See "A Gentle Tutorial of the EM Algorithm
# and its Application to Parameter
# Estimation for Gaussian Mixture and
# Hidden Markov Models"
# by Jeff A. Bilmes (bilmes@cs.berkeley.edu)
import jax
import jax.numpy as jnp
from jax import lax


def forward_backward_multinomial(obs_data, trans_mat, emission_mat, init_probs=None):
    # Compute forward probabilities using scan
    def forward_scan_fun(carry, obs_t):  # carry is alpha t minus 1
        alpha_t = jnp.dot(carry, trans_mat) * emission_mat[:, obs_t]
        # alpha_t /= jnp.sum(alpha_t)
        return alpha_t, alpha_t  # so carry is alpha_t, and the y (stored output) is alpha_t

    forward_init = init_probs * emission_mat[:, obs_data[0]]
    _, forward_after_init = lax.scan(f=forward_scan_fun, init=forward_init, xs=obs_data[1:])
    forward = jnp.append(jnp.array([forward_init]), forward_after_init, axis=0)

    # Compute backward probabilities using scan
    def backward_scan_fun(carry, obs_t):
        beta_t = jnp.dot(trans_mat, carry * emission_mat[:, obs_t])
        # beta_t /= jnp.sum(beta_t)
        return beta_t, beta_t

    backward_init = jnp.array([float(1) for _ in range(trans_mat.shape[0])])
    _, backward_after_init = lax.scan(f=backward_scan_fun, init=backward_init, xs=obs_data[1:], reverse=True)
    backward = jnp.append(backward_after_init, jnp.array([backward_init]), axis=0)

    return forward, backward


# %%
def forward_backward_gaussian(obs_data, trans_mat, means, standard_devs, init_probs=None):
    def normal_pdf_vec(obs_, means_, standard_devs_):
        return jnp.array([jax.scipy.stats.norm.pdf(obs_, means_[i], standard_devs_[i]) for i in range(len(means))])

    # Compute forward probabilities using scan
    def forward_scan_fun(carry, obs_t):
        alpha_t = jnp.dot(carry, trans_mat) * normal_pdf_vec(obs_t, means, standard_devs)
        # alpha_t /= jnp.sum(alpha_t)
        return alpha_t, alpha_t  # so carry is alpha_t, and the y (stored output) is alpha_t

    forward_init = init_probs * normal_pdf_vec(obs_data[0], means, standard_devs)
    _, forward_after_init = lax.scan(f=forward_scan_fun, init=forward_init, xs=obs_data[1:])
    forward = jnp.append(jnp.array([forward_init]), forward_after_init, axis=0)

    # Compute backward probabilities using scan
    def backward_scan_fun(carry, obs_t):
        beta_t = jnp.dot(trans_mat, carry * normal_pdf_vec(obs_t, means, standard_devs))
        # beta_t /= jnp.sum(beta_t)
        return beta_t, beta_t

    backward_init = jnp.array([float(1) for _ in range(trans_mat.shape[0])])
    _, backward_after_init = lax.scan(f=backward_scan_fun, init=backward_init, xs=obs_data[1:], reverse=True)
    backward = jnp.append(backward_after_init, jnp.array([backward_init]), axis=0)

    return forward, backward


# Need to modify so I can get the full matrix out
def conditional_probability(time, state, obs_data, forward_backward_kwargs,
                            is_multinomial=False, is_gaussian=False, output_forward_backward=False):
    # computes the gamma function which is prob of given hidden state at given time, given obs
    if is_multinomial:
        forward, backward = forward_backward_multinomial(obs_data, **forward_backward_kwargs)
    elif is_gaussian:
        forward, backward = forward_backward_gaussian(obs_data, **forward_backward_kwargs)
    else:
        raise (ValueError('Must set either multinomial or Gaussian flag to zero'))
    if output_forward_backward:
        return forward[time][state] * backward[time][state] / jnp.sum(forward[time] * backward[time]), forward, backward
    else:
        return forward[time][state] * backward[time][state] / jnp.sum(forward[time] * backward[time])


# Need to modify so I can get the full three-way array output
def conditional_transition(start_time, start_state, obs_data, forward_backward_kwargs,
                           is_multinomial=True, is_gaussian=False):
    if is_multinomial:
        conditional_prob, _, backward = conditional_probability(time=start_time, state=start_state, obs_data=obs_data,
                                                                **forward_backward_kwargs, is_multinomial=True)
    elif is_gaussian:
        conditional_prob, _, backward = conditional_probability(time=start_time, state=start_state, obs_data=obs_data,
                                                                **forward_backward_kwargs, is_gaussian=True)
    else:
        raise (ValueError('Must set either multinomial or Gaussian flag to zero'))

    conditional_transition
