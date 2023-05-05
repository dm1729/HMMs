# Intended to provide functionality of RHmm package for computation of the forward and backward probabilities
# See "A Gentle Tutorial of the EM Algorithm
# and its Application to Parameter
# Estimation for Gaussian Mixture and
# Hidden Markov Models"
# by Jeff A. Bilmes (bilmes@cs.berkeley.edu)
import jax
import jax.numpy as jnp
from jax import lax, vmap, jit
from functools import partial

@jit
def stationary_distribution_power_iteration(trans_mat, max_iter=1000, tol=1e-6):
    """
    Computes the stationary distribution of a transition matrix using the power iteration method.
    Implemented to avoid using CPU backend for eigenvector computation, as not implemented for GPU.

    Args:
        trans_mat (array-like): Transition matrix of the HMM (shape: [n_states, n_states])
        max_iter (int, optional): Maximum number of iterations for the power iteration method (default: 1000)
        tol (float, optional): Tolerance for convergence (default: 1e-6)

    Returns:
        jax.numpy.array: Array of stationary probabilities (shape: [n_states])
    """
    init_state_probs = jnp.ones(trans_mat.shape[0]) / trans_mat.shape[0]

    def cond_fun(val):
        state_probs, i = val
        return (i < max_iter) & (jnp.linalg.norm(jnp.dot(state_probs, trans_mat) - state_probs) >= tol)

    def body_fun(val):
        state_probs, i = val
        return jnp.dot(state_probs, trans_mat), i + 1

    stationary_probs = jax.lax.while_loop(cond_fun, body_fun, (init_state_probs, 0))[0]
    return stationary_probs

# def stationary_distribution_power_iteration(trans_mat, max_iter=1000, tol=1e-6):
#     """
#     Computes the stationary distribution of a transition matrix using the power iteration method.

#     Args:
#         trans_mat (array-like): Transition matrix of the HMM (shape: [n_states, n_states])
#         max_iter (int, optional): Maximum number of iterations for the power iteration method (default: 1000)
#         tol (float, optional): Tolerance for convergence (default: 1e-6)

#     Returns:
#         jax.numpy.array: Array of stationary probabilities (shape: [n_states])
#     """
#     state_probs = jnp.ones(trans_mat.shape[0]) / trans_mat.shape[0]

#     for _ in range(max_iter):
#         next_state_probs = jnp.dot(state_probs, trans_mat)
#         if jnp.linalg.norm(next_state_probs - state_probs) < tol:
#             break
#         state_probs = next_state_probs
#     return state_probs

def compute_emission_probs_multinomial(obs_t, emission_mat):
    """
    Computes the emission probabilities for a multinomial HMM for the given observation index and emission matrix.

    Args:
        obs_t (array-like): The observation indices at the current timestep
        emission_mat (array-like): Emission matrix of the HMM (shape: [n_states, n_emissions])

    Returns:
        jax.numpy.array: Array of emission probabilities (shape: [n_states] if obs_t is int, [n_states, len(obs_t)] if obs_t is array-like)
    """
    return emission_mat[:, obs_t]

def compute_emission_probs_gaussian(obs_t, means, standard_devs):
    """
    Computes the Gaussian probability density function (pdf) for each mean and standard deviation pair for the given observation.

    Args:
        obs_t (float or array-like): The observation value(s) for which to compute the pdf
        means (array-like): Array of means (shape: [n_states])
        standard_devs (array-like): Array of standard deviations (shape: [n_states])

    Returns:
        jax.numpy.array: Array of Gaussian pdf values (shape: [n_states]) if obs_t is a float,
                          or 2D array (shape: [n_states, len(obs_t)]) if obs_t is an array-like
    """
    pdf_single = lambda mean, std_dev: jax.scipy.stats.norm.pdf(obs_t, mean, std_dev)
    return vmap(pdf_single)(means, standard_devs)

@partial(jit, static_argnums=(2,)) # emission_func is static argument, so will be compiled once and reused
def forward_backward(obs_data, trans_mat, emission_func): # TODO: Update calls to partially evaluate emission_func to include emission_kwargs
    """
    Computes the forward and backward probabilities for a Hidden Markov Model (HMM) with the given emission function.

    Args:
        obs_data (array-like): Array of observed data (sequence of emission values)
        trans_mat (array-like): Transition matrix of the HMM (shape: [n_states, n_states])
        emission_func (function): Function to compute emission probabilities given the carry and observation.
        Should be partially evaluated ahead of time.

    Returns:
        tuple: Forward and backward probabilities (each a 2D array of shape [n_timesteps, n_states])
    """
    # Compute forward probabilities using scan
    def forward_scan_fun(carry, obs_t):
        log_norm_sum, alpha_prev = carry
        # Compute emission probabilities using the emission function
        emission_probs = emission_func(obs_t)
        # Calculate alpha at t using carry (alpha at t-1), transition matrix, and emission probabilities
        alpha_t = jnp.dot(alpha_prev, trans_mat) * emission_probs
        # Normalize alpha_t to avoid underflow issues
        alpha_sum = jnp.sum(alpha_t)
        alpha_t /= alpha_sum
        log_norm_sum += jnp.log(alpha_sum)
        # Return alpha_t as both the carry and output for this step, with log like in carry
        return (log_norm_sum, alpha_t), alpha_t

    # Compute initial probabilities if not provided
    # if init_probs is None:
    #     # with jax.default_device(jax.devices("cpu")[0]):  # eig only implemented on CPU backend
    #     #     leading_eigenvector = jnp.linalg.eig(trans_mat.T)[1][:, 0]
    #     # init_probs = jnp.abs(leading_eigenvector) / jnp.sum(jnp.abs(leading_eigenvector))
    #     init_probs = stationary_distribution_power_iteration(trans_mat)
    init_probs = stationary_distribution_power_iteration(trans_mat)
    # Compute initial forward probabilities (alpha at t=0) using initial probabilities and the first observation
    forward_init = init_probs * emission_func(obs_data[0])
    alpha_sum_init = jnp.sum(forward_init)
    forward_init /= alpha_sum_init
    log_norm_sum_init = jnp.log(alpha_sum_init)
    # Compute forward probabilities for the remaining timesteps using scan
    (log_likelihood,_), forward_after_init = lax.scan(f=forward_scan_fun, init=(log_norm_sum_init,forward_init), xs=obs_data[1:])
    # Combine initial forward probabilities with the rest of the forward probabilities
    forward = jnp.append(jnp.array([forward_init]), forward_after_init, axis=0)

    # Compute backward probabilities using scan
    def backward_scan_fun(carry, obs_t):
        # Compute emission probabilities using the emission function
        emission_probs = emission_func(obs_t)
        # Calculate beta at t using carry (beta at t+1), transition matrix, and emission probabilities
        beta_t = jnp.dot(trans_mat, carry * emission_probs)
        # Normalize beta_t to avoid underflow issues
        beta_t /= jnp.sum(beta_t)
        # Return beta_t as both the carry and output for this step
        return beta_t, beta_t

    # Initialize backward probabilities (beta at last timestep) as a uniform vector
    backward_init = jnp.ones_like(init_probs)
    # Compute backward probabilities for the remaining timesteps using scan (in reverse order)
    _, backward_after_init = lax.scan(f=backward_scan_fun, init=backward_init, xs=obs_data[1:], reverse=True)
    # Combine initial backward probabilities with the rest of the backward probabilities
    backward = jnp.append(backward_after_init, jnp.array([backward_init]), axis=0)

    return forward, backward, log_likelihood

@jit
def conditional_probability(forward, backward):
    """
    Computes the conditional probability (gamma) of the hidden state at each time point, given the observations and HMM parameters.
    Optionally provide forward and backward probabilities instead of the observed data and transition mat / emissions.

    Args:
        obs_data: A jnp array containing the observed data.
        trans_mat: A jnp array containing the transition probabilities between hidden states.
        emission_func: A function that takes the observed data as input and returns the emission probabilities for each hidden state.
        emission_kwargs: A dictionary containing keyword arguments for the emission function.
        output_forward_backward: A boolean flag indicating whether to output the forward and backward probabilities in addition to the conditional probabilities. Defaults to False.
        forward: forward probabilities. If provided with backward, then does not compute forward and backward probabilities from other args.
        backward: backward probabilities. If provided with forward, then does not compute forward and backward probabilities from other args.

    Returns:
        jax.numpy.array: A 2D array of shape [n_timesteps, n_states] containing the conditional probabilities of the hidden state at each time point, given the observations and HMM parameters.
        If output_forward_backward is True, returns a tuple containing the conditional probabilities, forward probabilities, and backward probabilities.
    """
    # assert obs_data is not None, "obs_data must be provided if forward and backward are not provided."
    # assert trans_mat is not None, "trans_mat must be provided if forward and backward are not provided."
    # assert emission_func is not None, "emission_func must be provided if forward and backward are not provided."
    # assert emission_kwargs is not None, "emission_kwargs must be provided if forward and backward are not provided."
    cond_prob = jnp.array(forward) * jnp.array(backward) / jnp.sum(jnp.array(forward) * jnp.array(backward), axis=1, keepdims=True)

    return cond_prob

@partial(jit, static_argnums=(4,))
def joint_conditional_probabilities(obs_data, trans_mat, forward, backward, emission_func):
    """
    Computes the joint conditional probabilities (xi) for an HMM given the observations.

    Args:
        obs_data: A numpy array containing the observed data.
        trans_mat: A numpy array containing the transition probabilities between hidden states.
        emission_func: A function that takes the observed data as input and returns the emission probabilities for each hidden state.
            Any other arguments must be partially applied to the function before passing it to this function.

    Returns:
        jax.numpy.array: A 3D array containing the joint conditional probabilities, indexed as t (time), i (time t state), j (time t+1 state)
    """
    conditional_prob = conditional_probability(forward,backward)
    likelihood_term = emission_func(obs_data[1:])
    joint_cond_probs = jnp.einsum("ti,ij,jt,tj ,ti -> tij", conditional_prob[:-1,:], trans_mat,
                                               likelihood_term, backward[1:,:], (1 / backward[:-1,:]))
    joint_cond_probs/= jnp.sum(joint_cond_probs , axis = (1,2), keepdims=True)
    # Since backward probs are renormalised to avoid underflow, the time t transitions do not sum to one
    # This normalisation factor depends on t, but not on i or j. So we renormalise the tensor t by t (sum over i,j)
    return joint_cond_probs

# def normal_pdf_vec(obs_, means_, standard_devs_):
#     """
#     Compute the Gaussian probability density function (pdf) for each mean and standard deviation pair, given the observation.

#     Parameters:
#     obs_ (float): The observation value for which to compute the pdf
#     means_ (array-like): Array of means (shape: [n_states])
#     standard_devs_ (array-like): Array of standard deviations (shape: [n_states])

#     Returns:
#     jax.numpy.array: Array of Gaussian pdf values (shape: [n_states])
#     """
#     pdf_single = lambda mean, std_dev: jax.scipy.stats.norm.pdf(obs_, mean, std_dev)
#     return vmap(pdf_single)(means_, standard_devs_)


# def forward_backward_multinomial(obs_data, trans_mat, emission_mat, init_probs=None):
#     """
#     Compute forward and backward probabilities for a discrete Hidden Markov Model (HMM) with multinomial emissions.
    
#     Parameters:
#     obs_data (array-like): Array of observed data (sequence of emission indices)
#     trans_mat (array-like): Transition matrix of the HMM (shape: [n_states, n_states])
#     emission_mat (array-like): Emission matrix of the HMM (shape: [n_states, n_emissions])
#     init_probs (array-like, optional): Initial state probabilities (default: stationary distribution)
    
#     Returns:
#     tuple: Forward and backward probabilities (each a 2D array of shape [n_states, n_timesteps])
#     """

#     # Compute forward probabilities using scan
#     def forward_scan_fun(carry, obs_t):
#         # carry is alpha at t minus 1
#         # Calculate alpha at t using carry (alpha at t-1), transition matrix, and emission matrix
#         alpha_t = jnp.dot(carry, trans_mat) * emission_mat[:, obs_t]
#         # Normalize alpha_t to avoid underflow issues
#         alpha_t /= jnp.sum(alpha_t)
#         # Return alpha_t as both the carry and output for this step
#         return alpha_t, alpha_t

#     # Compute initial probabilities if not provided
#     if init_probs is None:
#         with jax.default_device(jax.devices("cpu")[0]):  # eig only implemented on CPU backend
#             leading_eigenvector = jnp.linalg.eig(trans_mat.T)[1][:, 0]
#         init_probs = jnp.abs(leading_eigenvector) / jnp.sum(jnp.abs(leading_eigenvector))
    
#     # Compute initial forward probabilities (alpha at t=0) using initial probabilities and the first observation
#     forward_init = init_probs * emission_mat[:, obs_data[0]] / jnp.sum(init_probs * emission_mat[:, obs_data[0]])
#     # Compute forward probabilities for the remaining timesteps using scan
#     _, forward_after_init = lax.scan(f=forward_scan_fun, init=forward_init, xs=obs_data[1:])
#     # Combine initial forward probabilities with the rest of the forward probabilities
#     forward = jnp.append(jnp.array([forward_init]), forward_after_init, axis=0)

#     # Compute backward probabilities using scan
#     def backward_scan_fun(carry, obs_t):
#         # Calculate beta at t using carry (beta at t+1), transition matrix, and emission matrix
#         beta_t = jnp.dot(trans_mat, carry * emission_mat[:, obs_t])
#         # Normalize beta_t to avoid underflow issues
#         beta_t /= jnp.sum(beta_t)
#         # Return beta_t as both the carry and output for this step
#         return beta_t, beta_t

#     # Initialize backward probabilities (beta at last timestep) as a uniform vector
#     backward_init = jnp.ones(trans_mat.shape[0])
#     # Compute backward probabilities for the remaining timesteps using scan (in reverse order)
#     _, backward_after_init = lax.scan(f=backward_scan_fun, init=backward_init, xs=obs_data[1:], reverse=True)
#     # Combine initial backward probabilities with the rest of the backward probabilities
#     backward = jnp.append(backward_after_init, jnp.array([backward_init]), axis=0)

#     return forward, backward


# def forward_backward_gaussian(obs_data, trans_mat, means, standard_devs, init_probs=None):
#     """
#     Compute forward and backward probabilities for a continuous Hidden Markov Model (HMM) with Gaussian emissions.
    
#     Parameters:
#     obs_data (array-like): Array of observed data (sequence of continuous emission values)
#     trans_mat (array-like): Transition matrix of the HMM (shape: [n_states, n_states])
#     means (array-like): Array of means for the Gaussian emissions (shape: [n_states])
#     standard_devs (array-like): Array of standard deviations for the Gaussian emissions (shape: [n_states])
#     init_probs (array-like, optional): Initial state probabilities (default: stationary distribution)
    
#     Returns:
#     tuple: Forward and backward probabilities (each a 2D array of shape [n_timesteps, n_states])
#     """
#     # Compute forward probabilities using scan
#     def forward_scan_fun(carry, obs_t):
#         alpha_t = jnp.dot(carry, trans_mat) * normal_pdf_vec(obs_t, means, standard_devs)
#         alpha_t /= jnp.sum(alpha_t)
#         return alpha_t, alpha_t  # so carry is alpha_t, and the y (stored output) is alpha_t
    
#     if init_probs is None:
#         with jax.default_device(jax.devices("cpu")[0]): # eig only implemented on cpu backend
#             leading_eigenvector = jnp.linalg.eig(trans_mat.T)[1][:, 0]
#         init_probs = jnp.abs(leading_eigenvector) / jnp.sum(jnp.abs(leading_eigenvector))
#     forward_init = init_probs * normal_pdf_vec(obs_data[0], means, standard_devs)
#     _, forward_after_init = lax.scan(f=forward_scan_fun, init=forward_init, xs=obs_data[1:])
#     forward = jnp.append(jnp.array([forward_init]), forward_after_init, axis=0)

#     # Compute backward probabilities using scan
#     def backward_scan_fun(carry, obs_t):
#         beta_t = jnp.dot(trans_mat, carry * normal_pdf_vec(obs_t, means, standard_devs))
#         beta_t /= jnp.sum(beta_t)
#         return beta_t, beta_t

#     backward_init = jnp.ones(trans_mat.shape[0])
#     _, backward_after_init = lax.scan(f=backward_scan_fun, init=backward_init, xs=obs_data[1:], reverse=True)
#     backward = jnp.append(backward_after_init, jnp.array([backward_init]), axis=0)

#     return forward, backward