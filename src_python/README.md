# Python code for projects relating to HMMs

## Translation of src_R code (original) to python implementation with JAX

### Overview of binned sampler (see hmm_mcmc.py, using also hmm_helpers.py)

The R code (in src_R) using RHmm package (which uses Rcpp for forward/backward) is more efficient for small numbers of states, approx. 5x faster.
However, a non-parallel implementation of forward/backward scales much worse with the number of states (eventually quadratic)
The JAX-based GPU implementation has a compile-time scaling with the states, but the compute time hardly increases until the number of states is quite large (>100). This is because the forward-backward complexity relating to the states is driven by matrix multiplications. The compile-time does not scale with the number of iterations, and so for chains of reasonable length the compile time is negligible.

E.g. for constant size of data (2000), number of multinomial bins (4), we had the following timings for 100 states:
* Approx. 1200s for 1000 iters using R implementation
* Approx. 215s for 1000 iters using JAX implementation (approx. 120s for compilation, 95s for execution)

Note that if we had done much more iterations, the compilation time would be neglibible. and we would get approx a 12x speedup in JAX.

For 2 states, we instead had
* Approx 15s for 1000 iters using R implementation
* Approx 75s for 1000 iters (2s compile time, 73s execution)

Note that the Rcpp implementation is approx 5x faster for 2 states.

The point at which both perform similarly occurs at around 20 states. At this point we get:
* Approx 70s for 1000 iters using R implementation
* Approx 100s for 1000 iters (15s compile time, 85s execution)

The JAX implementation could be especially useful when using HMMs with mixtures as emissions, as we get a larger effective state space in sampling.

### Update: Using CPU backend

When using few states, it is reccomended simply use

```
with jax.default_device(jax.devices("cpu")[0]):
    sampler_out = binned_prior_sampler(...)
```

to use the CPU backend. This is much faster than using the GPU backend for small numbers of states (approx. 100x faster than GPU for 2 states)

For 2 states and 2000 samples, we had
* Approx 3.6s for 1000 iters (3s compile time, 0.6s execution)

And so actually the JAX CPU backend seems to get around a 25x speedup over the Rcpp implementation for 2 states, excluding compilation time.

When using the CPU backend for 50 states, we got
* Approx 98s for 1000 iters (84s compile time, 14s execution)

The GPU backend for 50 states compiled in 42s, and the CPU backend actually failed compilation for 100 states (which the GPU did compile successfully).
Generally CPU compilation was slow for large numbers of states and seemed a bit erratic too.


Overall, the GPU implementation is useful for large number of states (as arises if using mixtures in emissions). The GPU implementation could potentially also be vmapped over multiple different parameter configurations, and so this will contribute to the choice of backend when using small numbers of states.


### Overview of hmm_helpers.py

This contains helper functions for the HMM code, to replace the RHmm package in R. This includes functions for forward/backward and associated quantities, and is implemented in JAX.

## Other files

spectral_estimation.py: Code for estimation of number of states via tensor approximation. For comparison with marginal likelihood.

marginal_likelihood.py: Implementation of SIS algo of Hairault et al. (2022) https://arxiv.org/abs/2205.05416, for the multinomial HMM, for estimation of marginal likelihood for model selection.