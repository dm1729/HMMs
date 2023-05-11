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

### Overview of hmm_helpers.py

This contains helper functions for the HMM code, to replace the RHmm package in R. This includes functions for forward/backward and associated quantities, and is implemented in JAX.

## Other files

spectral_estimation.py: Code for estimation of number of states via tensor approximation. For comparison with marginal likelihood.

marginal_likelihood.py: Implementation of SIS algo of Hairault et al. (2022) https://arxiv.org/abs/2205.05416, for the multinomial HMM, for estimation of marginal likelihood for model selection.