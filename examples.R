# Examples
source('data_simulator.R')
source('hmm_mcmc.R')
source('marginal_likelihood.R')

# Selection of number of states - multinomial emissions:

## Multidimensional mixture data

seed_no <- 123
set.seed(seed = seed_no)

num_samples <- 100
mix_weights <- c(0.6,0.4)

# need num_states rows, num_bins columns

num_bins <- 4

emission_mat_dim1 <- gtools::rdirichlet(2,rep(1,num_bins))
  # for each dimension, the emission distribution for each state is uniform on simplex
emission_mat_dim2 <- gtools::rdirichlet(2,rep(1,num_bins))
emission_mat_dim3 <- gtools::rdirichlet(2,rep(1,num_bins))
emission_mat_list <- list(emission_mat_dim1,emission_mat_dim2,emission_mat_dim3)

obs_data <- simul_multinom_mixture(mix_weights,emission_mat_list, num_samples)

# Sis sampler with uniform prior

num_hidden_states <- 2 # 2 is the underlying truth
sis_iters <- 100
bin_weight_prior_par <- NULL
latent_prior_par <- NULL

sis_output <- sis_estimator(obs=obs_data,num_bins=num_bins,iters=sis_iters, num_hidden_states,
                          bin_weight_prior_par = bin_weight_prior_par,
                          latent_prior_par = latent_prior_par, is_mixture = TRUE)

## HMM data

seed_no <- 123
set.seed(seed = seed_no)

num_samples <- 100
trans_mat <- matrix( c(0.7,0.2,0.3,0.8), 2, 2)

# need num_states rows, num_bins columns

num_bins <- 4

emission_mat <- gtools::rdirichlet(2,rep(1,num_bins))

obs_data <- simul_multinom_hmm(trans_mat=trans_mat,emission_mat=emission_mat, num_samples=num_samples)$obs

# Sis sampler with uniform prior

num_hidden_states <- 3 # 2 is the underlying truth
sis_iters <- 100
bin_weight_prior_par <- NULL
latent_prior_par <- NULL

sis_output <- sis_estimator(obs=obs_data,num_bins=num_bins,iters=sis_iters, num_hidden_states=num_hidden_states,
                            bin_weight_prior_par = bin_weight_prior_par,
                            latent_prior_par = latent_prior_par, is_mixture = FALSE)

# Additional complexity from binning
