home_dir <- ''
source(paste0(home_dir, './src/data_simulator.R'))
source(paste0(home_dir, './src/hmm_mcmc.R'))
source(paste0(home_dir, './src/marginal_likelihood.R'))

# Basic example

## setting parameters
true_states <- 2
trans_mat <- matrix( c(0.7,0.3,0.2,0.8), nrow=true_states, ncol=true_states, byrow = TRUE )
normal_mean <- c(-1,1)
normal_var <- c(1,1)
num_samples <- 100

## generating real-valued observations
real_obs_data <- simul_normal_hmm(trans_mat = trans_mat, normal_mean = normal_mean,
                        normal_var = normal_var, num_samples = num_samples)$obs

## transforming to [0,1]
transformed_data <- truncated_inv_logit(real_obs_data)

## binning
num_bins <- 4
binned_data <- uniformly_bin(transformed_data, num_bins = num_bins)

## parallel sis algo

sis_iters <- 100
num_hidden_states <- 2 # number of states to fit

num_cores <- parallel::detectCores()
doParallel::registerDoParallel(cores=num_cores)
iters_per_core <- ceiling(sis_iters/num_cores)

system.time(
  sis_output <- unlist(foreach (i= 1:num_cores )%dopar%{
    sis_estimator(obs=binned_data,num_bins=num_bins,iters=iters_per_core, num_hidden_states,
                  bin_weight_prior_par = NULL,
                  latent_prior_par = NULL, is_mixture = FALSE)$evidence
  })
)

system.time(
  sis_output <- sis_estimator(obs=binned_data,num_bins=num_bins,iters=sis_iters, num_hidden_states,
                              bin_weight_prior_par = NULL,
                              latent_prior_par = NULL, is_mixture = FALSE)$evidence
)
