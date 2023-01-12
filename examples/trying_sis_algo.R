# Examples
# home_dir <- '/home/moss/Documents/HMMs/'
home_dir <- ''
source(paste0(home_dir, '../src/data_simulator.R'))
source(paste0(home_dir, '../src/hmm_mcmc.R'))
source(paste0(home_dir, '../src/marginal_likelihood.R'))

# Selection of number of states - multinomial emissions:

# ## Multidimensional mixture data
#
# num_samples <- 1000
# mix_weights <- c(0.6,0.4)
# true_mix_comps <- length(mix_weights)
#
# # need num_states rows, num_bins columns
#
# num_bins <- 4
#
# seed_no <- 123
# set.seed(seed = seed_no)
# emission_mat_dim1 <- gtools::rdirichlet(2,rep(1,num_bins))
#   # for each dimension, the emission distribution for each state is uniform on simplex
# emission_mat_dim2 <- gtools::rdirichlet(2,rep(1,num_bins))
# emission_mat_dim3 <- gtools::rdirichlet(2,rep(1,num_bins))
# emission_mat_list <- list(emission_mat_dim1,emission_mat_dim2,emission_mat_dim3)
#
# obs_data <- simul_multinom_mixture(mix_weights,emission_mat_list, num_samples)
#
# # Sis sampler with uniform prior
#
# num_hidden_states_ls <- c(2,3,4) # 2 is the underlying truth
# sis_iters <- 100
# bin_weight_prior_par <- NULL
# latent_prior_par <- NULL
# df <- NULL
# hdf5_filepath <- paste0(home_dir,'marginal_likelihood_data.h5')
# hdf5_key <- paste0('multinomial_mix_',num_samples,
#                    '_samples_',num_bins,'_bins_',
#                    true_mix_comps,'_comps')
#
# for (num_hidden_states in num_hidden_states_ls){
#   print(paste(Sys.time(),':','Commencing SIS procedure for',num_hidden_states,'hidden states'))
#   sis_output <- sis_estimator(obs=obs_data,num_bins=num_bins,iters=sis_iters, num_hidden_states,
#                               bin_weight_prior_par = bin_weight_prior_par,
#                               latent_prior_par = latent_prior_par, is_mixture = TRUE)
#   if (is.null(df)){
#     df <- data.frame(sis_output$evidence)
#     df_cols <- paste0('log_evid_',num_hidden_states,'_states')
#     colnames(df) <- df_cols
#   } else {
#     df <- cbind(df, data.frame(sis_output$evidence) )
#     df_cols <- c(df_cols, paste0('log_evid_',num_hidden_states,'_states'))
#     colnames(df) <- df_cols
#   }
# }
# rhdf5::h5write(df, file = hdf5_filepath, name = hdf5_key, native = TRUE)
#
# ## HMM data
#
# seed_no <- 123
# set.seed(seed = seed_no)
#
# num_samples <- 100
# trans_mat <- matrix( c(0.7,0.2,0.3,0.8), 2, 2)
#
# # need num_states rows, num_bins columns
#
# num_bins <- 4
#
# emission_mat <- gtools::rdirichlet(2,rep(1,num_bins))
#
# obs_data <- simul_multinom_hmm(trans_mat=trans_mat,emission_mat=emission_mat, num_samples=num_samples)$obs
#
# # Sis sampler with uniform prior
#
# num_hidden_states <- 2 # 2 is the underlying truth
# sis_iters <- 1000
# bin_weight_prior_par <- NULL
# latent_prior_par <- NULL
#
#
#
# sis_output <- sis_estimator(obs=obs_data,num_bins=num_bins,iters=sis_iters, num_hidden_states=num_hidden_states,
#                             bin_weight_prior_par = bin_weight_prior_par,
#                             latent_prior_par = latent_prior_par, is_mixture = FALSE)

# Additional complexity from binning

# Example 1: Three component (iid given states), mean p/m 1

num_samples <- 1000
mix_weights <- c(0.6,0.4)
true_mix_comps <- length(mix_weights)


mean_array <- matrix(rep(c(1,-1),3), nrow = 3, ncol = true_mix_comps, byrow = TRUE)
var_array <- matrix(1, nrow = 3, ncol = true_mix_comps)
# mixture of N(1,1) and N(-1,1)

seed_no <- 123
set.seed(seed = seed_no)

real_obs_data <- simul_normal_mixture(mix_weights,
                                 mean_array, # mean array to be obs_dim rows, num_states columns
                                 var_array, # dim same as mean_array
                                 num_samples)

num_bins_ls <- c(4,8,16)
transformed_data <- matrix(truncated_inv_logit(real_obs_data),
                           nrow = nrow(real_obs_data), ncol = ncol(real_obs_data) )



# Sis sampler with uniform prior
for (num_bins in num_bins_ls){
  binned_data <- uniformly_bin(transformed_data, num_bins = num_bins)
  num_hidden_states_ls <- c(2,3,4) # 2 is the underlying truth
  sis_iters <- 500
  bin_weight_prior_par <- NULL
  latent_prior_par <- NULL
  df <- NULL
  hdf5_filepath <- paste0(home_dir, '../local_data/marginal_likelihood_data.h5')
  hdf5_key <- paste0('normal_mix_ex_1_',num_samples,
                     '_samples_',num_bins,'_bins_',
                     true_mix_comps,'_comps')

  for (num_hidden_states in num_hidden_states_ls){
    print(paste(Sys.time(),':','Commencing SIS procedure for',num_hidden_states,'hidden states'))
    sis_output <- sis_estimator(obs=binned_data,num_bins=num_bins,iters=sis_iters, num_hidden_states,
                                bin_weight_prior_par = bin_weight_prior_par,
                                latent_prior_par = latent_prior_par, is_mixture = TRUE)
    if (is.null(df)){
      df <- data.frame(sis_output$evidence)
      df_cols <- c(paste0('log_evid_',num_hidden_states,'_states'))
      colnames(df) <- df_cols
    } else {
      df <- cbind(df, data.frame(sis_output$evidence) )
      df_cols <- c(df_cols, paste0('log_evid_',num_hidden_states,'_states'))
      colnames(df) <- df_cols
    }
  }
  rhdf5::h5write(df, file = hdf5_filepath, name = hdf5_key, native = TRUE)
}


# Example 2: Three component (iid given states), mean 1, var differs

num_samples <- 1000
mix_weights <- c(0.6,0.4)
true_mix_comps <- length(mix_weights)


mean_array <- matrix(0, nrow = 3, ncol = true_mix_comps, byrow = TRUE)
var_array <- matrix(rep(c(1,4),3), nrow = 3, ncol = true_mix_comps)
# mixture of N(0,1) and N(0,4)

seed_no <- 123
set.seed(seed = seed_no)

real_obs_data <- simul_normal_mixture(mix_weights,
                                      mean_array, # mean array to be obs_dim rows, num_states columns
                                      var_array, # dim same as mean_array
                                      num_samples)
# hist(real_obs_data[1,],breaks = 30)

num_bins_ls <- c(4,8,16)
transformed_data <- matrix(truncated_inv_logit(real_obs_data),
                           nrow = nrow(real_obs_data), ncol = ncol(real_obs_data) )



# Sis sampler with uniform prior
for (num_bins in num_bins_ls){
  binned_data <- uniformly_bin(transformed_data, num_bins = num_bins)
  num_hidden_states_ls <- c(2,3,4) # 2 is the underlying truth
  sis_iters <- 500
  bin_weight_prior_par <- NULL
  latent_prior_par <- NULL
  df <- NULL
  hdf5_filepath <- paste0(home_dir, '../local_data/marginal_likelihood_data.h5')
  hdf5_key <- paste0('normal_mix_ex_2_',num_samples,
                     '_samples_',num_bins,'_bins_',
                     true_mix_comps,'_comps')

  for (num_hidden_states in num_hidden_states_ls){
    print(paste(Sys.time(),':','Commencing SIS procedure for',num_hidden_states,'hidden states'))
    sis_output <- sis_estimator(obs=binned_data,num_bins=num_bins,iters=sis_iters, num_hidden_states,
                                bin_weight_prior_par = bin_weight_prior_par,
                                latent_prior_par = latent_prior_par, is_mixture = TRUE)
    if (is.null(df)){
      df <- data.frame(sis_output$evidence)
      df_cols <- c(paste0('log_evid_',num_hidden_states,'_states'))
      colnames(df) <- df_cols
    } else {
      df <- cbind(df, data.frame(sis_output$evidence) )
      df_cols <- c(df_cols, paste0('log_evid_',num_hidden_states,'_states'))
      colnames(df) <- df_cols
    }
  }
  rhdf5::h5write(df, file = hdf5_filepath, name = hdf5_key, native = TRUE)
}




## Examples using naive marginal likelihood evaluator

num_samples <- 5
# need num_states rows, num_bins columns
num_bins <- 8
num_its <- 10^6
seed_no <- 123

hdf5_filepath <- paste0(home_dir, '../local_data/naive_monte_carlo_evidence.h5')
hdf5_key <- paste0('samples')
mc_evidence_outputs <- vector("list")
obs_data_ls <- vector("list")
df2 <- NULL

for (true_mix_comps in 2:4){
  set.seed(seed = seed_no)
  mix_weights <- gtools::rdirichlet(1, rep(1,true_mix_comps))
  emission_mat_dim1 <- gtools::rdirichlet(true_mix_comps,rep(1,num_bins))
  # for each dimension, the emission distribution for each state is uniform on simplex
  emission_mat_dim2 <- gtools::rdirichlet(true_mix_comps,rep(1,num_bins))
  emission_mat_dim3 <- gtools::rdirichlet(true_mix_comps,rep(1,num_bins))
  emission_mat_list <- list(emission_mat_dim1,emission_mat_dim2,emission_mat_dim3)

  obs_data <- simul_multinom_mixture(mix_weights,emission_mat_list, num_samples)
  obs_data_ls[[paste0(true_mix_comps,'_true_comps')]] <- obs_data

  for (fit_states in 2:4){
    mc_evidence_outputs[[paste0(true_mix_comps,'_true_',fit_states,'_fitted'
    )]] <- monte_carlo_evidence(data_matrix = obs_data, num_its = num_its, num_states = fit_states,
                                               num_bins = 8, bin_weight_prior_par = 1, latent_prior_par = 1)
  }
}

df2 <- data.frame(mc_evidence_outputs)
rhdf5::h5write(df2, file = hdf5_filepath, name = hdf5_key, native = TRUE)

# same data, now with the sis

num_hidden_states_ls <- c(2:4)
sis_iters <- 10000
bin_weight_prior_par <- NULL
latent_prior_par <- NULL
df2 <- NULL
true_mix_comps <- 1

for (obs_data in obs_data_ls ){
  true_mix_comps <- true_mix_comps + 1
  for (num_hidden_states in num_hidden_states_ls){
    print(paste(Sys.time(),':','Commencing SIS procedure for',true_mix_comps,'true states',
                num_hidden_states,'fitted hidden states'))
    sis_output <- sis_estimator(obs=obs_data,num_bins=num_bins,iters=sis_iters, num_hidden_states,
                                bin_weight_prior_par = bin_weight_prior_par,
                                latent_prior_par = latent_prior_par, is_mixture = TRUE)
    if (is.null(df2)){
      df2 <- data.frame(sis_output$evidence)
      df_cols <- paste0(true_mix_comps,'_true_',num_hidden_states,'_fit')
      colnames(df2) <- df_cols
    } else {
      df2 <- cbind(df2, data.frame(sis_output$evidence) )
      df_cols <- c(df_cols, paste0(true_mix_comps,'_true_',num_hidden_states,'_fit'))
      colnames(df2) <- df_cols
    }
  }
}

hdf5_filepath <- paste0(home_dir, '../local_data/sis_draws_naive_comparison.h5')
hdf5_key <- paste0('samples')

rhdf5::h5write(df2, file = hdf5_filepath, name = hdf5_key, native = TRUE)