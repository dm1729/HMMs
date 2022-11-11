# Examples
home_dir <- '/home/moss/Documents/HMMs/'
source(paste0(home_dir,'data_simulator.R'))
source(paste0(home_dir,'hmm_mcmc.R'))
source(paste0(home_dir,'marginal_likelihood.R'))

# Selection of number of states - multinomial emissions:

## Multidimensional mixture data

num_samples <- 1000
mix_weights <- c(0.6,0.4)
true_mix_comps <- length(mix_weights)

# need num_states rows, num_bins columns

num_bins <- 4

seed_no <- 123
set.seed(seed = seed_no)
emission_mat_dim1 <- gtools::rdirichlet(2,rep(1,num_bins))
  # for each dimension, the emission distribution for each state is uniform on simplex
emission_mat_dim2 <- gtools::rdirichlet(2,rep(1,num_bins))
emission_mat_dim3 <- gtools::rdirichlet(2,rep(1,num_bins))
emission_mat_list <- list(emission_mat_dim1,emission_mat_dim2,emission_mat_dim3)

obs_data <- simul_multinom_mixture(mix_weights,emission_mat_list, num_samples)

# Sis sampler with uniform prior

num_hidden_states_ls <- c(2,3,4) # 2 is the underlying truth
sis_iters <- 100
bin_weight_prior_par <- NULL
latent_prior_par <- NULL
df <- NULL
hdf5_filepath <- paste0(home_dir,'marginal_likelihood_data.h5')
hdf5_key <- paste0('multinomial_mix_',num_samples,
                   '_samples_',num_bins,'_bins_',
                   true_mix_comps,'_comps')

for (num_hidden_states in num_hidden_states_ls){
  print(paste(Sys.time(),':','Commencing SIS procedure for',num_hidden_states,'hidden states'))
  sis_output <- sis_estimator(obs=obs_data,num_bins=num_bins,iters=sis_iters, num_hidden_states,
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

num_samples <- 1000
mix_weights <- c(0.6,0.4)
true_mix_comps <- length(mix_weights)


mean_array <- matrix(rep(c(1,-1),3), nrow = 3, ncol = true_mix_comps, byrow = TRUE)
var_array <- matrix(1, nrow = 3, ncol = true_mix_comps)

seed_no <- 123
set.seed(seed = seed_no)

real_obs_data <- simul_normal_mixture(mix_weights,
                                 mean_array, # mean array to be obs_dim rows, num_states columns
                                 var_array, # dim same as mean_array
                                 num_samples)

num_bins_ls <- c(2,16)
transformed_data <- matrix(truncated_inv_logit(real_obs_data),
                           nrow = nrow(real_obs_data), ncol = ncol(real_obs_data) )



# Sis sampler with uniform prior
for (num_bins in num_bins_ls){
  binned_data <- uniformly_bin(transformed_data, num_bins = num_bins)
  num_hidden_states_ls <- c(2,3,4) # 2 is the underlying truth
  sis_iters <- 100
  bin_weight_prior_par <- NULL
  latent_prior_par <- NULL
  df <- NULL
  hdf5_filepath <- paste0(home_dir,'marginal_likelihood_data.h5')
  hdf5_key <- paste0('normal_mix_',num_samples,
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

