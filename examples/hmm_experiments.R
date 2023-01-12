### See README
home_dir <- ''
hdf5_filepath <- paste0(home_dir, './local_data/sis_output_other.h5')
datasets_with_keys <- vector("list")

### Generating datasets
# ## Example 1A - two states
# hdf5_key <- paste0('Example_1A')
#
# true_states <- 2
# trans_mat <- matrix( c(0.7,0.3,0.2,0.8),
#                      nrow=true_states, ncol=true_states, byrow = TRUE )
# normal_mean <- c(-1,1)
# normal_var <- c(1,1)
# num_samples <- 1000
# set.seed(123)
# real_obs_data <- simul_normal_hmm(trans_mat = trans_mat, normal_mean = normal_mean,
#                                   normal_var = normal_var, num_samples = num_samples)$obs
# transformed_data <- truncated_inv_logit(real_obs_data)
#
# datasets_with_keys <- append(datasets_with_keys,
#                              list( list("hdf5_key"= hdf5_key, "data"= transformed_data ) ))
#   # each entry of list is itself a list with hdf5_key and dataset
#
# ## Example 1B - three states
# hdf5_key <- paste0('Example_1B')
#
# true_states <- 3
# trans_mat <- matrix( c(0.6,0.3,0.1,0.3,0.3,0.4,0.7,0.1,0.2),
#                      nrow=true_states, ncol=true_states, byrow = TRUE )
# normal_mean <- c(-2,0,2)
# normal_var <- c(1,1,1)
# num_samples <- 1000
# set.seed(123)
# real_obs_data <- simul_normal_hmm(trans_mat = trans_mat, normal_mean = normal_mean,
#                                   normal_var = normal_var, num_samples = num_samples)$obs
# transformed_data <- truncated_inv_logit(real_obs_data)
#
# datasets_with_keys <- append(datasets_with_keys,
#                              list( list("hdf5_key"= hdf5_key, "data"= transformed_data ) ))

## Example 1C - three states
hdf5_key <- paste0('Example_1C')

true_states <- 3
trans_mat <- matrix( c(0.6,0.3,0.1,0.3,0.3,0.4,0.7,0.1,0.2),
                     nrow=true_states, ncol=true_states, byrow = TRUE )
normal_mean <- c(-1,0,1)
normal_var <- c(0.01,0.01,0.01)
num_samples <- 500
set.seed(123)
real_obs_data <- simul_normal_hmm(trans_mat = trans_mat, normal_mean = normal_mean,
                                  normal_var = normal_var, num_samples = num_samples)$obs
transformed_data <- truncated_inv_logit(real_obs_data)

datasets_with_keys <- append(datasets_with_keys,
                             list( list("hdf5_key"= hdf5_key, "data"= transformed_data ) ))

# ## Example 2A - two states
# hdf5_key <- paste0('Example_2A')
#
# true_states <- 2
# trans_mat <- matrix( c(0.7,0.3,0.2,0.8),
#                      nrow=true_states, ncol=true_states, byrow = TRUE )
# normal_mean <- c(-0.5,0.5)
# normal_var <- c(1,1)
# num_samples <- 1000
# set.seed(123)
# real_obs_data <- simul_normal_hmm(trans_mat = trans_mat, normal_mean = normal_mean,
#                                   normal_var = normal_var, num_samples = num_samples)$obs
# transformed_data <- truncated_inv_logit(real_obs_data)
#
# datasets_with_keys <- append(datasets_with_keys,
#                              list( list("hdf5_key"= hdf5_key, "data"= transformed_data ) ))
# # each entry of list is itself a list with hdf5_key and dataset
#
# ## Example 2B - three states
# hdf5_key <- paste0('Example_2B')
#
# true_states <- 3
# trans_mat <- matrix( c(0.6,0.3,0.1,0.3,0.3,0.4,0.7,0.1,0.2),
#                      nrow=true_states, ncol=true_states, byrow = TRUE )
# normal_mean <- c(-1,0,1)
# normal_var <- c(1,1,1)
# num_samples <- 1000
# set.seed(123)
# real_obs_data <- simul_normal_hmm(trans_mat = trans_mat, normal_mean = normal_mean,
#                                   normal_var = normal_var, num_samples = num_samples)$obs
# transformed_data <- truncated_inv_logit(real_obs_data)
#
# datasets_with_keys <- append(datasets_with_keys,
#                              list( list("hdf5_key"= hdf5_key, "data"= transformed_data ) ))
#
# ## Example 2C - two states
# hdf5_key <- paste0('Example_2C')
#
# true_states <- 2
# trans_mat <- matrix( c(0.7,0.3,0.2,0.8),
#                      nrow=true_states, ncol=true_states, byrow = TRUE )
# normal_mean <- c(0,0)
# normal_var <- c(1,4)
# num_samples <- 1000
# set.seed(123)
# real_obs_data <- simul_normal_hmm(trans_mat = trans_mat, normal_mean = normal_mean,
#                                   normal_var = normal_var, num_samples = num_samples)$obs
# transformed_data <- truncated_inv_logit(real_obs_data)
#
# datasets_with_keys <- append(datasets_with_keys,
#                              list( list("hdf5_key"= hdf5_key, "data"= transformed_data ) ))
# # each entry of list is itself a list with hdf5_key and dataset
#
# ## Example 2D - three states
# hdf5_key <- paste0('Example_2D')
#
# true_states <- 3
# trans_mat <- matrix( c(0.6,0.3,0.1,0.3,0.3,0.4,0.7,0.1,0.2),
#                      nrow=true_states, ncol=true_states, byrow = TRUE )
# normal_mean <- c(0,0,0)
# normal_var <- c(1,4,9)
# num_samples <- 1000
# set.seed(123)
# real_obs_data <- simul_normal_hmm(trans_mat = trans_mat, normal_mean = normal_mean,
#                                   normal_var = normal_var, num_samples = num_samples)$obs
# transformed_data <- truncated_inv_logit(real_obs_data)
#
# datasets_with_keys <- append(datasets_with_keys,
#                              list( list("hdf5_key"= hdf5_key, "data"= transformed_data ) ))
#
# ## Example 3A - two states
# hdf5_key <- paste0('Example_3A')
#
# true_states <- 2
# trans_mat <- matrix( c(0.7,0.3,0.2,0.8),
#                      nrow=true_states, ncol=true_states, byrow = TRUE )
# normal_mean <- c(0,1)
# normal_var <- c(4,9)
# num_samples <- 1000
# set.seed(123)
# real_obs_data <- simul_normal_hmm(trans_mat = trans_mat, normal_mean = normal_mean,
#                                   normal_var = normal_var, num_samples = num_samples)$obs
# transformed_data <- truncated_inv_logit(real_obs_data)
#
# datasets_with_keys <- append(datasets_with_keys,
#                              list( list("hdf5_key"= hdf5_key, "data"= transformed_data ) ))
# # each entry of list is itself a list with hdf5_key and dataset
#
# ## Example 3B - three states
# hdf5_key <- paste0('Example_3B')
#
# true_states <- 3
# trans_mat <- matrix( c(0.6,0.3,0.1,0.3,0.3,0.4,0.7,0.1,0.2),
#                      nrow=true_states, ncol=true_states, byrow = TRUE )
# normal_mean <- c(0,1,2)
# normal_var <- c(4,1,9)
# num_samples <- 1000
# set.seed(123)
# real_obs_data <- simul_normal_hmm(trans_mat = trans_mat, normal_mean = normal_mean,
#                                   normal_var = normal_var, num_samples = num_samples)$obs
# transformed_data <- truncated_inv_logit(real_obs_data)
#
# datasets_with_keys <- append(datasets_with_keys,
#                              list( list("hdf5_key"= hdf5_key, "data"= transformed_data ) ))


### Computations

# Computational Parameters
sis_iters <- 1000
num_cores <- parallel::detectCores()  # leaving one core left over
doParallel::registerDoParallel(cores=num_cores)
sis_iters_per_core <- ceiling(sis_iters/num_cores)

# Statistical Parameter Configurations (each will be a dataframe column)
num_bins_ls <- c(4,8,16)
num_hidden_states_ls <- c(2,3,4)
bin_weight_prior_par <- NULL
latent_prior_par <- NULL

for (example in datasets_with_keys){
  # Initialise dataframe for example, to eventually be stored in HDF for given example$key
  df <- NULL
  for (num_bins in num_bins_ls){
    binned_data <- uniformly_bin(example$data, num_bins = num_bins)

    for (num_hidden_states in num_hidden_states_ls){
      print(paste(Sys.time(),':','Commencing SIS procedure for',example$hdf5_key,
                  num_bins,'bins', num_hidden_states,'fitted hidden states'))

      sis_output <- unlist(foreach (i= 1:num_cores )%dopar%{ # parallel computation of sis algo
        sis_estimator(obs=binned_data,num_bins=num_bins,iters=sis_iters_per_core, num_hidden_states,
                      bin_weight_prior_par = NULL,
                      latent_prior_par = NULL, is_mixture = FALSE)$evidence
      })

      if (is.null(df)){ # updating dataframe and column names
        df <- data.frame(sis_output)
        df_cols <- c(paste0('log_evid_',num_hidden_states,'_fitted_states_',num_bins,'_bins'))
        colnames(df) <- df_cols
      } else {
        df <- cbind(df, data.frame(sis_output) )
        df_cols <- c(df_cols, paste0('log_evid_',num_hidden_states,'_fitted_states_',num_bins,'_bins'))
        colnames(df) <- df_cols
      }
    }
  }
  rhdf5::h5write(df, file = hdf5_filepath, name = example$hdf5_key, native = TRUE)
}

doParallel::stopImplicitCluster()