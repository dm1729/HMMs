# Includes procedures to generate synthetic data from HMMs
# Dependencies: RHmm

simul_normal_hmm <- function(trans_mat, normal_mean, normal_var, num_samples) {
  # Generates N samples from param Q, mu, sigma^2
  for (row_idx in seq_len(nrow(trans_mat))){ # ensures unit row sums
    assertthat::assert_that(sum(trans_mat[row_idx,])==1)
  }
  leading_eigvec <- eigen(t(trans_mat))$vectors[, 1]
  stat_dist <- abs(leading_eigvec) / sum(abs(leading_eigvec)) # stationary dist
  dist_hmm <- RHmm::distributionSet(
    dis = "NORMAL", mean = normal_mean, var = normal_var)
  # paste puts class labels in as strings
  hmm <- RHmm::HMMSet(stat_dist, trans_mat, dist_hmm) # Sets models
  return(RHmm::HMMSim(num_samples, hmm))
}

simul_multinom_hmm <- function(trans_mat, emission_mat, num_samples) {
  # Generates N samples from param Q,W
  num_states <- nrow(emission_mat)
  num_bins <- ncol(emission_mat)
  leading_eigvec <- eigen(t(trans_mat))$vectors[, 1]
  stat_dist <- abs(leading_eigvec) / sum(abs(leading_eigvec))
  emissions_list <- split(t(emission_mat),
                          rep(1:num_states, each = num_bins)) # puts into list
  dist_hmm <- RHmm::distributionSet(dis = "DISCRETE",
                                    proba = emissions_list, labels = paste(c(1:num_bins)))
  # paste puts class labels in as strings
  hmm <- RHmm::HMMSet(stat_dist, trans_mat, dist_hmm) # Sets models
  return(RHmm::HMMSim(num_samples, hmm))
}

simul_normal_mixture <- function(mix_weights,
                                 mean_array, # mean array to be obs_dim rows, num_states columns
                                 var_array, # dim same as mean_array
                                 num_samples,
                                 obs_dim = 3){
  mix_weights <- mix_weights / sum(mix_weights)
  assertthat::assert_that(nrow(mean_array)==obs_dim)
  latents <- rmultinom(num_samples,1,mix_weights)
  obs_data <- matrix(0,nrow = obs_dim, ncol = num_samples)
  for (i in 1:num_samples){
    mean <- mean_array%*%latents[,i] # vector of length obs_dim
    var <- diag(as.double(var_array%*%latents[,i]))
    obs_data[,i] <- MASS::mvrnorm(1,mean,var)
  }
  return(obs_data)
}

simul_multinom_mixture <- function(mix_weights,
                                   emission_mat_list, # each entry to be num_states rows, num_bins columns
                                   num_samples,
                                   latents_out = FALSE,
                                   obs_dim = 3){
  mix_weights <- mix_weights / sum(mix_weights)
  assertthat::assert_that(length(emission_mat_list) == obs_dim)
  assertthat::assert_that(length(mix_weights)==nrow(emission_mat_list[[1]])) # so we have num_states rows
  num_bins <- ncol(emission_mat_list[[1]])
  latents <- rmultinom(num_samples,1,mix_weights) # gives nrow = num_states, ncol = num_samples
  obs_data <- matrix(0,nrow = obs_dim, ncol = num_samples)
  weights <- matrix(0, nrow = obs_dim, ncol = num_bins)
  for (i in 1:num_samples){
    for (dim in 1:obs_dim){
      weights[dim,] <- t(emission_mat_list[[dim]])%*%latents[,i] # vector of length num_bins
      # finds the weights associated to given dimension, and to given latent state
      obs_data[dim,i] <- sample(1:num_bins, size = 1, prob = as.double(weights[dim,]))
    }
  }
  if (latents_out){
    return(list('obs'=obs_data, 'latents'=latents))
  } else{
    return(obs_data)
  }
}