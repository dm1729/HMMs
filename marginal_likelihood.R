# Routines for computation of marginal likelihood
# Used for model selection via the Bayes factor
# Following routines are for mixtures of histograms

# mix_weight_prior_par - dirichlet hyperparameters for the mixing weights (length K = num_mix_comps)
# bin_weight_prior_par - dir hyperparameters for the bin weights (length Kappa_M = num_bins)

log_partition_prior <- function(latent_states, num_mix_comps, mix_weight_prior_par){ # Eq (14) in Hairault et al.
    # Evaluates the log partition prior density at the partition induced by a given vector of latent states
  sample_size <- length(latent_states)
  filled_mix_comps <- length(unique(latent_states))
  state_counts <- as.double(table(factor(latent_states, levels = seq_len(num_mix_comps))))
  log_constant_factor <- lgamma(num_mix_comps + 1) - lgamma(num_mix_comps - filled_mix_comps + 1)
  log_dir_prior_factor <- lgamma(sum(mix_weight_prior_par)) - lgamma(sample_size+sum(mix_weight_prior_par))
  log_state_count_factor <- sum( lgamma(state_counts + mix_weight_prior_par) - lgamma(mix_weight_prior_par) )
  return(log_constant_factor+log_dir_prior_factor+log_state_count_factor)
}

log_partition_likelihood <- function(obs, num_bins, latent_states, num_mix_comps, bin_weight_prior_par){
  # Eq (15) in Hairault et al.
  assertthat::assert_that(num_bins==length(bin_weight_prior_par))
  state_counts <- as.double(table(factor(latent_states, levels = seq_len(num_mix_comps))))
  cross_freq <- table(factor(obs,levels=1:num_bins),factor(latent_states,levels=1:num_mix_comps))
  log_like <- 0
  for (state in seq_len(num_mix_comps)){
    log_like <- log_like + ( sum( lgamma(cross_freq[,state]) - lgamma(bin_weight_prior_par) )
    + lgamma(sum(bin_weight_prior_par)) - lgamma(state_counts[state]+sum(bin_weight_prior_par)) )
      # Using Equation (15) with multinomial conjugate prior
  }
  return(log_like)
}

chib_partition_estimator <- function(obs,num_bins,latent_states_ls, num_mix_comps,
                                     bin_weight_prior_par = NULL, mix_weight_prior_par=NULL){
  if (is.null(bin_weight_prior_par)){
    bin_weight_prior_par <- rep(1, num_bins)
  }
  if (is.null(mix_weight_prior_par)){
    mix_weight_prior_par <- rep(1, num_mix_comps)
  }
  iters <- length(latent_states_ls)
  sum_log_prior_likelihood <- rep(0, iters)
  for (iter in seq_len(iters) ){
    log_prior <- log_partition_prior(latent_states_ls[[iter]], num_mix_comps, mix_weight_prior_par)
    log_likelihood <- log_partition_likelihood(obs, num_bins, latent_states_ls[[iter]],
                                               num_mix_comps, bin_weight_prior_par)
    sum_log_prior_likelihood[iter] <- log_prior + log_likelihood
  }
  map_iter <- which(sum_log_prior_likelihood == max(sum_log_prior_likelihood))
    # finds the iteration which is maximum a posteriori over the partitions
  map_posterior_estimate <- 0 # Initialises estimate for posterior density at MAP
  map_latent_states <- latent_states_ls[[map_iter]]
  for (iter in seq_len(iters)){
    map_posterior_estimate <- map_posterior_estimate + (1/iters)*partitions_agree(
      map_latent_states, latent_states_ls[[iter]] )
  }
  assertthat::assert_that(map_posterior_estimate >= (1/iters))
  log_marginal_like <- sum_log_prior_likelihood[map_iter] - log(map_posterior_estimate)
  return(log_marginal_like)
}

partitions_agree <- function(latent_states1,latent_states2){
  # checks if two sequences of latent states induce the same partition
  states_agree <- TRUE
  while (length(latent_states1) > 0){
    cluster1 <- which(latent_states1 == latent_states1[1])
    cluster2 <- which(latent_states2 == latent_states2[1])
    if (all(cluster1==cluster2)){
      latent_states1 <- latent_states1[-cluster1]
      latent_states2 <- latent_states2[-cluster2]
    }else{
      states_agree <- FALSE
      break
    }
  }
  if (states_agree){
    return(1)
  }
  else{
    return(0)
  }
}