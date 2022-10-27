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

sis_estimator <- function(obs,num_bins,iters, num_hidden_states,
                          bin_weight_prior_par = NULL, # gives a row of dir weights
                          latent_prior_par = NULL, # gives a row of dir weights, either for mix weights or trans rows
                          is_mixture = TRUE,
                          obs_dim = NULL){
  log_evidence_weights <- rep(0,iters)
  if (is.null(bin_weight_prior_par)){
    bin_weight_prior_par <- rep(1,num_bins)
  }
  if (is.null(latent_prior_par)){
    latent_prior_par <- rep(1, num_hidden_states)
  }

  assertthat::assert_that(num_hidden_states==length(latent_prior_par))
  assertthat::assert_that(num_bins==length(bin_weight_prior_par))
  if (is_mixture){
    num_obs <- ncol(obs)
    obs_dim <- nrow(obs)
    assertthat::assert_that(obs_dim>=3)
  } else { # if fitting HMM
    obs <- as.matrix(obs)
    dim(obs) <- c(min(nrow(obs),ncol(obs)),max(nrow(obs),ncol(obs)))
      # coerce into matrix for compatibility with mixture observations
    num_obs <- ncol(obs)
  }
  latents_ls <- matrix(0, nrow = iters, ncol = num_obs)
  for (iter in seq_len(iters)){
    if (iter%%10 == 0){
      print(paste("Iteration number",iter))
    }
    latents <- rep(0, num_obs)
    latents[1] <- sample(1:num_hidden_states,1,prob = latent_prior_par/sum(latent_prior_par))
    log_evidence_weights[iter] <- baseline_log_evidence(as.matrix(obs[,1]),num_bins,
                                                bin_weight_prior_par, is_mixture = is_mixture)
    for (data_idx in 2:num_obs ){
      gamma <- rep(0, length(num_hidden_states))
      # print(paste("data index =",data_idx))
      for (state in seq_len(num_hidden_states)){
        # print(paste("calculating gamma_k for k=",state))
        # print(paste("current latents are ", latents[latents>0]))
        gamma[state] <- gamma_coef(idx = data_idx, state = state, obs = obs, latents = latents[1:(data_idx-1)],
                                   num_bins = num_bins, num_states = num_hidden_states,
                                   bin_weight_prior_par = bin_weight_prior_par,
                                   latent_prior_par = latent_prior_par, is_mixture = is_mixture)
      }
      # print(paste("sum of gamma is ",sum(gamma)))
      if ( sum(gamma)==0 ){
        print(paste("Error: sum of gamma is equal to zero"))
      }
      print(paste("gamma weights are:"))
      print(gamma/sum(gamma))
      print("sampling latent...")
      latents[data_idx] <- sample(1:num_hidden_states,1,prob = gamma/sum(gamma))
      print(paste("new latent sample is",latents[data_idx]))
      log_evidence_weights[iter] <- log_evidence_weights[iter] + log( sum(gamma) )
    }
    latents_ls[iter,] <- latents
  }
  return(list("evidence"=log_evidence_weights,"latents"=latents_ls))
}

gamma_coef <- function(idx, state, obs, latents, # obs is a matrix with ncol = sample size
                       num_bins, num_states,
                       bin_weight_prior_par,
                       latent_prior_par,
                       is_mixture = TRUE){ # gamma_k in Hairault et al.
  assertthat::assert_that(idx <= (length(latents) + 1) )
  assertthat::assert_that(idx >= 2)
  eff_obs <- as.matrix( as.matrix(obs[,1:(idx-1)])[,latents[1:(idx-1)]==state] )
  #print("call of gamma function")
  #print(paste("effective observations for evidence are",eff_obs))
  #print(paste("number of rows in eff_obs matrix is ",nrow(eff_obs)))
  #print(paste("number of rows in augmented eff_obs matrix is ",nrow(cbind(eff_obs, obs[,idx] ))))
  log_evid_new <- baseline_log_evidence(cbind(eff_obs, obs[,idx] ), num_bins, bin_weight_prior_par, is_mixture)
  # gives the m(C_k union {y_i} )
  log_evid_old <- baseline_log_evidence(eff_obs, num_bins, bin_weight_prior_par, is_mixture)
  print(log_evid_new)
  print(log_evid_old)
  posterior_latent_weight <- post_latent_weight(state,latents[1:(idx-1)], num_states,
                                                latent_prior_par, start_state = latents[(idx-1)], is_mixture)
  print(paste("posterior latent weight for state",state,"is",posterior_latent_weight))
  # gives the integral of eta_k ( or Q_{z_{i-1},k} ) with respect to the posterior given (y,z)_{1:i-1}
  evid_ratio <- exp(log_evid_new - log_evid_old)
  if (evid_ratio ==0 ){
    print(evid_new)
    print(evid_old)
    print(paste("Evidence ratio for the gamma_k is equal to zero, with k =",state,"data idx =",idx))
  }
  return( evid_ratio*posterior_latent_weight )
}

transition_count <- function(latent_states){
  sample_size <- length(latent_states)
  start_states <- latent_states[1:(sample_size-1)]
  end_states <- latent_states[2:sample_size]
  trans_count_mat <- table(start_states,end_states)
  return(as.matrix(trans_count_mat))
}

# bins_by_latent_count <- function(obs, latents, num_hidden_states){
#   # counts the number of obs in each bin for each value of latent
#   # used in posterior update (given obs, latents) of the emission weights
#
# }

baseline_log_evidence <- function(obs, num_bins, bin_weight_prior_par, is_mixture = TRUE){
  if (length(obs) == 0){
    return(0)
  }
  # print(paste("length of obs is ",obs))
  # The m(C_k(z)) quantity from Hairault et al. equation (15)
  # Uses multinomial conjugacy
  # the effective sample size is the number of i such that z_i = k
  # computation of the eff samp size to be done outside this function
  if (is_mixture){
    obs_dim <- nrow(obs)
    log_evidence <- 0
    #print(obs_dim)
    for (dim in seq_len(obs_dim)){
      bin_counts <- as.double(table(factor(obs[dim,], levels = 1:num_bins)))
      bin_weight_posterior_par <- bin_weight_prior_par + bin_counts
      log_evidence <- log_evidence + sum(lgamma(bin_weight_posterior_par)) - lgamma(sum(bin_weight_posterior_par))
                      + lgamma(sum(bin_weight_prior_par)) - sum(lgamma(bin_weight_prior_par))
        # because we have independence given states, the integral over the dimensions is a product
        # and the log_evidence's add
    }
  } else {
    bin_counts <- as.double(table(factor(obs, levels = 1:num_bins)))
    bin_weight_posterior_par <- bin_weight_prior_par + bin_counts
    # Given dirichlet parameter can compute normalising constant:
    # Normalising constant is a beta which is a ratio of gammas
    log_evidence <- log_evidence + sum(lgamma(bin_weight_posterior_par)) - lgamma(sum(bin_weight_posterior_par))
    + lgamma(sum(bin_weight_prior_par)) - sum(lgamma(bin_weight_prior_par))
  }

  return(log_evidence)
}

post_latent_weight <- function(state, latents, num_states, latent_prior_par, start_state = NULL, is_mixture = TRUE){
  if (is_mixture){
    state_counts <- as.double(table(factor(latents, levels = 1:num_states)))
    latent_post_par <- latent_prior_par + state_counts
    # print(latents)
    # print(latent_prior_par)
    # print(state_counts)
    # print(latent_post_par)
    return(latent_post_par[state]/sum(latent_post_par))
  } else {
    # need non-null start state
    transition_counts_from_start_state <- transition_count(latent_states = latents)[start_state,]
      # gives the historical counts from the relevant start state to other states
    latent_post_par <- latent_prior_par + transition_counts_from_start_state
    return(latent_post_par[state]/sum(latent_post_par))
  }
}