mixture_sampler <- function(
  obs_data, # obs_data should be pre-binned
  obs_dim = 3,
  num_states, # number of mixture states
  num_bins, # number of bins for data
  num_its, # mcmc iterations
  mix_weight_dir_par = 1,
  emission_dir_par = 1,
  hidden_states_init = NULL,
  sample_hidden_states = TRUE,
  store_hidden_states = FALSE,
  update_every = 1000
) {
  assertthat::assert_that(obs_dim = nrow(obs_data))
  prior_params <- prior_set(num_states, num_bins,
                             rep(mix_weight_dir_par, num_states), rep(emission_dir_par, num_bins), set_mix_prior = TRUE)
  mix_weight_prior <- prior_params$mix_weight_prior
  emission_prior <- prior_params$emission_prior
  num_obs <- ncol(obs_data)
  if (is.null(hidden_states_init)) {
    hidden_states <- sample( seq_len(num_states), size = num_obs, replace = TRUE)
    # Initial state vector, drawn randomly if not provided
  } else {
    hidden_states <- hidden_states_init
  }
  # initialisation of lists of draws
  mix_weight_draws <- vector("list", num_its)
  emission_draws <- vector("list", num_its)
  llh_draws <- vector("list", num_its)
  if (store_hidden_states){
    hidden_states_draws <- vector("list", num_its)
  }

  for (i in seq_len(num_its) ) {
    if (i%%update_every == 0){
      print(paste(Sys.time(), "Iteration",i) )
    }
    mix_weight_draws[[i]] <- sample_mix_weights(hidden_states, num_states, mix_weight_prior)
    emission_draws[[i]] <- sample_emission_weights_multidim(
      hidden_states, obs_data, emission_prior, obs_dim)
    hidden_states_and_llh <- sample_latent_mixture_states(
      obs_data, mix_weight_draws[[i]], emission_draws[[i]])
    # also computes log likelihood for Q[i] and W[i]
    llh_draws[[i]] <- hidden_states_and_llh$log_like
    hidden_states <- hidden_states_and_llh$hidden_states
    if (store_hidden_states){
      hidden_states_draws[[i]] <- hidden_states
    }else if (llh_draws[[i]] == max(unlist(llh_draws)[1:i])){
      # store latents when encountering new maximum for likelihood
      hidden_states_draws <- list(
        "Iteration Number" = i, "states" = hidden_states)
    }
  }
  return(list("mix_weight_draws" = mix_weight_draws,
              "emission_weight_draws" = emission_draws,
              "hidden_states_draws" = hidden_states_draws, # output depends on store_hidden_states flag
              "log_like_draws" = llh_draws))
}

sample_mix_weights <- function(hidden_states, num_states, mix_weight_prior){
  hidden_state_counts <- as.double(table(factor(hidden_states, levels = seq_len(num_states))))
  posterior_dir_weights <- mix_weight_prior + hidden_state_counts
  mix_weights_draw <- gtools::rdirichlet(1, posterior_dir_weights)
  return( as.double(mix_weights_draw) )
}

sample_emission_weights_multidim <- function(
  hidden_states, obs_data, emission_prior, obs_dim) {
  num_states <- nrow(emission_prior)
  num_bins <- ncol(emission_prior)
  # counts number of X in i, Y in m (to update Dir)
  emission_weights_draw <- vector("list", length = obs_dim)
  for (dim in seq_len(obs_dim)){
    cross_freq_by_dim <- table(factor(hidden_states,levels=1:num_states),factor(obs_data[dim,],levels=1:num_bins))
    emission_weights_draw[[dim]] <- matrix(0,nrow=num_states, ncol=num_bins)
    for (state in seq_len(num_states) ){
      emission_weight_dir_par <- emission_prior[state,] + cross_freq_by_dim[state,]
      emission_weights_draw[[dim]][state,] <- gtools::rdirichlet(1,emission_weight_dir_par)
    }
  }
  return(emission_weights_draw)
}

sample_latent_mixture_states <- function(obs_data, mix_weights, emission_weights){
    # emission weights a list, one entry per dim, each entry matrix of dimension (num_states x num_bins)
  num_obs <- ncol(obs_data)
  obs_dim <- nrow(obs_data)
  num_states <- length(mix_weights)
  # Calculation of latent state draw
  latents <- rep(0, num_obs)
  for (i in seq_len(num_obs)){
    state_probs <- rep(0, num_states)
    for (state in seq_len(num_states)){
      state_probs[state] <- mix_weights[state]
      for (dim in obs_dim){
        state_probs[state] <- state_probs[state] * emission_weights[[dim]][state, obs_data[dim, i] ]
      }
    }
    state_probs <- state_probs / sum(state_probs)
    latents[i] <- sample(1:num_states,1,prob = state_probs)
  }
  # Calculation of log-likelihood of parameters, p(obs|mix_weights,emission_weights)
  log_like <- mixture_likelihood(obs_data, mix_weights, emission_weights)
  return(list("hidden_states" = latents, "log_like"=log_like))
}

mixture_likelihood <- function(obs_data, mix_weights, emission_weights){

  llh_per_obs <- function(single_obs, mix_weights, emission_weights){
    num_states <- length(mix_weights)
    obs_dim <- length(single_obs)
    likelihood_per_state <- rep(0, num_states)
    for (state in seq_len(num_states)){
      for (dim in seq_len(obs_dim)){
        likelihood_per_state[state] <- likelihood_per_state[state] * (
          emission_weights[[dim]][state,single_obs[dim]] )
      }
    }
    log_likelihood <- log(sum(likelihood_per_state*mix_weights))
  }

  global_llh_function <- Vectorize(llh_per_obs, vectorize.args = "single_obs")

  log_likelihood <- sum(global_llh_function(obs_data,mix_weights,emission_weights) )

  return(log_likelihood)
}