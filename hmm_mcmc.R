# Source code for HMM MCMC functions.
# Includes MCMC samplers and output processing (thinning, label-swapping)
# Also includes functions for calculating smoothing probabilities
# Dependencies: RHmm, gtools, invgamma

prior_set <- function(
  num_states,
  num_bins,
  single_trans_row_prior = NULL,
  single_emission_prior = NULL,
  set_mix_prior = FALSE
  ) {
  # set prior
  # specify num_states many Dirichlet-num_states vectors for transition matrix
  # specify num_states many Dirichlet-num_bins vectors for emission vectors
  if (is.null(single_trans_row_prior)) {
    single_trans_row_prior <- rep(1, num_states)
  }
  if (is.null(single_emission_prior)) {
    single_emission_prior <- rep(1, num_bins)
  }
  emission_prior <- t(replicate(num_states, single_emission_prior))
  if (set_mix_prior){
    return(list("mix_weight_prior" = single_trans_row_prior,
                "emission_prior" = emission_prior))
  }
  else{
    trans_mat_prior <- t(replicate(num_states, single_trans_row_prior))
    return(list("trans_mat_prior" = trans_mat_prior,
                "emission_prior" = emission_prior))
  }
}

sample_trans_mat <- function(hidden_states, trans_mat_prior) {
  num_states <- nrow(trans_mat_prior) # recover the number of distinct states
  transition_count <- matrix(0, nrow = num_states, ncol = num_states)
  sample_size <- length(hidden_states)
  for (i in 1:(sample_size - 1)) {
    transition_count[
      hidden_states[i],
      hidden_states[i + 1]
    ] <- transition_count[
      hidden_states[i],
      hidden_states[i + 1]
    ] + 1
    # Counts according to transition
  }
  trans_mat_post <- trans_mat_prior + transition_count # New Dirichlet weights
  trans_mat_draw <- matrix(0, nrow = num_states, ncol = num_states)
  for (i in seq_len(num_states)) {
    trans_mat_draw[i, ] <- gtools::rdirichlet(1, trans_mat_post[i, ])
    # draws Q from newly updated Dirichlet weights
  }
  return(trans_mat_draw)
}

sample_emission_weights <- function(hidden_states, obs_data, emission_prior) {
    # States X (Count) Data Y Prior parameters B
  num_states <- nrow(emission_prior)
  num_bins <- ncol(emission_prior)
  emission_count <- matrix(0, nrow = num_states, ncol = num_bins)
    # counts number of X in i, Y in m (to update Dir)
  for (i in seq_len(num_states)) {
    for (j in seq_len(num_bins)) {

      emission_count[i, j] <- (hidden_states == i) %*% (obs_data == j)
      # adds one every time both X_t=i and Y_t=j
    }
  }
  emission_post <- emission_prior + emission_count
    # Updates dirichlet weights when observation for bin j, state i comes in
  emissions_draw <- matrix(0, nrow = num_states, ncol = num_bins)
  for (i in c(1:num_states)) {
    emissions_draw[i, ] <- gtools::rdirichlet(1, emission_post[i, ])
      # draws weights from newly updated Dirichlet params
  }
  return(emissions_draw)
}

sample_hidden_states <- function(obs_data, trans_mat, emission_mat) {
    # Samples X vector using forward-backward algo
    # Sample from the joint distribution of the latent variables, given params
    # library(RHmm) gives access to forward-backward algo
  num_states <- nrow(emission_mat)
  num_bins <- ncol(emission_mat)
    # Sets the number of states/bins based on number of columns of W

  leading_eigvec <- eigen(t(trans_mat))$vectors[, 1]
  stat_dist <- abs(leading_eigvec) / sum(abs(leading_eigvec))
    # stationary dist - leading eigenvector of Q transpose
    # leading eigvec could be all negative from sign choice so abs
  emissions_list <- split(t(emission_mat),
    rep(1:num_states, each = num_bins))
    # puts into list as required for RHmm package
  dist_hmm <- RHmm::distributionSet (dis = "DISCRETE",
    proba = emissions_list, labels = paste( c(1:num_bins) ) )
    # initialise distribution
  hmm <- RHmm::HMMSet(stat_dist, trans_mat, dist_hmm)
    # gets the HMM object to which we apply algos
  fb_out <- RHmm::forwardBackward(hmm, obs_data)
    # stores forwardbackward variables
  gamma_ <- fb_out$Gamma # Marginal probabilities of X_t=i given data, params
  xsi <- fb_out$Xsi
    # List of T, Xsi[[t]]_{rs}=Xsi_{rs}(t)=P(X_t=r,X_{t+1}=s | Y,Param)
  log_like <- fb_out$LLH
    # Gives log likelihood for input parameters, used in label swapping
  num_obs <- length(obs_data)
  hidden_states_draw <- rep(0, num_obs)
  # populating the hidden_states_draw vector:
  hidden_states_draw[1] <- sample(num_states, 1, prob = gamma_[1, ])
  for (i in c(2:num_obs)) {
    next_state_trans_prob <- xsi[[(i - 1)]][ # double [ indexes list
      hidden_states_draw[i - 1], ] / gamma_[i - 1, hidden_states_draw[i - 1]]
      # Proposal for drawing X_i | X_{i-1}
    hidden_states_draw[i] <- sample(
      seq_len(num_states), size = 1, prob = next_state_trans_prob)
  }
  return(list("hidden_states" = hidden_states_draw, "log_like" = log_like))
}

binned_prior_sampler <- function(
  obs_data, # obs_data should be pre-binned
  num_states,
  num_bins,
  num_its,
  trans_mat_dir_par = 1,
  emission_dir_par = 1,
  hidden_states_init = NULL,
  sample_hidden_states = TRUE,
  store_hidden_states = FALSE,
  update_every = 1000
  ) {
  prior_params <- prior_set(num_states, num_bins,
    rep(trans_mat_dir_par, num_states), rep(emission_dir_par, num_bins))
  trans_mat_prior <- prior_params$trans_mat_prior
  emission_prior <- prior_params$emission_prior
  num_obs <- length(obs_data)
  if (is.null(hidden_states_init)) {
    hidden_states <- sample( seq_len(num_states), size = num_obs, replace = TRUE)
      # Initial state vector, drawn randomly if not provided
  } else {
    hidden_states <- hidden_states_init
  }
  # initialisation of lists of draws
  trans_mat_draws <- vector("list", num_its)
  emission_draws <- vector("list", num_its)
  llh_draws <- vector("list", num_its)
  if (store_hidden_states){
    hidden_states_draws <- vector("list", num_its)
  }

  for (i in seq_len(num_its) ) { # Here we will store draws on Q and W
    if (i%%update_every == 0){
      print(paste(Sys.time(), "Iteration",i) )
    }
    trans_mat_draws[[i]] <- sample_trans_mat(hidden_states, trans_mat_prior)
    emission_draws[[i]] <- sample_emission_weights(
        hidden_states, obs_data, emission_prior)
    hidden_states_and_llh <- sample_hidden_states(
        obs_data, trans_mat_draws[[i]], emission_draws[[i]])
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
  return(list("trans_mat_draws" = trans_mat_draws,
    "emission_weight_draws" = emission_draws,
    "hidden_states_draws" = hidden_states_draws, # output depends on store_hidden_states flag
    "log_like_draws" = llh_draws))
}

distance <- function(x, y) {
  return(sqrt(sum((x - y)^2)))
}

label_swap <- function(
  trans_mat_draws,
  emission_weight_draws,
  log_like_draws,
  thin_every = 10, # rate of thinning
  trans_mat_prior = NULL, # prior used to compute MAP (optional, default unif)
  emission_prior = NULL,
  reference_trans_mat = NULL, # optionally provide reference for relabelling
  reference_emission_weights = NULL, # if not provided defer to MAP
  only_trans_mat_distance = FALSE # relabels only on dist between trans mats
  ) {
  num_states <- nrow(emission_weight_draws[[1]])
    # Recovers number of hidden states
  num_bins <- ncol(emission_weight_draws[[1]])
    # Recoves number of bins
  num_its <- length(trans_mat_draws)
    # Total number of iterations
  num_thinned_its <- floor( num_its / thin_every )
    # Number of iterations in thinned output

  if (is.null(reference_trans_mat) | is.null(reference_emission_weights) ){
      # Compute MAP index to get reference
    if (is.null(trans_mat_prior)) {
    trans_mat_prior <- prior_set(num_states, num_bins)$trans_mat_prior
    }
    if (is.null(emission_prior)) {
      emission_prior <- prior_set(num_states, num_bins)$emission_prior
    }
    post_density <- rep(0, num_its)
    for (i in seq_len(num_its)) { # Finds the  posterior mode aka MAP
      prior_density <- sum((trans_mat_prior - 1) * log(trans_mat_draws[[i]]))
      + sum((emission_prior - 1) * log(emission_weight_draws[[i]]))
        # prior density up to constant
      post_density[i] <- log_like_draws[[i]] + prior_density
    }
    map_index <- which(post_density == max(post_density))[1]
      # Finds argmax of posterior
  }

  if (is.null(reference_trans_mat)) {
    reference_trans_mat <- trans_mat_draws[[map_index]]
  }

  if (is.null(reference_emission_weights) & !only_trans_mat_distance){
      # only compute ref emission weights if we use in distance calc
    reference_emission_weights <-  emission_weight_draws[[map_index]]
  }
  # initialising empty lists for the trans mat and weight draws
  thinned_trans_mat_draws <- vector("list", num_thinned_its)
  thinned_emission_weight_draws <- vector("list", num_thinned_its)
  thinned_log_likes <- vector("list", num_thinned_its)

  thin_idx <- function(idx){
      # local function to go between thinned list and original list
    return(1 + thin_every*(idx-1))
  }

  hidden_label_perms <- gtools::permutations(num_states, num_states)
    # Matrix of permutations
  num_perms <- factorial(num_states)
  for (i in seq_len(num_thinned_its)) {
    distances <- rep(0, num_perms)
    trans_mat <- trans_mat_draws[[thin_idx(i)]] # Initialise
    emission_weights <- emission_weight_draws[[thin_idx(i)]]
    for (j in seq_len(num_perms)) { # perms from gtools
      for (k in seq_len(num_states)) {
        emission_weights[k, ] <- emission_weight_draws[[
            thin_idx(i)]][hidden_label_perms[j, k], ]
              # Permute each vector of emission weights according to perm j
        for (l in seq_len(num_states)) {
          trans_mat[k, l] <- trans_mat_draws[[thin_idx(i)]][
              hidden_label_perms[j, k], hidden_label_perms[j, l]]
                # Permute trans mat accoridng to perm j
        }
      } # End loop over matrix entries for particular perm
      if (only_trans_mat_distance){ # Computing distances
        distances[j] <- distance(trans_mat, reference_trans_mat)
          # finds distance between permuted trans mat and ref
      } else {
        distances[j] <- distance(c(trans_mat, emission_weights),
          c(reference_trans_mat, reference_emission_weights))
          # Finds distance between permuted full params and ref params
      }
      if (distances[j] == min(distances[1:j])){
          # smallest distance considered so far
        thinned_trans_mat_draws[[i]] <- trans_mat
        thinned_emission_weight_draws[[i]] <- emission_weights
        thinned_log_likes[[i]] <- log_like_draws[[thin_idx(i)]]
          # update thinned list with current best label swapped version
      }
    } # End loop over particular perm: Next apply best perm
  }
  return(list("thinned_trans_mat_draws" = thinned_trans_mat_draws,
    "thinned_emission_weight_draws" = thinned_emission_weight_draws,
  "thinned_log_likes" = thinned_log_likes))
}

uniformly_bin <- function(obs_data, num_bins, link = NULL) {
    # obs_data must be in [0,1). Provide link function otherwise
  if (is.null(link) == FALSE) {
    obs_data <- link(obs_data)
  } # if link has been specified, apply it
  binned_data <- floor(num_bins * obs_data) + 1
    # Gives bin label in ascending order
  binned_data[binned_data == (num_bins + 1)] <- num_bins
    # Resolves boundary issues from Y==1
  return(binned_data)
}

truncated_inv_logit <- Vectorize( function(x, lower = -2 , upper = 2) {
    # Define function which we will vectorize
  assertthat::assert_that(lower<=upper)
    # verify that lower is less than upper
  if (lower == upper) {
    y <- gtools::inv.logit(x)
  } else if (x < lower) {
    y <- gtools::inv.logit(x)
  } else if (x > upper) {
    y <- gtools::inv.logit(x)
  } else {
    y <- gtools::inv.logit(lower) + (
      (x - lower) / (upper - lower)) * (
        gtools::inv.logit(upper) - gtools::inv.logit(lower))
  }
  return(y)
}, "x" ) # vectorizes the function in terms of the "x" argument

# Code for sampling from Dir process mixture models
# Further details in Chapter 5 Ghosal & vdV (2017)
# See Equation (5.7) for latent variable formulation
# We use Dirichlet-multinomial approximation as base measure
# See Chapter 4, Sec 4.3.3 in G & vdV (2017)

# Samples the latent location variables for the mixing measure
latent_loc_sample <- function(
  latent_mixture_state_value, # value for the pointer index
  hidden_state_value, # value for the hidden chain
  latent_mixture_states, # vector of pointers
  hidden_states,
  obs_data,
  centering_mean,
  centering_var,
  scale # vector of scales, entries for each hidden state value
  ) {
    # Uses conjugacy of Gaussian base measure
  sum_of_obs <- sum(obs_data[
    (latent_mixture_states==latent_mixture_state_value)&(
      hidden_states==hidden_state_value)
    ]) # The sufficient statistic - sum(relevant obs)
  eff_sample_size <- sum(
    (latent_mixture_states==latent_mixture_state_value)&(
      hidden_states==hidden_state_value)
    ) # The 'effective sample size' for updating this theta
  conditional_var <- (
    (1/centering_var)+eff_sample_size/scale[hidden_state_value] )^(-1)
      # Using https://en.wikipedia.org/wiki/Conjugate_prior
  conditional_mean <- conditional_var*(
    centering_mean/centering_var + sum_of_obs/scale[hidden_state_value] )
  latent_loc <- rnorm(1,conditional_mean,sqrt(conditional_var))
  return(latent_loc)
}

# Samples scale parameter, common parameter for each HMM hidden state
scale_sample <- function(
  hidden_state_value,
  latent_mixture_states,
  hidden_states,
  obs_data,
  latent_locs,
  inv_gamma_shape,
  inv_gamma_rate
  ){
  eff_sample_size <- sum(hidden_states==hidden_state_value)
  conditional_shape <- inv_gamma_shape + (eff_sample_size/2) # using conjugacy
  relevant_obs <- obs_data[hidden_states==hidden_state_value]
  relevant_means <- latent_locs[
    latent_mixture_states[hidden_states==hidden_state_value],
    hidden_state_value]
      # pick out pointers correpsonding to relevant_obs
      # since hidden_state value fixed, second index of
      # latent_mixture_states matrix is fixed
  sum_of_squares <- sum((relevant_obs-relevant_means)^2 )
  conditional_rate <- inv_gamma_rate + 0.5*sum_of_squares
  scale_for_given_hidden_state <- invgamma::rinvgamma(
    1,shape=conditional_shape, rate=conditional_rate)
    # gives variance corresponding to given hidden state value
  return(scale_for_given_hidden_state)
}

# Samples Dirichlet allocation vector
dir_weights_sample <- function(
  max_mix_comps,
  latent_mixture_states,
  hidden_states,
  prior_precision,
  num_states){
  dir_weights <- matrix(0,nrow=max_mix_comps,ncol=num_states)
  for (hidden_state_value in seq_len(num_states)){
    prior_dir_param <- rep(prior_precision/max_mix_comps,max_mix_comps)
    value_counts <- table(
      factor(latent_mixture_states[hidden_states==hidden_state_value],
       levels=1:max_mix_comps)
      )
      # counts number of obs in each bin, similar to pd.DataFrame.value_counts()
      # factor creates R equivalent of pandas categorical
    conditional_dir_param <- prior_dir_param + value_counts
      # increases dirichlet weight by one per unit assignment
    dir_weights[,hidden_state_value] <- gtools::rdirichlet(1,conditional_dir_param)
  }
  return(dir_weights)
}

# Samples HMM hidden states and latent mixture states simulatenously
bivariate_states_sample <- function(
  obs_data,
  trans_mat,
  dir_weights,
  latent_locs,
  scale
  ){
  max_mix_comps <- nrow(dir_weights) #(Recovers SMax)
  num_states <- ncol(dir_weights) #Recovers R
  bivar_trans_mat <- get_bivar_trans_mat(trans_mat,dir_weights)
    #sets transition matrix for bivariate state space
  leading_eigvec <- eigen(t(bivar_trans_mat))$vectors[, 1]
  stat_dist <- abs(leading_eigvec) / sum(abs(leading_eigvec))
  #first state X=1 S=1 second state X=2 S=1 third state X=1 S=2 etc.
  dist_hmm <- RHmm::distributionSet(
    dis="NORMAL",mean = as.vector(t(latent_locs)),var = rep(scale,max_mix_comps)
    )
  hmm <- RHmm::HMMSet(stat_dist,bivar_trans_mat,dist_hmm)
  fb_out <- RHmm::forwardBackward(hmm,obs_data)
  gamma_ <- fb_out$Gamma
  xsi <- fb_out$Xsi
  log_like <- fb_out$LLH
  num_obs <- length(obs_data)
  bivar_hidden_states_draw <- rep(0, num_obs)
  num_bivar_states <- max_mix_comps * num_states
  bivar_hidden_states_draw[1] <- sample(num_bivar_states, 1, prob = gamma_[1, ])
  for (i in c(2:num_obs)) {
    next_state_trans_prob <- xsi[[(i - 1)]][
      bivar_hidden_states_draw[i - 1], ] / gamma_[
        i - 1, bivar_hidden_states_draw[i - 1]]
      # Proposal for drawing X_i | X_{i-1}
    bivar_hidden_states_draw[i] <- sample(
      seq_len(num_bivar_states), size = 1, prob = next_state_trans_prob)
  }
  hidden_state_draws <- Vectorize(mod)(bivar_hidden_states_draw,num_states)
    #gives corresponding hidden chain states
  latent_mixture_state_draws <- floor((bivar_hidden_states_draw-1)/num_states)+1
    #gives corresponding mixture allocation
  return( list("hidden_state_draws"=hidden_state_draws,
    "latent_mixture_state_draws"=latent_mixture_state_draws,
    "log_like"=log_like) )
}

# Gets transition matrix for bivariate latent space
get_bivar_trans_mat <- function(hmm_trans_mat,mixture_dir_weights){
  num_states <- ncol(mixture_dir_weights)
  max_mix_states <- nrow(mixture_dir_weights)
  num_bivar_states <- num_states*max_mix_states
  bivar_trans_mat <- matrix(0,nrow=num_bivar_states,ncol=num_bivar_states)
    #allocates enlarged transition matrix
  for (i in seq_len(num_bivar_states) ){ #find a better way of doing this
    for (j in seq_len(num_bivar_states) ){
      bivar_trans_mat[i,j] <- hmm_trans_mat[
        mod(i,num_states),mod(j,num_states)
        ]*mixture_dir_weights[
          floor((j-1)/num_states)+1,mod(j,num_states)
          ]
    }
  }
  return(bivar_trans_mat)
}

# Returns i mod R but R=R rather than R=0
mod <- function(i,R){
  return( (i+(R-1))%%R+1 )
}

# Main sampler for cut posterior given transition matrices
dir_proc_mix_cut_sampler <- function(
    # Fits cut posterior sampler
  obs_data, # Data should be real-valued
  prior_precision, # Prior precision "M" parameter for DPMM
  centering_mean, # Mean for normal base measure
  centering_var, # Variance for normal base measure
  inv_gamma_shape, # Shape parameter for inv gamma prior on scale
  inv_gamma_rate, # Rate parameter for inv gamma prior on scale
  trans_mat_draws, # (Thinned, swapped) list of trans mats
  hidden_states_init =NULL,
    # Initial hidden states (e.g. MLE from binned sampler)
  max_mix_comps=NULL,
    # Number of mixture components allowed in the DPMM
    # Theoretically should be uncapped but capped for computation
    # See Ghosal/van der Vaart (2017) Chapter 5
    # If null, defaults to sqrt of sample size
  num_inner_iters=10, # Number of iterations per inner (mini) chain
    # Overall iterations will be dictated by number of trans mat draws
  update_every = 100 # number of iterations before log updates
  ){
  start_time <- Sys.time()
  sample_size <- length(obs_data)
  num_states <- nrow(trans_mat_draws[[1]]) #Number of states
  num_outer_iters <- length(trans_mat_draws)
  # Setting max mixture components if null
  if (is.null(max_mix_comps)){
    max_mix_comps <- max( 20,floor(sqrt(sample_size)) )
  }
  # Initialising output lists
  latent_loc_draws <- vector("list",num_outer_iters)
  dir_weight_draws <- vector("list",num_outer_iters)
  log_like_draws <- vector("list",num_outer_iters)
  scale_draws <- vector("list",num_outer_iters)
  # Initialise inverse variance from prior
  scale <- invgamma::rinvgamma(num_states,
    shape=inv_gamma_shape, rate=inv_gamma_rate)
  # Initialize dir weights from prior
  dir_weights <- t(gtools::rdirichlet(num_states,
    rep((prior_precision/max_mix_comps),max_mix_comps)))
  if (is.null(hidden_states_init)){
      #random initialisation of X if none specified
    hidden_states_init <- sample(
      c(1:num_states), size = num_obs, replace = TRUE)
  }
  # Initialize pointers from prior
  latent_mixture_states <- rep(0,sample_size)
  for (state in seq_len(num_states)){
    fil_state <- (hidden_states_init==state)
    latent_mixture_states[fil_state] <- sample(
      seq_len(max_mix_comps),sum(fil_state),
      replace = TRUE,prob=dir_weights[,state])
  }
  assertthat::assert_that(all(latent_mixture_states>0))
    # checks all states updated
  # Initial allocation of theta array
  latent_locs <- matrix(0,nrow=max_mix_comps,ncol=num_states)
  # Sampler
  for ( outer_iter in seq_len(num_outer_iters) ){
    if (outer_iter%%update_every == 0){
      print(paste(Sys.time(), "Outer iteration",outer_iter))
    }

    hidden_states <- hidden_states_init # returned from previous loop

    for (inner_iter in seq_len(num_inner_iters) ){
      for (state in seq_len(num_states) ){ #Theta, Precisions vary state-by-state
        # Updating latent locations (theta variables in book)
        occupied_mix_states <- unique(latent_mixture_states[hidden_states==state])
        for (mix_state in occupied_mix_states ){
            # go through pairs for which we have non-trivial likelihood
            # the rest are redrawn from the prior afterwards
          latent_locs[mix_state, state] <- latent_loc_sample(
            mix_state,state,latent_mixture_states,hidden_states,obs_data,
            centering_mean,centering_var,scale)
        }
        latent_locs[-occupied_mix_states,state] <- rnorm(
          max_mix_comps-length(occupied_mix_states),centering_mean,sqrt(centering_var))
          # prior draws for trivial updates
        
        # Updating scale based on Normal/InvGamma conjugacy
        scale[state] <- scale_sample(state,latent_mixture_states,hidden_states,
          obs_data,latent_locs,inv_gamma_shape,inv_gamma_rate)
        
      }
      
      # Update Dirichlet weights
      dir_weights <-dir_weights_sample(max_mix_comps,latent_mixture_states,
        hidden_states,prior_precision,num_states)
      
      # Update hidden states of chain and latent mixture states using forward/backward
      bi_states <- bivariate_states_sample(obs_data,trans_mat_draws[[outer_iter]],
        dir_weights,latent_locs,scale)
      hidden_states <- bi_states$hidden_state_draws
      latent_mixture_states <- bi_states$latent_mixture_state_draws
      
      if (inner_iter==num_inner_iters){ #stores final draw of chain
        latent_loc_draws[[outer_iter]] <- latent_locs
        dir_weight_draws[[outer_iter]] <- dir_weights
        log_like_draws[[outer_iter]] <- bi_states$log_like
        scale_draws[[outer_iter]] <- scale
        hidden_states_init <- hidden_states
          # initial hidden state allocation for next loop
          # could consider using same initialiser for all
          # or using MLE along the inner-chain
      }
      
    }
    
  }
  
  end_time=Sys.time()

  return( list("latent_loc_draws"=latent_loc_draws,
    "dir_weight_draws"=dir_weight_draws,
    "log_like_draws"=log_like_draws,
    "scale_draws"=scale_draws,
    "elapsed"=end_time-start_time) )
}

# Sampler for fully Bayesian approach, does not require inner chains to mix
dir_proc_mix_full_sampler <- function(
  obs_data, # Data should be real-valued
  prior_precision, # Prior precision "M" parameter for DPMM
  centering_mean, # Mean for normal base measure
  centering_var, # Variance for normal base measure
  inv_gamma_shape, # Shape parameter for inv gamma prior on scale
  inv_gamma_rate, # Rate parameter for inv gamma prior on scale
  num_states, # Number of states for HMM
  num_outer_iters, # Number of outer iterations (each has a minichain)
  hidden_states_init=NULL,
    # Initial hidden states (e.g. MLE from binned sampler)
  max_mix_comps=NULL,
    # Number of mixture components allowed in the DPMM
    # Theoretically should be uncapped but capped for computation
    # See Ghosal/van der Vaart (2017) Chapter 5
    # If null, defaults to sqrt of sample size
    # Overall iterations will be dictated by number of trans mat draws
  trans_mat_prior=NULL,
  update_every=100 # number of iterations before log updates
  ){
  start_time <- Sys.time()
  sample_size <- length(obs_data)
  # Setting max mixture components if null
  if (is.null(max_mix_comps)){
    max_mix_comps <- max( 20,floor(sqrt(sample_size)) )
  }
  # Initialising output lists
  trans_mat_draws <- vector("list",num_outer_iters)
  latent_loc_draws <- vector("list",num_outer_iters)
  dir_weight_draws <- vector("list",num_outer_iters)
  log_like_draws <- vector("list",num_outer_iters)
  scale_draws <- vector("list",num_outer_iters)
  # Initialise inverse variance from prior
  scale <- invgamma::rinvgamma(num_states,
    shape=inv_gamma_shape, rate=inv_gamma_rate)
  # Initialize dir weights from prior
  dir_weights <- t(gtools::rdirichlet(num_states,
    rep((prior_precision/max_mix_comps),max_mix_comps)))
  if (is.null(hidden_states_init)){
      #random initialisation of X if none specified
    hidden_states <- sample(
      c(1:num_states), size = num_obs, replace = TRUE)
  } else {
    hidden_states <- hidden_states_init
  }
  if (is.null(trans_mat_prior)){
    trans_mat_prior <- matrix(1, nrow = num_states, ncol = num_states)
  }
  # Initialize pointers from prior
  latent_mixture_states <- rep(0,sample_size)
  for (state in seq_len(num_states)){
    fil_state = (hidden_states_init==state)
    latent_mixture_states[fil_state] <- sample(
      c(1:max_mix_comps),sum(fil_state),
      replace = TRUE,prob=dir_weights[,state])
  }
  assertthat::assert_that(all(latent_mixture_states>0))
    # checks all states updated
  # Initial allocation of theta array
  latent_locs <- matrix(0,nrow=max_mix_comps,ncol=num_states)
  # Sampler
  for ( outer_iter in seq_len(num_outer_iters) ){
    if (outer_iter%%update_every == 0){
      print(paste(Sys.time(), "Outer iteration",outer_iter))
    }
    trans_mat <- sample_trans_mat(hidden_states,trans_mat_prior)
    for (state in seq_len(num_states) ){ #locs, scales vary state-by-state
      # Updating latent locations (theta variables in book)
      occupied_mix_states <- unique(latent_mixture_states[hidden_states==state])
      for (mix_state in occupied_mix_states ){
          # go through pairs for which we have non-trivial likelihood
          # the rest are redrawn from the prior afterwards
        latent_locs[mix_state, state] <- latent_loc_sample(
          mix_state,state,latent_mixture_states,hidden_states,obs_data,
          centering_mean,centering_var,scale)
      }
      latent_locs[-occupied_mix_states,state] <- rnorm(
        max_mix_comps-length(occupied_mix_states),centering_mean,sqrt(centering_var))
        # prior draws for trivial updates
      
      # Updating scale based on Normal/InvGamma conjugacy
      scale[state] <- scale_sample(state,latent_mixture_states,hidden_states,
        obs_data,latent_locs,inv_gamma_shape,inv_gamma_rate)
      
    }
    
    # Update Dirichlet weights
    dir_weights <-dir_weights_sample(max_mix_comps,latent_mixture_states,
      hidden_states,prior_precision,num_states)
    
    # Update hidden states of chain and latent mixture states using forward/backward
    bi_states <- bivariate_states_sample(obs_data,trans_mat,dir_weights,latent_locs,scale)
    hidden_states <- bi_states$hidden_state_draws
    latent_mixture_states <- bi_states$latent_mixture_state_draws

    # Storing draws
    trans_mat_draws[[outer_iter]] <- trans_mat
    latent_loc_draws[[outer_iter]] <- latent_locs
    dir_weight_draws[[outer_iter]] <- dir_weights
    log_like_draws[[outer_iter]] <- bi_states$log_like
    scale_draws[[outer_iter]] <- scale

      
  }
  
  end_time <- Sys.time()

  return( list( "trans_mat_draws"=trans_mat_draws,
    "latent_loc_draws"=latent_loc_draws,
    "dir_weight_draws"=dir_weight_draws,
    "log_like_draws"=log_like_draws,
    "scale_draws"=scale_draws,
    "elapsed"=end_time-start_time) )
}

# Code for getting smoothing probabilities

# Smoothing probabilities when using cut approach, for which our result applies
smoothing_cut_posterior <- function(
  obs_data, # real-valued data
  sampler_output, # output of cut posterior sampler
  trans_mat_draws # provided separately as not part of cut output
  ){ 
  num_iters <- length(trans_mat_draws)
  smoothing_probs <- vector("list", length = num_iters)
  for (iter in seq_len(num_iters)){
    trans_mat <- trans_mat_draws[[iter]]
    leading_eigvec <- eigen(t(trans_mat))$vectors[, 1]
    stat_dist <- abs(leading_eigvec) / sum(abs(leading_eigvec))
    dist_hmm <- RHmm::distributionSet(
      dis="MIXTURE",
      mean = mix_params_for_smoothing(iter, sampler_output)$mean ,
      var = mix_params_for_smoothing(iter, sampler_output)$var ,
      proportion = mix_params_for_smoothing(iter, sampler_output)$mix_props)
    hmm <- RHmm::HMMSet(stat_dist,trans_mat,dist_hmm)
    smoothing_probs[[iter]] <- RHmm::forwardBackward(hmm,obs_data)$Gamma
  }
  return(smoothing_probs) 
}

# Smoothing probabilities using only binned prior, for comparison
smoothing_bin_posterior <- function(
  obs_data, # binned data
  sampler_output # output of bin posterior sampler
  ){ 
  num_iters <- length(sampler_output$trans_mat_draws) 
  num_states <- nrow(sampler_output$emission_weight_draws[[1]])
  num_bins <- nrow(sampler_output$emission_weight_draws[[1]])
  smoothing_probs <- vector("list", length = num_iters)
  for (iter in seq_len(num_iters)){
    trans_mat <- sampler_output$trans_mat_draws[[iter]]
    leading_eigvec <- eigen(t(trans_mat))$vectors[, 1]
    stat_dist <- abs(leading_eigvec) / sum(abs(leading_eigvec))
    emission_mat <- sampler_output$emission_weight_draws[[iter]]
    emissions_list <- split(t(emission_mat),
      rep(1:num_states, each = num_bins) )
      # splits matrix into a list of the rows, entry for each state
    dist_hmm <- RHmm::distributionSet(dis = "DISCRETE",
      proba = emissions_list, labels = paste(c(1:num_bins)))
    hmm <- RHmm::HMMSet(stat_dist,trans_mat,dist_hmm)
    smoothing_probs[[iter]] <- RHmm::forwardBackward(hmm,obs_data)$Gamma
  }
  return(smoothing_probs)
}

# Smoothing probabilities using fully Bayesian approach
smoothing_bayes_posterior <- function(
  obs_data, # real-valued data
  sampler_output # output of bayes posterior sampler
  ){ 
  num_iters <- length(sampler_output$trans_mat_draws)
  smoothing_probs <- vector("list", length = num_iters)
  for (iter in seq_len(num_iters)){
    trans_mat <- sampler_output$trans_mat_draws[[iter]]
    leading_eigvec <- eigen(t(trans_mat))$vectors[, 1]
    stat_dist <- abs(leading_eigvec) / sum(abs(leading_eigvec))
    dist_hmm <- RHmm::distributionSet(
      dis="MIXTURE",
      mean = mix_params_for_smoothing(iter, sampler_output)$mean ,
      var = mix_params_for_smoothing(iter, sampler_output)$var ,
      proportion = mix_params_for_smoothing(iter, sampler_output)$mix_props)
    hmm <- RHmm::HMMSet(stat_dist,trans_mat,dist_hmm)
    smoothing_probs[[iter]] <- RHmm::forwardBackward(hmm,obs_data)$Gamma
  }
  return(smoothing_probs)
}

# Sets list required to set mixtures as emissions using RHmm
mix_params_for_smoothing <- function(iter,sampler_output){
  mix_locs <- sampler_output$latent_loc_draws[[iter]]
  mix_scales <- sampler_output$scale_draws[[iter]]
  mix_props <- sampler_output$dir_weight_draws[[iter]]
  num_states <- ncol(mix_locs)
  num_mix_comps <- nrow(mix_locs)
  mean_list <- vector("list", length = num_states)
  var_list <- vector("list", length = num_states)
  mix_props_list <- vector("list", length = num_states)
  for (state in seq_len(num_states)){
    mean_list[[state]] <- mix_locs[,state]
    var_list[[state]] <- rep_len(mix_scales[state], length.out = num_mix_comps)
      # each is a vector of length num_mix_comps
    mix_props_list[[state]] <- mix_props[,state]
  }
  return( list( "mean"=mean_list,
    "var"=var_list,
    "mix_props"=mix_props_list) )
}

smoothing_probs <- function(obs_data, # should be real-valued
  trans_mat,
  emission_means, # assumes normal distribution for emissions
  emission_vars
  ){
  leading_eigvec <- eigen(t(trans_mat))$vectors[, 1]
  stat_dist <- abs(leading_eigvec) / sum(abs(leading_eigvec))
  dist_hmm <- distributionSet(dis="NORMAL",emission_means,emission_vars)
  hmm <- HMMSet(stat_dist,trans_mat,dist_hmm)
  smoothing_probs <- forwardBackward(hmm,obs_data)$Gamma
  return(smoothing_probs)
}