# Includes procedures to generate synthetic data from HMMs
# Dependencies: RHmm

simul_normal_hmm <- function(trans_mat, normal_mean, normal_var, num_samples) {
    # Generates N samples from param Q, mu, sigma^2
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