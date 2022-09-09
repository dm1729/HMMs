# Includes procedures to post process output from MCMC samplers
# Processed output then saved for loading into Python environment

source("D:/Users/Dan/Documents/PhD/R Code/Semiparametric-HMMs/mcmc_samplers.R")
  # replace when packaged

posterior_mean <- function(trans_mat_draws) {
    # Input MCMC sample list (or thinned list) to get mean of entries
    # Should label swap output first
  return(Reduce("+", trans_mat_draws) / length(trans_mat_draws))
}

matrix_entries <- function(matrix_list, row_idx, col_idx) {
  # Converts list of matrices to vector of (row_idx,col_idx) entries
  num_mats <- length(matrix_list)
  vec_of_entries <- vector("double", length = num_mats)
  for (i in seq_len(num_mats)) {
    vec_of_entries[i] <- matrix_list[[i]][row_idx, col_idx]
  }
  return(vec_of_entries)
}

# Outputs dataframe
process_binned_output <- function(
  binned_output,
  thin_every,
  hdf5_filepath = NULL,
  hdf5_key = NULL) {
  thinned_output <- label_swap(
    binned_output$trans_mat_draws,
    binned_output$emission_weight_draws,
    binned_output$log_like_draws,
    thin_every
    )
  num_states <- nrow(thinned_output$thinned_emission_weight_draws)
  num_bins <- ncol(thinned_output$thinned_emission_weight_draws)
  emission_weight_entries <- vector("list")
  trans_mat_entries <- vector("list")
  for (state in seq_len(num_states)){
    for (bin in seq_len(num_bins)){
      list_entry_name <- paste0("weight_", "state_", state, "bin_", bin)
      emission_weight_entries[[list_entry_name]] <- matrix_entries(
          thinned_output$thinned_emission_weight_draws, state, bin)
    }
  }

  for (row_state in seq_len(num_states)){
    for (col_state in seq_len(num_states)){
      list_entry_name <- paste0("trans_mat_", "row_", row_state,
        "col_", col_state)
      trans_mat_entries[[list_entry_name]] <- matrix_entries(
          thinned_output$thinned_trans_mat_draws, state, bin)
    }
  }

log_likes <- binned_output$log_like_draws
param_df <- data.frame(log_likes, emission_weight_entries, trans_mat_entries)
if (is.null(hdf5_filepath)|is.null(hdf5_key)){
  return(param_df)
} else {
  # save to hdf5
}
}

process_cut_output <- function(cut_output){
  # as above for cut output
}

process_full_bayes_output <- function(full_bayes_output){
  # as above for fully bayesian approach
}