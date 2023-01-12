# Includes procedures to post process output from MCMC samplers
# Processed output then saved for loading into Python environment

source("hmm_mcmc.R")
  # replace when packaged

matrix_entries <- function(matrix_list, row_idx, col_idx = 1) {
  # Converts list of matrices to vector of (row_idx,col_idx) entries
  num_mats <- length(matrix_list)
  vec_of_entries <- vector("double", length = num_mats)
  for (i in seq_len(num_mats)) {
    vec_of_entries[i] <- as.matrix(matrix_list[[i]])[row_idx, col_idx]
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
  num_states <- nrow(thinned_output$thinned_emission_weight_draws[[1]])
  num_bins <- ncol(thinned_output$thinned_emission_weight_draws[[1]])
  emission_weight_entries <- vector("list")
  trans_mat_entries <- vector("list")
  for (state in seq_len(num_states)){
    for (bin in seq_len(num_bins)){
      list_entry_name <- paste0("weight_", "state_", state, "_bin_", bin)
      emission_weight_entries[[list_entry_name]] <- matrix_entries(
          thinned_output$thinned_emission_weight_draws, state, bin)
    }
  }

  for (row_state in seq_len(num_states)){
    for (col_state in seq_len(num_states)){
      list_entry_name <- paste0("trans_mat_", "row_", row_state,
        "_col_", col_state)
      trans_mat_entries[[list_entry_name]] <- matrix_entries(
          thinned_output$thinned_trans_mat_draws, row_state, col_state)
    }
  }

  log_likes <- unlist(thinned_output$thinned_log_likes)
  param_df <- data.frame(log_likes, emission_weight_entries, trans_mat_entries)
  if (is.null(hdf5_filepath)|is.null(hdf5_key)){
    return(param_df)
  } else {
    rhdf5::h5write(param_df, file = hdf5_filepath, name = hdf5_key, native = TRUE)
    # writes to h5 file readable in pandas
  }
  }

process_cut_output <- function(cut_output, hdf5_filepath = NULL, hdf5_key = NULL){
  num_mix_comps <- nrow(cut_output$latent_loc_draws[[1]])
  num_states <- ncol(cut_output$latent_loc_draws[[1]])
  latent_loc_entries <- vector("list")
  mix_weight_entries <- vector("list")
  scale_entries <- vector("list")
  for (state in seq_len(num_states)){
    scale_entry_name <- paste0("scale_state_",state)
    scale_entries[[scale_entry_name]] <- matrix_entries(cut_output$scale_draws, row_idx=state)
    for (mix_comp in seq_len(num_mix_comps)){
      latent_loc_entry_name <- paste0("mix_loc_comp_",mix_comp,"_state_",state)
      mix_weight_entry_name <- paste0("mix_weight_comp_",mix_comp,"_state_",state)
      latent_loc_entries[[latent_loc_entry_name]] <- matrix_entries(
        cut_output$latent_loc_draws, row_idx=mix_comp, col_idx=state)
      mix_weight_entries[[mix_weight_entry_name]] <- matrix_entries(
        cut_output$dir_weight_draws, row_idx=mix_comp, col_idx=state)
    }
  }
  log_likes <- unlist(cut_output$log_like_draws)
  param_df <- data.frame(log_likes, scale_entries, latent_loc_entries, mix_weight_entries)
  if (is.null(hdf5_filepath)|is.null(hdf5_key)){
    return(param_df)
  } else {
    rhdf5::h5write(param_df, file = hdf5_filepath, name = hdf5_key, native = TRUE)
    # writes to h5 file readable in pandas
  }
}

process_full_bayes_output <- function(
  full_bayes_output,
  thin_every = 1,
  hdf5_filepath = NULL,
  hdf5_key = NULL){
  # as above for fully bayesian approach
  # need to use label swapper with distance based only on Q
    # or implement distance when not based on Q
  # needs thinning
  thinned_output <- label_swap( trans_mat_draws= full_bayes_output$trans_mat_draws,
                                log_like_draws = full_bayes_output$log_like_draws,
                                scale_draws = full_bayes_output$scale_draws,
                                latent_loc_draws = full_bayes_output$latent_loc_draws,
                                dir_weight_draws = full_bayes_output$dir_weight_draws,
                                thin_every = thin_every,
                                full_bayes_output = TRUE )
  num_states <- nrow(thinned_output$thinned_trans_mat_draws[[1]])
  trans_mat_entries <- vector("list")
  for (row_state in seq_len(num_states)){
    for (col_state in seq_len(num_states)){
      list_entry_name <- paste0("trans_mat_", "row_", row_state,
                                "_col_", col_state)
      trans_mat_entries[[list_entry_name]] <- matrix_entries(
        thinned_output$thinned_trans_mat_draws, row_state, col_state)
    }
  }
  num_mix_comps <- nrow(thinned_output$thinned_latent_loc_draws[[1]])
    # number of components in DPMM (so O(sqrt(n)) using algo implemented)
  num_states <- ncol(thinned_output$thinned_latent_loc_draws[[1]])
  latent_loc_entries <- vector("list")
  mix_weight_entries <- vector("list")
  scale_entries <- vector("list")
  for (state in seq_len(num_states)){
    scale_entry_name <- paste0("scale_state_",state)
    scale_entries[[scale_entry_name]] <- matrix_entries(thinned_output$thinned_scale_draws, row_idx=state)
    for (mix_comp in seq_len(num_mix_comps)){
      latent_loc_entry_name <- paste0("mix_loc_comp_",mix_comp,"_state_",state)
      mix_weight_entry_name <- paste0("mix_weight_comp_",mix_comp,"_state_",state)
      latent_loc_entries[[latent_loc_entry_name]] <- matrix_entries(
        thinned_output$thinned_latent_loc_draws, row_idx=mix_comp, col_idx=state)
      mix_weight_entries[[mix_weight_entry_name]] <- matrix_entries(
        thinned_output$thinned_dir_weight_draws, row_idx=mix_comp, col_idx=state)
    }
  }
  log_likes <- unlist(thinned_output$thinned_log_likes)
  param_df <- data.frame(log_likes, trans_mat_entries, scale_entries,
                         latent_loc_entries,mix_weight_entries)
  if (is.null(hdf5_filepath)|is.null(hdf5_key)){
    return(param_df)
  } else {
    rhdf5::h5write(param_df, file = hdf5_filepath, name = hdf5_key, native = TRUE)
    # writes to h5 file readable in pandas
  }
}