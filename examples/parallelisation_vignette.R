require(foreach)
require(doParallel)
library(foreach)


sum_up_to <- function(num,identifier){
  counter <- 0
  print(paste(Sys.time(),"Running identifier",identifier))
  for (i in seq_len(num)){
    counter <- counter + i
  }
  return(counter)
}


system.time(
  for (i in 1:6 ){
    out <- sum_up_to(10^6,i)
  }
)

system.time(
  foreach (i= 1:6 )%do%{
    sum_up_to(10^6,i)
  }
)

num_cores <- parallel::detectCores()
doParallel::registerDoParallel(cores=num_cores)


system.time(
  out <- unlist(foreach (i= 1:6 )%dopar%{
    sum_up_to(10^6,i)
  })
)