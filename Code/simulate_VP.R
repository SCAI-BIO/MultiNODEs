simulate_VPs = function(fitted,n=NA){
  library(bnlearn)
  
  n_VP=n
  
  VP = c()
  iter = 1
  
  # loops until we have a full dataset of VPs (overshoots so data is not always < n_ppts)
  while(NROW(VP) < n_VP){
    cat("iteration = ", iter, "\n")
    
    # generate data (until no NAs in any variables)
    generatedDF = rbn(fitted, n = n_VP)
    comp<-F
    while (!comp){ # using mixed data sometimes results in NAs in the generated VPs. These VPs are rejected.
      generatedDF<-generatedDF[complete.cases(generatedDF),]
      gen<-n_VP-dim(generatedDF)[1]
      if (gen>0){
        generatedDF<-rbind(generatedDF,rbn(fitted, n = gen)) # draw virtual patients
      }else{
        comp<-T 
      }
    }
    
    # VPs are iteratively rejected if they have less than 50% chance to be classified as "real" in a network focussing on correctly classifying real ppts.
    acceptedVPs = generatedDF
    VP = rbind.data.frame(VP, acceptedVPs)
    iter = iter + 1
    print(NROW(VP))
  }
  VP
}
