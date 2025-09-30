library(BGLR)
library(stringr)
library(data.table)
library(dplyr)
library(iml)


RKHS <- function(train, valid, test, params){
  
  params <- unlist(params)
  nIter <- params[1]
  burnIn <-  params[2]
  Shapley_num <- params[3]
  get_effect <- params[4]

  data <- rbind(train, valid, test)
  data_qtl <- data.frame(lapply(data[,1:(ncol(data)-1)], as.numeric))
  data_pheno <- data[,ncol(data):ncol(data)]
  
  D <- as.matrix(dist(data_qtl,method="euclidean"))^2
  D <- D/mean(D)
  h <- 1
  K <- exp(-h*D)
  
  y <- as.numeric(unlist(data_pheno))
  
  y_test <- y
  y_test[(nrow(data)-(nrow(valid)+nrow(test))+1):nrow(data)] <- NA
  
  fm <- BGLR(y=y_test,ETA=list(list(K=K,model='RKHS')),
             nIter=nIter,burnIn=burnIn,verbose=FALSE,saveAt='./Result/eig_')
  
  y_predicted <- fm$yHat[(nrow(data)-nrow(test)+1):nrow(data)]
  y_actual <- y[(nrow(data)-nrow(test)+1):nrow(data)]

  pearson <- cor(y_predicted, y_actual, method = c("pearson"))
  MSE <- mean((y_predicted - y_actual)^2)
  
  y_predicted_train <- fm$yHat[1:nrow(train)]
  y_actual_train <- y[1:nrow(train)] 
  
  if(nrow(valid)!=0){
      y_predicted_valid <- fm$yHat[(nrow(data)-(nrow(valid)+nrow(test))+1):(nrow(train)+nrow(valid))]
      y_actual_valid <- y[(nrow(data)-(nrow(valid)+nrow(test))+1):(nrow(train)+nrow(valid))] 
  }else{
      y_predicted_valid <- data.frame()
      y_actual_valid <- data.frame()
  }
  
  if (get_effect == TRUE){
    true_model <- function(newdata) {
      qtl <- rbind(data_qtl,newdata)
      pred <- c(y_test,rep(NA, nrow(newdata)))
      len_beg <- nrow(data_qtl)+ 1
      len_end <- nrow(data_qtl)+ nrow(newdata)
      
      D <- as.matrix(dist(qtl,method="euclidean"))^2
      D <- D/mean(D)
      h <- 1
      K <- exp(-h*D)
      
      f <- BGLR(y=pred,ETA=list(list(K=K,model='RKHS')),
                nIter=nIter,burnIn=burnIn,verbose=FALSE,saveAt='eig_')
      
      return(f[["yHat"]][len_beg:len_end])
    }
    
    predictor <- Predictor$new(NULL, data = data_qtl, y=fm[["yHat"]], predict.fun=true_model)
    
    effect <- data.frame()
    if(nrow(test) < Shapley_num){len <- nrow(test)}else{len <- Shapley_num}
    for(j in 1:len){
      shapley <- Shapley$new(predictor, x.interest = data_qtl[j+nrow(train)+nrow(valid), ], sample.size = 1)
      tmp <- data.frame(t(shapley$results[,1:2]))
      colnames(tmp) <- colnames(data)[2:(ncol(data)-1)]
      effect <- dplyr::bind_rows(effect, tmp[2,])
    }
    
    effect <- effect %>% mutate_all(as.numeric)
    effect <- colSums(abs(effect))
    effect <- data.frame(t(effect))
    #effect <- y_predicted
  }else{
    effect <- data.frame()
  }
  
  return(list(pearson, MSE, effect, y_predicted, y_predicted_valid, y_predicted_train))
  
}