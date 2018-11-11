library(plyr)
library(mvtnorm)
library(data.table)

#--------------------
# base experts
#--------------------
gen_all_h <- function(num_dim, num_base_models, scales, seed=0, h0=NULL){
    # Creates a total of num_base_models * |scales| models.
    require(plyr)
    require(mvtnorm)

    set.seed(seed) # made the models reproducible.

    # sample randomly on unit sphere
    normal_sample <- rmvnorm(num_base_models, mean = rep(0,num_dim))
    normal_sample <- normal_sample  / apply(normal_sample, 1, function(x){sqrt(sum(x^2))})

    if (!is.null(h0)){ # returned hypothesis will be close to h0 (>=0.7 correlation)
        normal_sample <- normal_sample * 0.7 + rep(h0,each=num_base_models)
        normal_sample <- normal_sample  / apply(normal_sample, 1, function(x){sqrt(sum(x^2))})
    }
    all_h_org <- normal_sample

    # scales up
    all_h <- ldply(lapply(scales,function(x){all_h_org * x}))
    all_h <- as.matrix(all_h)

    return(all_h)
}


#--------------------
# algorithm related
#--------------------
loss_func <- function
(h, # a vector of predictions
 y, # label
 loss = c('logistic','misclass','softlogistic')[1]
 ){
    if (loss == 'logistic'){ # overflow control
        ret = ifelse(-y*h>50, -y*h, log(1+exp(-y*h)))
        return (ret)
    } else if (loss == 'misclass'){
        return (as.numeric(y*h<0))
    } else if (loss == 'softlogistic'){
        ret = 1/(1+exp(y*h))
        return (ret)
    } else { stop }

}



mul_otb <- function(testX, testy, all_h, testk, opt_Its){

    ntest <- nrow(testX)
    r_per_h <- length(opt_Its)
    OTB1_curr <- 0; OTB1_curr_logistic <- 0
    OTB1_curr_softlogistic <- 0

    for (r in seq_len(r_per_h)){
        subX <- testX[testk==r,]; suby <- testy[testk==r]
        subpred <- subX %*% all_h[opt_Its[r],] 

        OTB1_curr <- OTB1_curr + sum(loss_func(subpred , suby, 'misclass'))
        OTB1_curr_logistic <- OTB1_curr_logistic + sum(loss_func(subpred , suby))
        OTB1_curr_softlogistic <- OTB1_curr_softlogistic + sum(loss_func(subpred , suby, 'softlogistic'))
    }

    return (c(OTB1_curr/ntest, OTB1_curr_logistic/ntest, OTB1_curr_softlogistic/ntest ))
}
