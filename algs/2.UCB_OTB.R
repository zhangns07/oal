#!/usr/bin/env Rscript
# Run UCB type algorithms.
# Perform online to batch (OTB) by the end.
library("optparse")

option_list <- 
    list(make_option(c("-d", "--dataset"), type="character", default=NULL, 
                      help="dataset file name, one of skin, shuttle, and HTRU_2"),
         make_option(c("-c", "--cost"), type="numeric", default=NULL,
                     help="request cost"),
         make_option(c("-a", "--algorithm"), type="character", default=NULL,
                     help="algoirthm, ucb, ucblcb, ucbfl"),
         make_option(c("-o", "--otb"), type="logical", default=TRUE,
                     help="whether to run online to batch")); 

opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser);

req_cost <- opt$cost
dataset <- opt$dataset
alg <- opt$algorithm

source('0.init.R')
beta <- 2.05   # parameter in slack term
n_warmup <- 10    # warmup rounds, keep requesting label

#-------------------- 
# Load data
#-------------------- 

source(paste0('1.data.',dataset,'.R'))

#-------------------- 
# Input to algorithm
# X, y, all_h, all_thre
#-------------------- 
r_per_h <- length(all_thre)
nh <- nrow(all_h)
if(dataset == 'skin'){
    nT <- 100000
    ntest <- 5000
} else {
    nT <- 0.8 * nrow(X)
    ntest <- 0.2 * nrow(X)
}

for (rep in c(1:10)){

    set.seed(rep); shuffle <- sample(nrow(X),nrow(X),replace = FALSE)
    shuffle_test <- shuffle[(nT+1):nrow(X)]

    cum_obs <- matrix(1e-5, nrow = nh, ncol = r_per_h) # cumulative obs for (h,r)
    cum_loss <- matrix(1e-5, nrow = nh, ncol = r_per_h) # cumulative conditional loss for (h,r)

    cum_loss_misclass <- 0 # cumulative 0/1 error of the prediction for this round
    cum_loss_logistic <- 0 # cumulative logistics error of prediction for this round
    cum_loss_al <- 0 # cumulative active learning loss: req_cost if request, loss if not request
    cum_label <- 0 # cumulative number of label requests

    # warmup
    for (i in seq_len(n_warmup)){
        x_t <- X[shuffle[i],]
        y_t <- y[shuffle[i]]
        pred_t <- all_h %*% x_t # prediction
        pred_loss_t <- loss_func(pred_t, y_t) # prediction loss

        curr_ret <- cond_loss_allpairs(all_thre, pred_t, pred_loss_t)
        cum_loss <- cum_loss + curr_ret[1:nh,]
        cum_obs <- cum_obs + curr_ret[(1:nh)+nh,]
    }

    cum_loss_al <- req_cost * n_warmup
    cum_label <-  n_warmup


    # start active learning
    num_experts_tokeep <- nh * r_per_h * 0.1
    num_orderings <- floor(nT / 100)

    RET <- matrix(c(0),ncol = 8) # book keeping
    OTB <- matrix(c(0),ncol = 4) # book keeping
    H_IDX <- (matrix(rep(seq_len(nh),r_per_h),ncol = r_per_h))
    R_IDX <- (matrix(rep(seq_along(all_thre),each = nh),ncol = r_per_h))
    ORDERING <- matrix(c(0), ncol = num_experts_tokeep, nrow = num_orderings)
    HITS <- rep(0, num_orderings)
    RITS <- rep(0, num_orderings)

    for (i in c((1+n_warmup) : nT)){
        x_t <- X[shuffle[i],]
        y_t <- y[shuffle[i]]

        pred_t <- all_h %*% x_t
        pred_loss_t <- loss_func(pred_t, y_t) # prediction loss
        non_req_t <- apply(array(all_thre),1, function(x){(x - abs(pred_t))<=0})

        UCB <- (cum_loss/cum_obs + sqrt(2*beta*log(i)/cum_obs))[non_req_t]
        if(alg == 'ucb'){
            LCB <- (cum_loss/cum_obs + sqrt(2*beta*log(i)/cum_obs))[non_req_t]
        } else if (alg == 'ucblcb') {
            LCB <- (cum_loss/cum_obs - sqrt(2*beta*log(i)/cum_obs))[non_req_t]
        } else if (alg == 'ucbfl'){
            LCB <- (cum_loss/cum_obs)[non_req_t]
        }

        # choose expert for this round
        h_idx <- H_IDX[non_req_t]
        r_idx <- R_IDX[non_req_t]
        It <- which.min(LCB)
        h_It <- h_idx[It] 
        r_It <- r_idx[It]
        if (any(non_req_t)){ # some pair do not request
            #not_request <- ifelse(UCB[It] < req_cost,TRUE,FALSE)
            not_request <- ifelse(UCB[which.min(UCB)] < req_cost,TRUE,FALSE)
        } else {# all pairs request
            not_request <- FALSE
        }

        if(not_request){ # not request
            cum_loss_al <- cum_loss_al + pred_loss_t[h_It]
        } else{ # request 
            cum_loss_al <- cum_loss_al + req_cost
            cum_label <- cum_label + 1
            true_curr_ret <- cond_loss_allpairs(all_thre, pred_t, pred_loss_t)
            cum_loss <- cum_loss  + true_curr_ret[1:nh,]
            cum_obs <- cum_obs + true_curr_ret[(1:nh)+nh,]
        }

        cum_loss_misclass  <- cum_loss_misclass + loss_func(pred_t[h_It],y_t,'misclass')
        cum_loss_logistic  <- cum_loss_logistic + loss_func(pred_t[h_It],y_t)

        if (i %% 100 ==0){
            cond_mu <- (cum_loss/cum_obs)
            cond_mu_order <- order(cond_mu)
            which_best_nonreq <- which(cond_mu_order <= nh)[1]
            curr_ordering <- cond_mu_order[1:which_best_nonreq]
            curr_ordering <- c(curr_ordering, rep(0,1500 - length(curr_ordering)))
            ORDERING[i/100, ] <- curr_ordering

            HITS[i/100] <- h_It
            RITS[i/100] <- r_It
        }

        if (i %% 1000 ==0){
            cat('i:',i, 
                ', num of labels:',cum_label,
                ', expert:',h_It,
                ', min UCB:', min(UCB),
                ', misclass loss/i:', cum_loss_misclass/i,
                ', logistic loss/i:', cum_loss_logistic/i,
                '\n')

            if (length(h_It)==0){ h_It <- 0;}
            RET <- rbind(RET,c(i,cum_label, h_It, r_It, UCB[which.min(UCB)],
                               cum_loss_al, cum_loss_misclass, cum_loss_logistic))

            if(opt$otb){
                OTB1_curr <- 0
                OTB2_curr <- 0
                OTB3_curr <- 0
                cond_mu <- (cum_loss/cum_obs)

                for(j in c(1:ntest)){
                    x_test <- X[shuffle_test[j],]
                    y_test <- y[shuffle_test[j]]
                    pred_t <- all_h %*% x_test
                    non_req_t <- apply(array(all_thre),1, function(x){(x - abs(pred_t))<=0})
                    pred_nonreq <-  (matrix(rep(pred_t,r_per_h),ncol = r_per_h))[non_req_t] # pred among non-requesters

                    # OTB1: follow awake leader
                    weights <- cond_mu[non_req_t]
                    OTB1_w <- as.numeric(weights == min(weights)) # weights for OTB
                    OTB1_curr <- OTB1_curr + loss_func(sum(pred_nonreq * OTB1_w) , y_test, 'misclass')

                    # OTB2: averaging orderings
                    avg_pred <- 0
                    for (k in c(1:floor(i/100))){
                        curr_ordering <- ORDERING[k,]
                        curr_h <- which(non_req_t[curr_ordering] == TRUE)[1]
                        curr_pred <- pred_t[H_IDX[curr_ordering[curr_h]]]
                        avg_pred <- avg_pred + curr_pred
                    }
                    OTB2_curr <- OTB2_curr + loss_func(avg_pred , y_test, 'misclass')

                    # OTB3: averaging past awake experts
                    avg_pred <- 0
                    num_awake <- 0
                    for (k in c(1:floor(i/100))){
                        h_It <- HITS[k]
                        r_It <- RITS[k]
                        if (pred_t[h_It] > all_thre[r_It]){
                            num_awake <- num_awake + 1
                            avg_pred <- avg_pred + pred_t[h_It]
                        }
                    }
                    if (num_awake == 0){
                        OTB3_curr <- OTB3_curr + loss_func(sum(pred_nonreq * OTB1_w) , y_test, 'misclass')
                    } else {
                        OTB3_curr <- OTB3_curr + loss_func(avg_pred, y_test, 'misclass')
                    }
                }

                OTB <- rbind(OTB,c(i,OTB1_curr/ntest, OTB2_curr/ntest, OTB3_curr/ntest))
                cat('i:',i, 
                    ', OTB:', OTB1_curr/ntest, OTB2_curr/ntest, OTB3_curr/ntest,
                    '\n')

            }
        }
    }

    filename <- paste0(dataset,'_',alg,'_req',req_cost,'_rep',rep,'.csv')
    write.table(RET,filename, sep = ',',col = FALSE,row.names = FALSE)

    if(opt$otb){
        filename <- paste0(dataset, '_otb_', alg,'_req',req_cost,'_rep',rep,'.csv')
        write.table(OTB,filename, sep = ',',col = FALSE,row.names = FALSE)
    }
}

