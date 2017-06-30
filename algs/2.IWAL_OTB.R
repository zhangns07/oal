#!/usr/bin/env Rscript
#----------
# IWAL-OTB: use h_It, the best after that round
# Need: h_It
# Compute on heldout data after algorithm finished.
library("optparse")

option_list <- list(
                    make_option(c("-d", "--dataset"), type="character", default=NULL, 
                                help="dataset file name, one of skin, shuttle, and HTRU_2"),
                    make_option(c("-c", "--cost"), type="numeric", default=NULL,
                                help="request cost"),
                    make_option(c("-a", "--algorithm"), type="character", default='iwal',
                                help="algoirthm, iwal"),
                    make_option(c("-o", "--otb"), type="logical", default=TRUE,
                                help="whether to run online to batch")
                    ); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

req_cost <- opt$cost
dataset <- opt$dataset
alg <- opt$algorithm

source('0.init.R')

#-------------------- 
# Load data
#-------------------- 

source(paste0('1.data.',dataset,'.R'))

x_range <- c(quantile(abs(X),0.95))
names(x_range) <- NULL
M <- log(1+exp(sqrt(3)*x_range*max(scales))) # upper bound of logistic loss

#--------------------
# input
# X,y
# all_h, max_dis
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

for (rep in  c(1:10)){
    set.seed(rep); shuffle <- sample(nrow(X),nrow(X),replace = FALSE)
    Xtest <- as.matrix(X[shuffle[(nT+1):(nT+ntest)],])
    ytest <- y[shuffle[(nT+1):(nT+ntest)]]


    cum_loss_misclass <- 0 # cumulative 0/1 error of the prediction for this round
    cum_loss_logistic <- 0 # cumulative logistics error of prediction for this round
    cum_loss_al <- 0 # cumulative active learning loss: req_cost if request, loss if not request
    cum_label <- 0 # cumulative number of label requests

    cum_loss <- rep(0,nh)
    Ht <- rep(TRUE,nh)

    RET_iwal <- matrix(c(0),ncol = 7) # book keeping
    OTB_iwal <- matrix(c(0),ncol = 2) # book keeping

    for (i in seq_len(nT)){
        x_t <- X[shuffle[i],]
        y_t <- y[shuffle[i]]

        pred_t <- all_h %*% x_t

#        loss0 <- (loss_func(pred_t,-1, 'logistic'))[Ht] / M
#        loss1 <- (loss_func(pred_t,1,'logistic'))[Ht] / M
#        p_t <- max_dis(loss0,loss1)

        pred_max <- max(pred_t[Ht])
        pred_min <- min(pred_t[Ht])
        p_t <- max(log(1+exp(-pred_min)) - log(1+exp(-pred_max)),
                  log(1+exp(pred_max)) - log(1+exp(pred_min))) / M

        Q_t <- as.numeric(runif(1) < p_t)

        if (Q_t > 0){
            cum_loss_al <- cum_loss_al + req_cost
            cum_label <- cum_label + 1 

            cum_loss <- cum_loss + loss_func(pred_t,y_t,'logistic')/(M*p_t) # importance weighted cum_loss
            min_err <- min(cum_loss[Ht])
            slack_t <- sqrt(i*log(i+1)) # a more aggresive slack term than IWAL paper
            Ht <- (Ht  & cum_loss <= min_err + slack_t)
        } else{
            It <- (seq_len(nh)[Ht])[which.min(cum_loss[Ht])] # best expert for this round
            cum_loss_al <- cum_loss_al + loss_func(pred_t[It], y_t)
        }

        It <- (seq_len(nh)[Ht])[which.min(cum_loss[Ht])] # best expert for this round
        cum_loss_misclass <- cum_loss_misclass + loss_func(pred_t[It],y_t,'misclass')
        cum_loss_logistic <- cum_loss_logistic + loss_func(pred_t[It],y_t)

        if (i %% 1000 ==0){
            cat('num of rounds:',i,
                ', num of labels:',cum_label,
                ', expert:',It,
                ', misclass_loss/i:', cum_loss_misclass/i,
                ', logistic_loss/i:', cum_loss_logistic/i,'\n')

            RET_iwal <- rbind(RET_iwal,c(i,cum_label,It, p_t,cum_loss_al ,cum_loss_misclass, cum_loss_logistic))

            if(opt$otb){
                OTB_iwal <- rbind(OTB_iwal,
                                  c(i,sum(Xtest  %*% all_h[It,] * ytest < 0 )/ntest))
            }
        }
    }

    filename <- paste0(dataset,'_',alg,'_req',req_cost,'_rep',rep,'.csv')
    write.table(RET_iwal,filename, sep = ',',col = FALSE,row.names = FALSE)

    if(opt$otb){
        filename <- paste0(dataset, '_otb_', alg,'_req',req_cost,'_rep',rep,'.csv')
        write.table(OTB_iwal,filename, sep = ',',col = FALSE,row.names = FALSE)
    }
}


