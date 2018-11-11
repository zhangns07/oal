#!/usr/bin/env Rscript
library(optparse)
library(plyr)
library(mvtnorm)
library(data.table)
library(caret)
source('0.init.R')
option_list <- list(make_option(c("-d", "--dataset"), type="character", default='skin',
                                 help="dataset file name"),
                    make_option(c("-f", "--datafolder"), type="character", default='../../datasets/RData',
                                help="dataset file name"),
                    make_option(c("-l", "--lognorm"), type="numeric", default=5,
                                help="log2 of maximal norm of model coefficients"),
                    make_option(c("-b", "--basemodel"), type="numeric", default=500,
                                help="number of base nmodels (unit norm)"),
                    make_option(c("-a", "--alg"), type="character", default='oneiwal',
                                help="oneiwal or oneminmax"),
                    make_option(c("-s", "--slack"), type="character", default='azuma',
                                help="type of slack term, azuma, or bernstein"),
                    make_option(c("-o", "--out_directory"), type="character", default=".",
                                help="whether to save output files")
                    )

opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser);

FLAGS <- opt
n_warmup <- 100    # warmup rounds, keep requesting label

#--------------------
# Load data
#--------------------
datafile <- paste0(FLAGS$datafolder, '/', FLAGS$dataset, '.RData')
load(datafile)
nT <- nrow(X)
beta <- 2

for (rep in c(1:20)){
    opt2 <- as.list(FLAGS) ;opt2$datafolder <- NULL ;opt2$otb <- NULL ;opt2$help <- NULL ;opt2$out_directory <- NULL
    basefilename <- paste0(paste0(names(opt2),'_',opt2), collapse = '_')
    filename <- paste0(FLAGS$out_directory,'/',basefilename, '_otb_rep',rep,'.csv')
    if(file.exists(filename)){next}

    set.seed(rep); shuffle <- sample(nrow(X),nrow(X),replace = FALSE)
    ntrain <- floor(nT * 0.8); ntest <- nT - ntrain

    trainX_tmp <- X[shuffle[1:ntrain],]; testX_tmp <- X[shuffle[-c(1:ntrain)],]
    col_to_keep <- apply(trainX_tmp, 2, var, na.rm=TRUE)  != 0
    trainX_tmp  <- trainX_tmp[,col_to_keep]; testX_tmp  <- testX_tmp[,col_to_keep];
    trainy <- y[shuffle[1:ntrain]];  testy <- y[shuffle[-c(1:ntrain)]]


    #traink <- k[shuffle[1:ntrain]];  testk <- k[shuffle[-c(1:ntrain)]]
    # rescale X and testX
    if(ncol(trainX_tmp) <=10){
        preprop <- preProcess(trainX_tmp,method=c('center','scale'))
    } else {
        preprop <- preProcess(trainX_tmp,method=c('center','scale','pca'),pcaComp=10)
    }
    trainX <- predict(preprop, trainX_tmp)
    testX <- predict(preprop, testX_tmp) 


    trainX <- cbind(1,as.matrix(trainX)); testX <- cbind(1,as.matrix(testX))
    traink <- rep(1,length(trainy));testk <- rep(1,length(testy))

    scales <- 2^seq(0,FLAGS$lognorm,1)
    all_h <- gen_all_h(num_dim=ncol(trainX), num_base_models=FLAGS$basemodel, scales)#, h0=h0)
    nh <- nrow(all_h)

    # When models' norm scales, scales thre as well, for each region separately.
    # upper bound of logistic loss
    trainl2norm <- sqrt(apply(trainX,1,function(x){sum(x^2)}))
    max_x_norm <- quantile(trainl2norm,0.95)
    M <- apply(array(max_x_norm),1,function(x){
                   ifelse(x * max(scales)>50, x * max(scales),
                          log(1+exp(x * max(scales))))})

    cum_loss_misclass <- 0 # cumulative 0/1 error of the prediction for this round
    cum_loss_logistic <- 0 # cumulative logistics error of prediction for this round
    cum_label <- 0 # cumulative number of label requests
    cumpt <- 0 # cumulative request prob

    checkpoint <- min(100,floor(nT / (100*10)) * 100)
    if (checkpoint==0){checkpoint <- 25}

    r_per_h <- 1
    cum_loss <- matrix(0,nrow=nh,ncol=r_per_h)
    Ht <- matrix(TRUE,nrow=nh,ncol=r_per_h)
    cum_samples <- rep(0,r_per_h)
    cum_obs <- rep(0, r_per_h)

    OTB_iwal <- matrix(c(0),ncol = 6) # book keeping
    last_i <- 0
    last_cum_label <- 0

    for (i in seq_len(ntrain)){
        x_t <- trainX[i,]; y_t <- trainy[i]; k_t <- traink[i]; M_t <- M[k_t]
        pred_t <- all_h %*% x_t
        cum_samples[k_t] <- cum_samples[k_t] +1
        if (i <= n_warmup){
            p_t <- 1
        } else{
            allpreds <- pred_t[Ht[,k_t]]
            if(opt$alg == 'oneiwal'){
                pred_max <- max(allpreds)
                pred_min <- min(allpreds)
                p_t <- max(log(1+exp(-pred_min)) - log(1+exp(-pred_max)),
                           log(1+exp(pred_max)) - log(1+exp(pred_min))) / M_t
            } else if (opt$alg == 'oneminmax'){
                pred_max <- max(allpreds[allpreds>0])
                pred_min <- min(allpreds[allpreds>0])
                p_t_pos <- max(log(1+exp(-pred_min)) - log(1+exp(-pred_max)),
                           log(1+exp(pred_max)) - log(1+exp(pred_min))) / M_t

                pred_max <- max(allpreds[allpreds<0])
                pred_min <- min(allpreds[allpreds<0])

                p_t_neg <- max(log(1+exp(-pred_min)) - log(1+exp(-pred_max)),
                           log(1+exp(pred_max)) - log(1+exp(pred_min))) / M_t

                p_t <- min(p_t_pos, p_t_neg)
            }
        }
        cumpt <- cumpt+p_t

        Q_t <- as.numeric(runif(1) < p_t)
        if (Q_t > 0){
            cum_obs[k_t] <- cum_obs[k_t]+1; cum_label <- cum_label + 1 
            cum_loss[,k_t] <- cum_loss[,k_t] + loss_func(pred_t,y_t,'logistic')/(M_t*p_t) # importance weighted cum_loss
            min_err <- min((cum_loss[,k_t])[Ht[,k_t]])
            T_t <- cum_samples[k_t]

            if(opt$slack=='azuma'){
                #slack_t <- sqrt(T_t*log(T_t+1)*beta) # a more aggresive slack term than IWAL paper
                slack_t <- sqrt(T_t) # a more aggresive slack term than IWAL paper
            } else if (opt$slack=='bernstein'){
                slack_t <- sqrt(cumpt)
            }
            Ht[,k_t] <- (Ht[,k_t]  & cum_loss[,k_t] <= min_err + slack_t)
        } 

        if (i!= last_i & cum_label != last_cum_label & (i %% checkpoint ==0 || cum_label %% checkpoint == 0)){
            last_i <-  i
            last_cum_label <- cum_label
            cat('num of rounds:',i,
                ', num of labels:',cum_label,'\n')

            opt_Its <- rep(0,r_per_h)
            for(r in c(1:r_per_h)){ opt_Its[r] <- (seq_len(nh)[Ht[,r]])[which.min((cum_loss[,r])[Ht[,r]])] }
            curr_otb <- mul_otb(testX, testy, all_h, testk, opt_Its)
            OTB_iwal <- rbind(OTB_iwal, c(i, i, cum_label, curr_otb))
        }

        if (cum_label > 5000){break}
    }

    opt_Its <- rep(0,r_per_h)
    for(r in c(1:r_per_h)){ opt_Its[r] <- (seq_len(nh)[Ht[,r]])[which.min((cum_loss[,r])[Ht[,r]])] }
    curr_otb <- mul_otb(testX, testy, all_h, testk, opt_Its)
    OTB_iwal <- rbind(OTB_iwal, c(i, i, cum_label, curr_otb))

    # save to file
    opt2 <- as.list(FLAGS)
    opt2$datafolder <- NULL
    opt2$otb <- NULL
    opt2$help <- NULL
    opt2$out_directory <- NULL
    basefilename <- paste0(paste0(names(opt2),'_',opt2), collapse = '_')

    filename <- paste0(FLAGS$out_directory,'/',basefilename, '_otb_rep',rep,'.csv')
    colnames(OTB_iwal) <- c('round','accepts', 'labels','loss_misclass','loss_logistic','loss_logistic')
    write.table(OTB_iwal,filename, sep = ',', row.names = FALSE)
}


#numline = 2:150
#plot(OTB_iwal_one[numline,3],OTB_iwal_one[numline,4],type='l')
#lines(OTB_iwal_minmax[numline,3],OTB_iwal_minmax[numline,4],col='red')
