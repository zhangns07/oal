#!/usr/bin/env Rscript
library(optparse)
library(plyr)
library(mvtnorm)
library(data.table)
library(caret)
library(e1071)
library(rpart)
source('0.init.R')
option_list <- list(make_option(c("-d", "--dataset"), type="character", default='skin',
                                 help="dataset file name"),
                    make_option(c("-f", "--datafolder"), type="character", default='../../datasets/RData',
                                help="dataset file name"),
                    make_option(c("-l", "--lognorm"), type="numeric", default=5,
                                help="log2 of maximal norm of model coefficients"),
                    make_option(c("-b", "--basemodel"), type="numeric", default=500,
                                help="number of base nmodels (unit norm)"),
                    make_option(c("-a", "--alg"), type="character", default='rbiwal',
                                help="type of algorithm, rbiwal, optrb, naiveiwal"),
                    make_option(c("-r", "--region"), type="character", default='circle',
                                help="type of regions, circle, hyper, or cluster"),
                    make_option(c("-n", "--nregion"), type="numeric", default=10,
                                help="number of regions"),
                    make_option(c("-o", "--out_directory"), type="character", default=".",
                                help="whether to save output files"),
                    make_option(c("-t", "--test"), type="logical", default=FALSE,
                                help="whether it is testing"),
                    make_option(c("-s", "--slack"), type="character", default='azuma',
                                help="type of slack term, azuma, or bernstein"),
                    make_option(c("-c", "--const"), type="numeric", default=1,
                                help="constant in front of slack term")
                    )

opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser);
n_warmup <- 100 # warmup rounds, keep requesting label

#--------------------
# Load data
#--------------------
datafile <- paste0(opt$datafolder, '/', opt$dataset, '.RData')
load(datafile)
nT <- nrow(X)
beta <- 2 ## constant in slack term
if(opt$test){ maxrep <-1 } else { maxrep <- 20}

for (rep in c(1:maxrep)){
    opt2 <- as.list(opt) ;opt2$datafolder <- NULL ;opt2$otb <- NULL ;opt2$help <- NULL ;opt2$out_directory <- NULL; opt2$test <- NULL
    basefilename <- paste0(paste0(names(opt2),'_',opt2), collapse = '_')
    filename <- paste0(opt$out_directory,'/',basefilename, '_otb_rep',rep,'.csv')
    if(file.exists(filename)){cat('next\n');next}

    set.seed(rep); shuffle <- sample(nrow(X),nrow(X),replace = FALSE)
    ntrain <- floor(nT * 0.8); ntest <- nT - ntrain

    trainX_tmp <- X[shuffle[1:ntrain],]; testX_tmp <- X[shuffle[-c(1:ntrain)],]
    col_to_keep <- apply(trainX_tmp, 2, var, na.rm=TRUE)  != 0
    trainX_tmp  <- trainX_tmp[,col_to_keep]; testX_tmp  <- testX_tmp[,col_to_keep];
    trainy <- y[shuffle[1:ntrain]];  testy <- y[shuffle[-c(1:ntrain)]]

    # rescale X and testX
    if(ncol(trainX_tmp) <=10){
        preprop <- preProcess(trainX_tmp,method=c('center','scale'))
    } else {
        preprop <- preProcess(trainX_tmp,method=c('center','scale','pca'),pcaComp=10)
    }
    trainX <- predict(preprop, trainX_tmp)
    testX <- predict(preprop, testX_tmp) 

    if(opt$region == 'circle'){
        trainl2norm <- sqrt(apply(trainX,1,function(x){sum(x^2)}))
        testl2norm <- sqrt(apply(testX,1,function(x){sum(x^2)}))
        thre_minmax <- log(quantile(trainl2norm,c(0.05,0.95)))
        if(opt$nregion>=3){
            thres <- c(0,exp(seq(thre_minmax[1],thre_minmax[2],length.out = opt$nregion-1)),Inf)
        } else {
            thres <- c(0,quantile(trainl2norm,0.1),Inf)
        }

        traink <- as.numeric(cut(trainl2norm, thres, include.lowest=TRUE))
        testk <- as.numeric(cut(testl2norm, thres, include.lowest=TRUE))
        r_per_h <- opt$nregion

    } else if (opt$region == 'hyper'){
        rand_planes <- matrix(rnorm(ncol(trainX)*opt$nregion), ncol=ncol(trainX))
        rand_planes <- rand_planes / sqrt(rowSums(rand_planes^2))

        traindist <- abs(as.matrix(trainX) %*% t(rand_planes))
        testdist <- abs(as.matrix(testX) %*% t(rand_planes))
        traink <- apply(traindist,1,which.min)
        testk <- apply(testdist,1,which.min)
        r_per_h <- opt$nregion

    } else if (opt$region == 'tree'){
        tmp <- data.frame(cbind(y=trainy,trainX)[1:n_warmup,])
        obj <- rpart(y~.,data=tmp,method='class',control=rpart.control(minsplit=n_warmup/(opt$nregion-1)))
        obj2 <- obj
        obj2$frame$yval2 <- NULL
        obj2$frame[grepl('leaf',obj2$frame$var),]$yval <- sort(unique(obj2$where))

        traink <- predict(obj2,data.frame(trainX),type='matrix')
        testk <- predict(obj2,data.frame(testX),type='matrix')
        #x <- table(traink,trainy); cbind(x / rowSums(x), rowSums(x))

        # convert to 1,2,..
        traink <- as.numeric(as.factor(traink))
        testk <- as.numeric(as.factor(testk))

        r_per_h <- length(unique(obj2$where))
    } else if (opt$region=='cluster'){
        cluster <- kmeans(trainX,centers = opt$nregion, nstart=2*opt$nregion)
        traink <- fitted(cluster,method='classes')
        testk <- apply(testX, 1, function(x){
                   which.min(colSums((t(cluster$centers) - x)^2)) })
        r_per_h <- opt$nregion
    }

    trainX <- cbind(1,as.matrix(trainX)); testX <- cbind(1,as.matrix(testX))
    scales <- 2^seq(0,opt$lognorm,1)
    all_h <- gen_all_h(num_dim=ncol(trainX), num_base_models=opt$basemodel, scales)#, h0=h0)
    nh <- nrow(all_h)

    # When models' norm scales, scales thre as well, for each region separately.
    # upper bound of logistic loss
    trainl2norm <- sqrt(apply(trainX,1,function(x){sum(x^2)}))
    max_x_norm <- quantile(trainl2norm,0.95)
    M <- ifelse(max_x_norm * max(scales)>50, max_x_norm * max(scales), log(1+exp(max_x_norm * max(scales))))
    M <- rep(M,r_per_h)

    checkpoint <- min(100,floor(nT / (100*10)) * 100)
    if (checkpoint==0){checkpoint <- 25}

    cum_label <- 0 # cumulative number of label requests
    cum_accept <- 0 # cumulative number of label requests
    cum_loss <- matrix(0,nrow=nh,ncol=r_per_h)
    Ht <- matrix(TRUE,nrow=nh,ncol=r_per_h)
    region_samples <- rep(0,r_per_h) # incoming per region
    region_accepts <- rep(0, r_per_h) # accpeted per region
    region_labels <- rep(0, r_per_h) # labels per region
    region_cumpt <- rep(0, r_per_h) # cumulative request prob

    #cum_samples <- rep(0,r_per_h)
    #cum_obs <- rep(0, r_per_h)

    OTB_iwal <- matrix(c(0),ncol = 6) # book keeping
    last_i <- 0
    last_cum_label <- 0; last_cum_accept <- 0

    if(opt$alg == 'optrb'){
        tmppk <- table(traink);tmppk<- tmppk/sum(tmppk)
    }


    if(opt$test){ maxi <-1000 } else { maxi<- ntrain}
    for (i in seq_len(maxi)){
        x_t <- trainX[i,]; y_t <- trainy[i]; k_t <- traink[i]; M_t <- M[k_t]
        pred_t <- all_h %*% x_t
        region_samples[k_t] <- region_samples[k_t] +1

        if(opt$alg == 'rbiwal'){
            cum_accept <- cum_accept+1
            if (i <= n_warmup){
                p_t <- 1
            } else{
                pred_max <- max(pred_t[Ht[,k_t]])
                pred_min <- min(pred_t[Ht[,k_t]])
                p_t <- max(log(1+exp(-pred_min)) - log(1+exp(-pred_max)),
                           log(1+exp(pred_max)) - log(1+exp(pred_min))) / M_t
            }
            region_cumpt[k_t] <- region_cumpt[k_t]  + p_t

            Q_t <- as.numeric(runif(1) < p_t)
            if (Q_t > 0){
                region_labels[k_t] <- region_labels[k_t]+1; cum_label <- cum_label + 1 
                cum_loss[,k_t] <- cum_loss[,k_t] + loss_func(pred_t,y_t,'logistic')/(M_t*p_t) # importance weighted cum_loss
                min_err <- min((cum_loss[,k_t])[Ht[,k_t]])
                T_t <- region_samples[k_t]
                if(opt$slack=='azuma'){
                    #slack_t <- sqrt(T_t*log((T_t+1)*r_per_h)*beta) # a more aggresive slack term than IWAL paper
                    slack_t <- opt$const * sqrt(T_t) # a more aggresive slack term than IWAL paper
                } else if (opt$slack == 'bernstein'){
                    slack_t <- opt$const * sqrt(region_cumpt[k_t])
                }
                Ht[,k_t] <- (Ht[,k_t]  & cum_loss[,k_t] <= min_err + slack_t)
            } 
        } else if (opt$alg == 'optrb'){

            if (i <= n_warmup){
                A_t <- 1; p_t <- 1; Q_t <- 1
            } else{
                if(region_samples[k_t] <= n_warmup/r_per_h){
                    A_t <- 1
                } else {
                    # compute alpha_k on the fly
                    tmpck <- region_cumpt / region_accepts
                    tmpalp <- tmpck/tmppk
                    alp <- tmpalp^(1/3)/max(tmpalp^(1/3))
                    A_t <- as.numeric(runif(1) < alp[k_t])
                }

                if(A_t > 0 ){
                    pred_max <- max(pred_t[Ht[,k_t]])
                    pred_min <- min(pred_t[Ht[,k_t]])
                    p_t <- max(log(1+exp(-pred_min)) - log(1+exp(-pred_max)),
                               log(1+exp(pred_max)) - log(1+exp(pred_min))) / M_t
                    p_t <- pmin(p_t, 1)
                    Q_t <- as.numeric(runif(1) < p_t)
                }
            }

            if(A_t > 0){
                region_cumpt[k_t] <- region_cumpt[k_t]  + p_t
                region_accepts[k_t] <- region_accepts[k_t]+1
                cum_accept <- cum_accept+1

                if (Q_t > 0){
                    region_labels[k_t] <- region_labels[k_t]+1; cum_label <- cum_label + 1 
                    cum_loss[,k_t] <- cum_loss[,k_t] + loss_func(pred_t,y_t,'logistic')/(M_t*p_t) # importance weighted cum_loss
                    min_err <- min((cum_loss[,k_t])[Ht[,k_t]])
                    T_t <- region_accepts[k_t]
                    if(opt$slack=='azuma'){
                        #slack_t <- sqrt(T_t*log((T_t+1)*r_per_h)*beta) # a more aggresive slack term than IWAL paper
                        slack_t <- opt$const * sqrt(T_t) # a more aggresive slack term than IWAL paper
                    } else if (opt$slack == 'bernstein'){
                        slack_t <- opt$const * sqrt(region_cumpt[k_t])
                    }
                    Ht[,k_t] <- (Ht[,k_t]  & cum_loss[,k_t] <= min_err + slack_t)
                } 
            }
        } else if (opt$alg == 'naiveiwal'){

            cum_accept <- cum_accept+1
            if (i <= n_warmup){
                p_t <- 1
            } else{
                pred_max <- max(pred_t[Ht[,k_t]])
                pred_min <- min(pred_t[Ht[,k_t]])
                p_t <- max(log(1+exp(-pred_min)) - log(1+exp(-pred_max)),
                           log(1+exp(pred_max)) - log(1+exp(pred_min))) / M_t
            }

            region_cumpt[k_t] <- region_cumpt[k_t]  + p_t
            Q_t <- as.numeric(runif(1) < p_t)
            if (Q_t > 0){
                region_labels[k_t] <- region_labels[k_t]+1; cum_label <- cum_label + 1 
                cum_loss[,k_t] <- cum_loss[,k_t] + loss_func(pred_t,y_t,'logistic')/(M_t*p_t) # importance weighted cum_loss
                min_err <- min((cum_loss[,k_t])[Ht[,k_t]])
                T_t <- i
                if(opt$slack=='azuma'){
                    #slack_t <- sqrt(T_t*log(T_t+1)*beta*r_per_h) # a more aggresive slack term than IWAL paper
                    slack_t <- opt$const * sqrt(T_t*r_per_h) # [wrong a more aggresive slack term than IWAL paper
                } else if (opt$slack == 'bernstein'){
                    slack_t <- opt$const * sqrt(region_cumpt[k_t]*r_per_h)
                }

                Ht[,k_t] <- (Ht[,k_t]  & cum_loss[,k_t] <= min_err + slack_t)
            } 
        }

        if (i!= last_i & cum_label != last_cum_label & cum_accept != last_cum_accept & 
            (i %% checkpoint ==0 || cum_label %% checkpoint == 0 || cum_accept %% checkpoint == 0)){
            last_i <-  i
            last_cum_label <- cum_label
            laxt_cum_accept <- cum_accept
            cat('num of rounds:',i,
                ', num of labels:',cum_label,'\n')

            opt_Its <- rep(0,r_per_h)
            for(r in c(1:r_per_h)){ opt_Its[r] <- (seq_len(nh)[Ht[,r]])[which.min((cum_loss[,r])[Ht[,r]])] }
            curr_otb <- mul_otb(testX, testy, all_h, testk, opt_Its)
            OTB_iwal <- rbind(OTB_iwal, c(i, cum_accept, cum_label, curr_otb))
        }
    }

    opt_Its <- rep(0,r_per_h)
    for(r in c(1:r_per_h)){ opt_Its[r] <- (seq_len(nh)[Ht[,r]])[which.min((cum_loss[,r])[Ht[,r]])] }
    curr_otb <- mul_otb(testX, testy, all_h, testk, opt_Its)
    OTB_iwal <- rbind(OTB_iwal, c(i, cum_accept, cum_label, curr_otb))

    # save to file
    opt2 <- as.list(opt)
    opt2$datafolder <- NULL ;opt2$otb <- NULL ;opt2$help <- NULL ;opt2$out_directory <- NULL; opt2$test <- NULL
    basefilename <- paste0(paste0(names(opt2),'_',opt2), collapse = '_')

    filename <- paste0(opt$out_directory,'/',basefilename, '_otb_rep',rep,'.csv')
    colnames(OTB_iwal) <- c('round','accepts', 'labels','loss_misclass','loss_logistic', 'loss_softlogistic')
    write.table(OTB_iwal,filename, sep = ',', row.names = FALSE)
}


if(1==0){
    for(r in c(1:r_per_h)){ 
        ret <- rep(0,6)
        for(j in c(1:6)){
            tokeep <- c(1:500)+(j-1)*500
            ret[j] <- min(cum_loss[tokeep,r][Ht[tokeep,r]])
        }
        cat(ret,which.min(ret),'\n')
    }
}

