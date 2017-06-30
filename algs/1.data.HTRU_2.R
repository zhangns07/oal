library(plyr)
library(mvtnorm)
library(data.table)

HTRU <- fread('../datasets/HTRU_2.csv')

y <- 2 * HTRU[,V9] - 1
X <- HTRU[,-9]
X <- scale(X) 
X <- cbind(1,X)

# experts
# 1. sample on the uniform surface, by sample from gaussian and normalize
# 2. scale
num_base_models <- 500
num_dim <- 9
set.seed(12345) # made the models consistent 
normal_sample <- rmvnorm(num_base_models, mean = rep(0,num_dim))
normal_sample <- normal_sample  / apply(normal_sample, 1, function(x){sqrt(sum(x^2))})
all_h_org <- normal_sample 

scales <- c(2^seq(0, 5, 1))
all_h <- ldply(lapply(scales,function(x){all_h_org * x}))
all_h <- as.matrix(all_h)
all_thre <- unique(c(seq(0,0.1,0.01),seq(0.1, 0.3,0.1))) * max(scales)

if (1==0){
    # look at baselines
    tmpdata <- HTRU
    logml <- glm(V9 ~   V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 , data = tmpdata,family = 'binomial')
    sqrt(sum(logml$coefficients^2))
    sum((predict(logml,type = 'response') - 0.5) * y< 0) / length(y)


    nh <- nrow(all_h)
    nT <- nrow(X_train)
    batch_size <- 100
    train_cum_loss0 <- rep(0,nh)
    train_cum_loss1 <- rep(0,nh)

    for (i in seq_len(100)){
        Xbatch <- X[seq(((i-1)*batch_size+1) , i*batch_size),] 
        ybatch <- y[seq(((i-1)*batch_size+1) , i*batch_size)]

        train_cum_loss0 <- train_cum_loss0 + 
        apply(all_h %*% t(Xbatch) * rep(ybatch, each = nh), 1, 
              function(x){sum(log(1+exp(-x)))})

        train_cum_loss1 <- train_cum_loss1 + 
        apply(all_h %*% t(Xbatch) * rep(ybatch, each = nh), 1, 
              function(x){sum(x<0)})
    }

    It <- which.min(train_cum_loss0)
    train_cum_loss1[It] / (100*batch_size)
    min(train_cum_loss1) / (100*batch_size)

}
