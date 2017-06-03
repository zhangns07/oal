# 
source('0.init.R')
#---------- 
# setup
xd <- 2 # feature dimension
h_per_d <- 30    # for each dimension, number of equally spaces value to take 
                # between (-1,1). So the total number of h is h_per_d^d.
M <-  h_per_d^xd    # number of predictors h
r_per_h <- 30 #number of r to pair with each h

set.seed(0);true_h <- 1-2*runif(xd)
all_h <- gen_h(h_per_d, xd, max_norm = 2)
all_thres <- seq(0,2,length.out = r_per_h)
X <- matrix(1 - 2*runif(xd * 5000),ncol = xd)
y <- ifelse(c(X %*% true_h ) >0, 1, -1)


