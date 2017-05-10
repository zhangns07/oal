source('0.init.R')

req_cost <- 0.4 # request cost, c in the paper
beta <- 3   # parameter in slack term
n_warmup <- 10    # warmup rounds, keep requesting label

#-------------------- 
# Load data
#-------------------- 
source('../datasets/skin.R')

#-------------------- 
# Input to algorithm
# iven X, y, all_h, all_thre
#-------------------- 
nT <- nrow(X)
r_per_h <- length(all_thre)
nh <- nrow(all_h)

for (rep in c(1:10)){

    set.seed(rep); shuffle <- sample(seq_len(nT),replace = FALSE)

    cum_obs <- matrix(0,nrow = nh, ncol = r_per_h) # cumulative obs for (h,r)
    cum_loss <- matrix(0,nrow = nh, ncol = r_per_h) # cumulative incurred loss for (h,r)
    cum_reg <- 0 # cumulative regret
    cum_label <- 0 # cumulative number of label requests

    # warmup
    for (i in seq_len(n_warmup)){
        x_t <- X[shuffle[i],]
        y_t <- y[shuffle[i]]
        pred_t <- all_h %*% t(t(x_t)) # prediction
        pred_loss_t <- loss_func(pred_t, y_t) # prediction loss

        curr_ret <- cond_loss_allpairs(all_thre, pred_t, pred_loss_t)
        cum_loss <- cum_loss + curr_ret[1:nh,]
        cum_obs <- cum_obs + curr_ret[(1:nh)+nh,]

    }

    cum_reg <- req_cost * n_warmup
    cum_label <-  n_warmup

    # start active learning
    RET <- matrix(c(0),ncol = 5) # book keeping
    for (i in c((1+n_warmup) : nrow(X))){
#    for (i in c((1+n_warmup) : 1000)){
        x_t <- X[shuffle[i],]
        y_t <- y[shuffle[i]]

        pred_t <- all_h %*% t(t(x_t))
        pred_loss_t <- loss_func(pred_t, y_t) # prediction loss

        # create pseudo label
        non_req_t <- apply(array(all_thre),1, function(x){(x - abs(pred_t))<=0})
        mu <- (cum_loss/cum_obs)[non_req_t]
        pred <- (matrix(rep(pred_t,r_per_h),ncol = r_per_h))[non_req_t]
        rm_idx <- cum_obs[non_req_t] ==0 
        pse_y_t <- gen_pseudo_label(pred[!rm_idx],mu[!rm_idx])
        pred_loss_pse_t <- loss_func(pred_t, pse_y_t) # prediction loss with pseudo label

        # update UCB with pseudo label
        curr_ret <- cond_loss_allpairs(all_thre, pred_t, pred_loss_pse_t)
        cum_loss <- cum_loss + curr_ret[1:nh,]
        cum_obs <- cum_obs + curr_ret[(1:nh)+nh,]
        UCB <- (cum_loss/cum_obs + sqrt(2*beta*log(i)/cum_obs))[non_req_t]

        # choose expert for this round
        h_idx <- (matrix(rep(seq_along(pred_t),r_per_h),ncol = r_per_h))[non_req_t]
        It <- which.min(UCB)
        h_It <- h_idx[It]

        if (any(non_req_t)){ # some pair do not request
            if (UCB[It] < req_cost){
                not_request <- TRUE
            } else {
                not_request  <- FALSE
            }
        } else {# all want to request
            not_request <- FALSE
        }

        if(not_request){
            cum_reg <- cum_reg + loss_func(pred[It],y_t)
        } else{ # request label
            cum_reg <- cum_reg + req_cost
            cum_label <- cum_label + 1
            if (y_t != pse_y_t){ # correct the bias from pseudo label
                true_curr_ret <- cond_loss_allpairs(all_thre, pred_t, pred_loss_t)
                cum_loss <- cum_loss - curr_ret[1:nh,] + true_curr_ret[1:nh,]
            }
        }

        if (i %% 50 ==0){
#            cat('num of rounds:',i,
#                ', num of labels:',cum_label,
#                ', min UCB:', min(UCB),'\n')

            RET <- rbind(RET,c(i,cum_label, h_It,min(UCB),cum_reg))
        }
    }
    filename <- paste0('ucbhp_rep',rep,'.csv')
    write.table(RET,filename, sep = ',',col = FALSE,row.names = FALSE)

}

# best expert inhindside
# have to make sure that best inhindside has \hat \mu < req_cost
min_err <- c()
for (req_cost in seq(0,1,0.1)){
    cum_loss_uncond <- matrix(0,nrow = nh, ncol = r_per_h)
    for (i in seq_len(nT)){
        x_t <- X[i,]
        y_t <- y[i]
        pred_t <- all_h %*% t(t(x_t))
        pred_loss_t <- loss_func(pred_t, y_t) # prediction loss

        curr_loss_uncond <- apply(array(all_thre),1, function(x){
                                      r <- x - abs(pred_t)
                                      ifelse(r>0, req_cost, pred_loss_t)})
        cum_loss_uncond <- cum_loss_uncond + curr_loss_uncond
    }

    best_pair  <- which.min(cum_loss_uncond)
    best_h <- best_pair %% nh
    best_r <- 1+ floor(best_pair / nh)
    min_err <- c(min_err, cum_loss_uncond[best_pair]/ nT)
}

# for logistic loss, req_cost has to be >= 0.5 to have 
# min(\hat \mu) <= req_cost 
# [1] 0.09303666 0.17965895 0.26628125 0.35290355 0.43333897 0.47872352
# [7] 0.52410806 0.56261959 0.56906095 0.57550231 0.58194367

