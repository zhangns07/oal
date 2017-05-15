source('0.init.R')

req_cost <- 0.9 # request cost, this is just a starting point.
## Every 1000 rounds the req_cost will be set to the min(UCB) from the previous round

beta <- 3   # parameter in slack term
n_warmup <- 10    # warmup rounds, keep requesting label

#-------------------- 
# Load data
#-------------------- 
source('1.data.skin.R')

#-------------------- 
# Input to algorithm
# X, y, all_h, all_thre
#-------------------- 
nT <- nrow(X)
nT=100000
r_per_h <- length(all_thre)
nh <- nrow(all_h)

for (rep in c(1:10)){

    set.seed(rep); shuffle <- sample(nrow(X),nT,replace = FALSE)

    cum_obs <- matrix(0,nrow = nh, ncol = r_per_h) # cumulative obs for (h,r)
    cum_loss <- matrix(0,nrow = nh, ncol = r_per_h) # cumulative incurred loss for (h,r)

    cum_loss_alg_0= 0 # cumulative 0/1 error of the pseudo label prediction
    cum_reg <- 0 # cumulative regret
    cum_label <- 0 # cumulative number of label requests
    cum_random = 0; ## number of times that all experts are non-requesting
    # warmup
    for (i in seq_len(n_warmup)){
        x_t <- X[shuffle[i],]
        y_t <- y[shuffle[i]]
        pred_t <- all_h %*% t(x_t) # prediction
        pred_loss_t <- loss_func(pred_t, y_t) # prediction loss

        curr_ret <- cond_loss_allpairs(all_thre, pred_t, pred_loss_t)
        cum_loss <- cum_loss + curr_ret[1:nh,]
        cum_obs <- cum_obs + curr_ret[(1:nh)+nh,]

    }

    cum_reg <- req_cost * n_warmup
    cum_label <-  n_warmup

    # start active learning
    RET <- matrix(c(0),ncol = 6) # book keeping
    for (i in c((1+n_warmup) : nT)){
        x_t <- X[shuffle[i],]
        y_t <- y[shuffle[i]]

        pred_t <- all_h %*% t(x_t)
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
        cum_loss_tmp <- cum_loss + curr_ret[1:nh,]
        cum_obs_tmp <- cum_obs + curr_ret[(1:nh)+nh,]
        UCB <- (cum_loss_tmp/cum_obs_tmp + sqrt(2*beta*log(i)/cum_obs_tmp))[non_req_t]
#       UCB <- (cum_loss_tmp/cum_obs_tmp - sqrt(2*beta*log(i)/(0.001+cum_obs_tmp)))[non_req_t]
        # choose expert for this round
        h_idx <- (matrix(rep(seq_along(pred_t),r_per_h),ncol = r_per_h))[non_req_t]
        r_idx <- (matrix(rep(seq_along(all_thre),each = nh),ncol = r_per_h))[non_req_t]
        It <- which.min(UCB)
        h_It <- h_idx[It] 
        r_It <- r_idx[It]
        if (sum(non_req_t)==0){cum_random = cum_random+1}
        if (any(non_req_t)){ # some pair do not request
            not_request <- ifelse(UCB[It] < req_cost,TRUE,FALSE)
        } else {# all pairs request
            not_request <- FALSE
        }

        if(not_request){ # not request
            cum_reg <- cum_reg + pred_loss_t[h_It]
##            cum_loss[h_It,r_It] <- cum_loss[h_It,r_It] + pred_loss_pse_t[h_It] # update the chosen expert
##            cum_obs[h_It,r_It] <- cum_obs[h_It,r_It] + 1 # update the chosen expert
## Update all the non_requesting experts
            cum_loss[non_req_t] = cum_loss_tmp[non_req_t]
            cum_obs[non_req_t] = cum_obs_tmp[non_req_t]
        } else{ # request 
            cum_reg <- cum_reg + req_cost
            cum_label <- cum_label + 1
            true_curr_ret <- cond_loss_allpairs(all_thre, pred_t, pred_loss_t)
            cum_loss <- cum_loss  + true_curr_ret[1:nh,]
            cum_obs <- cum_obs + true_curr_ret[(1:nh)+nh,]
        }

        cum_loss_alg_0<- cum_loss_alg_0+ loss_func(pse_y_t,y_t,'misclass')
        ## if (i %% 100 == 0) browser()
        if (i %% 1000 ==0){
            req_cost = min(UCB)
            cat('i:',i, 
                ', num of labels:',cum_label,
                ', expert:',h_It,
                ', min UCB:', min(UCB),
                ', loss/i:', cum_loss_alg_0/i,
                '\n')
            
            if (length(h_It)==0){h_It <- 0; }
            RET <- rbind(RET,c(i,cum_label, h_It,min(UCB),cum_reg,cum_loss_alg_0))
        }
    }
    filename <- paste0('ucbhp_rep',rep,'.csv')
    write.table(RET,filename, sep = ',',col = FALSE,row.names = FALSE)
    
}


