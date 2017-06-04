library(stats)
source('0.init.R')
source('1.data.skin.R')

req_cost <- 0.55 # request cost, c in the paper
gamma <- 0.1
eta <- 0.1

#--------------------
# input
# X,y
# all_h, max_dis
#--------------------
nT <- nrow(X)
nT = 100000
r_per_h <- length(all_thre)
nh <- nrow(all_h)


for (rep in c(1:10)){

    set.seed(rep); shuffle <- sample(nrow(X),nT,replace = FALSE)
    cum_label <- 0 # cumulative number of label requests
    cum_reg <- 0
    cum_loss_alg_0 <- 0
    u <- 1/(r_per_h * nh) #uniform distribution over experts
    qt <- rep(u, nh * r_per_h)

    RET <- matrix(c(0),ncol = 5) # book keeping

    for(i in seq_len(nT)){
        x_t <- X[shuffle[i],]
        y_t <- y[shuffle[i]]
        pred_t <- all_h %*% t(x_t) # prediction
        pred_loss_t <- loss_func(pred_t, y_t) # prediction loss

        # sample an expert
        pt  <- (1-gamma) * qt + gamma * u
        It <- which.max(rmultinom(1, size = 1, prob = pt))
        h_It <- It %% nh
        r_It <- ceiling(It / nh)
        if (h_It ==0){h_It <- nh}

        # update regret
        if (all_thre[r_It] - abs(pred_t[h_It]) >0){ # request 
            cum_label <- cum_label +1
            cum_reg <- cum_reg + req_cost
            curr_ret <- loss_allpairs(all_thre, pred_t, pred_loss_t, req_cost, request = TRUE)
        } else{ # not request
            cum_reg <- cum_reg + pred_loss_t[h_It]
            curr_ret <- loss_allpairs(all_thre, pred_t, pred_loss_t, req_cost, request = FALSE)
        }

        # update weights
        req_t <- apply(array(all_thre),1, function(x){(x - abs(pred_t))>0}) # requester
        req_t_wts <- sum(pt[req_t]) #requesters weights
        Pt <- ifelse(req_t, 1, req_t_wts) 

        lt_hat <- curr_ret[1:nh,] / Pt
        qt <- qt * exp(-eta * lt_hat )
        qt <- qt / sum(qt)

        cum_loss_alg_0<- cum_loss_alg_0 + loss_func(pred_t[h_It],y_t,'misclass')
        if (i %% 1000 ==0){
            cat('i:',i, 
                ', num of labels:',cum_label,
                ', expert:',h_It,
                ', loss/i:', cum_loss_alg_0/i,
                '\n')

            if (length(h_It)==0){h_It <- 0; }
            RET <- rbind(RET,c(i,cum_label, h_It,cum_reg,cum_loss_alg_0))
        }
    }

    filename <- paste0('exp3al_rep',rep,'.csv')
    write.table(RET,filename, sep = ',',col = FALSE,row.names = FALSE)

}
