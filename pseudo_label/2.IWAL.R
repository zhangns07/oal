source('0.init.R')
source('1.data.skin.R')

x_range <- max(max(X), -min(X))
M <- log(1+exp(sqrt(3)*x_range)) # upper bound of logistic loss
req_cost <- 0.55 # request cost, c in the paper

#--------------------
# input
# X,y
# all_h, max_dis
#--------------------
nT <- nrow(X)
nT = 100000
nh <- nrow(all_h)

for (rep in  c(3)){
    set.seed(rep); shuffle <- sample(nrow(X),nT,replace = FALSE)
    cum_loss <- rep(0,nh)
    Ht <- rep(TRUE,nh)
    cum_label <- 0
    cum_reg <- 0
    cum_loss_alg_0= 0 ## cumulative 0/1 error of the best expert per round

    RET_iwal <- matrix(c(0),ncol = 6) # book keeping
    for (i in seq_len(nT)){
        x_t <- X[shuffle[i],]
        y_t <- y[shuffle[i]]

        pred_t <- all_h %*% t(x_t)
        loss0 <- (loss_func(pred_t,-1, 'logistic'))[Ht] / M
        loss1 <- (loss_func(pred_t,1,'logistic'))[Ht] / M

        p_t <- max_dis(loss0,loss1)
        Q_t <- as.numeric(runif(1) < p_t)

        if (Q_t > 0){
            cum_reg <- cum_reg + req_cost
            cum_label <-cum_label + 1 

            cum_loss <- cum_loss + loss_func(pred_t,y_t,'logistic')/(M*p_t) # importance weighted cum_loss
            min_err <- min(cum_loss[Ht])
            slack_t <- sqrt(i*log(i)) # a more aggresive slack term than IWAL paper
            Ht <- (Ht  & cum_loss <= min_err + slack_t)
        } else{
            It <- (seq_len(nh)[Ht])[which.min(cum_loss[Ht])] # best expert for this round
            cum_reg <- cum_reg + loss_func(pred_t[It], y_t)
        }

        It <- (seq_len(nh)[Ht])[which.min(cum_loss[Ht])] # best expert for this round
        cum_loss_alg_0<- cum_loss_alg_0+ loss_func(pred_t[It],y_t,'misclass')

        if (i %% 1000 ==0){
            cat('num of rounds:',i,
                ', num of labels:',cum_label,
                ', expert:',It,
                ', loss/i:', cum_loss_alg_0/i,'\n')

            RET_iwal <- rbind(RET_iwal,c(i,cum_label,It, p_t, cum_reg,cum_loss_alg_0))
        }
    }

    filename <- paste0('iwal_rep',rep,'.csv')
    write.table(RET_iwal,filename, sep = ',',col = FALSE,row.names = FALSE)
    
}
