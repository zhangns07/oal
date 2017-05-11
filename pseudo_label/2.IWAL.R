source('0.init.R')
source('1.data.skin.R')

x_range <- max(max(X), -min(X))
M <- log(1+exp(x_range))
req_cost <- 0.5 # request cost, c in the paper

#--------------------
# input
# X,y
# all_h, max_dis
#--------------------
nT <- nrow(X)
nh <- nrow(all_h)

for (rep in  c(1:10)){
    set.seed(rep); shuffle <- sample(seq_len(nT),replace = FALSE)
    cum_loss <- rep(0,nh)
    Ht <- rep(TRUE,nh)
    cum_label <- 0
    cum_reg <- 0

    RET <- matrix(c(0),ncol = 4) # book keeping
    for (i in seq_len(nT)){
        x_t <- X[shuffle[i],]
        y_t <- y[shuffle[i]]

        pred_t <- all_h %*% x_t
        loss0 <- loss_func(pred_t,-1, 'logistic') / M
        loss1 <- loss_func(pred_t,1,'logistic') / M

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

        if (i %% 50 ==0){
        #    cat('num of rounds:',i,
        #        ', num of labels:',cum_label,'\n')

            RET <- rbind(RET,c(i,cum_label,It, cum_reg))
        }
    }

    filename <- paste0('iwal_rep',rep,'_cost',req_cost,'.csv')
    write.table(RET,filename, sep = ',',col = FALSE,row.names = FALSE)
    
}
