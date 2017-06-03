source('0.init.R')

#req_cost <- 0.5 # request cost, c in the paper
req_cost <- 0.6 # request cost, c in the paper
n_warmup <- 10    # warmup rounds, keep requesting label

#-------------------- 
# Load data
#-------------------- 
source('1.data.skin.R')

#-------------------- 
# Input to algorithm
# iven X, y, all_h, all_thre
#-------------------- 
nT <- nrow(X)
r_per_h <- length(all_thre)
nh <- nrow(all_h)


# best expert inhindside
# have to make sure that best inhindside has \hat \mu < req_cost
for (rep in c(1:10)){
    set.seed(rep); shuffle <- sample(seq_len(nT),replace = FALSE)

    cum_loss_uncond <- matrix(0,nrow = nh, ncol = r_per_h)
    RET <- matrix(c(0),ncol = 2) # book keeping
    for (i in seq_len(nT)){
        x_t <- X[shuffle[i],]
        y_t <- y[shuffle[i]]
        pred_t <- all_h %*% x_t
        pred_loss_t <- loss_func(pred_t, y_t) # prediction loss

        curr_loss_uncond <- apply(array(all_thre),1, function(x){
                                      r <- x - abs(pred_t)
                                      ifelse(r>0, req_cost, pred_loss_t)})
        cum_loss_uncond <- cum_loss_uncond + curr_loss_uncond

        if (i %% 50==0){
            RET <- rbind(RET,c(i,min(cum_loss_uncond)))
            #cat(which.min(cum_loss_uncond),'\n')
        }
    }

    filename <- paste0('bestexp_rep',rep,'_cost',req_cost,'.csv')
    write.table(RET,filename, sep = ',',col = FALSE,row.names = FALSE)
}



