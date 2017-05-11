#--------------------
# synthetic data
#--------------------
gen_x <- function # generate feature in R^d.
(xd,  # feature dimension
 xrange = 1 # each ordinate of x takes value from [-x_range, x_range]
 ){
    return (1 - 2*runif(xd))
}

gen_h <- function # generate base hypotheses h.
# Return a h_per_d^xd by xd matrix
(h_per_d, xd, max_norm)
{
    hrange <- max_norm/sqrt(xd)
    hbasis <- seq(-hrange,hrange,length.out = h_per_d)

    hraw <- expand.grid(replicate(xd, hbasis, simplify=FALSE))
    return (as.matrix(hraw))
}

#--------------------
# algorithm related
#--------------------
loss_func <- function
(h, # a vector of predictions
 y, # label
 loss = c('logistic','misclass')[1]
 ){
    if (loss == 'logistic'){
        return (log(1+exp(-y*h)))
    } else if (loss == 'misclass'){
        return (as.numeric(y*h<0))
    } else { stop }

}

#---------
# For UCB-HP
cond_loss_allpairs <- function
(all_thre, # a vector of thresholds, length n
 pred, # a vector of expert predictions, length m
 pred_loss # a vector of expert predictions loss, length m
 ){
    # return a 2m x n matrix
    # first m x n is loss, second m x n is obs

    # if request, loss = 0, obs = 0
    # if not request, loss = prediciton loss, obs = 1
    curr_loss <- apply(array(all_thre),1, function(x){
                           r <- x - abs(pred)
                           c(ifelse(r>0, 0, pred_loss), # loss
                             as.numeric(r<=0)) }) # obs
    return (curr_loss)
}

gen_pseudo_label <- function
(h, # prediction
 mu # average loss so far
 ){
    return(ifelse(sum((1-mu)*h)>0,1,-1))
}

#---------
# For IWAL
max_dis <- function # return maximum disagreement, 
# or equivalently, requesting probability in IWAL
(loss0, # a vector of predictions error, for y = -1
 loss1# a vector of predictions error, for y = 1
 ){
    if(length(loss0) != length(loss1)){stop}
    nh <- length(loss0)

    glb_max <- 0
    for (i in seq_len(nh-1)){
        j <- c((i+1):nh)
        loc_max <- max(pmax(loss0[i] - loss0[j], loss1[i]-loss1[j]),
                       pmax(loss0[j] - loss0[i], loss1[j]-loss1[i]))
        glb_max <- max(loc_max, glb_max)

    }
    return (glb_max)
}


