# 1. find the best model with logistic loss
# and look at its misclassification error

# 2. find the best model with misclassification error

# 3. run logistic regression to find best in class

# 4. why IWAL does not converge to 1 but to 2 ???

source('0.init.R')
source('1.data.skin.R')

rep = 1
nT = 10000
nh = nrow(all_h)
set.seed(rep); shuffle <- sample(nrow(X),nrow(X),replace = FALSE)
Xtrain <- as.matrix(X[shuffle[1:nT],])
ytrain <- y[shuffle[1:nT]]


batch_size <- 100
train_cum_loss <- rep(0,nh)
for (i in seq_len(nT / batch_size)){
    Xbatch <- Xtrain[seq(((i-1)*batch_size+1) , i*batch_size),] 
    ybatch <- ytrain[seq(((i-1)*batch_size+1) , i*batch_size)]

    train_cum_loss <- train_cum_loss + 
#    apply(all_h %*% t(Xbatch) * rep(ybatch, each = nh), 1, 
#          function(x){sum(x<0)})
    apply(all_h %*% t(Xbatch) * rep(ybatch, each = nh), 1, 
          function(x){sum(log(1+exp(-x)))})
}


train_cum_loss <- rep(0,nh)
for (i in seq_len(nT)){
    x_t <- X[shuffle[i],]
    y_t <- y[shuffle[i]]

    pred_t <- all_h %*% t(x_t)
    loss <- (loss_func(pred_t,y_t, 'logistic'))
    train_cum_loss <- train_cum_loss + loss
}

It <- which.min(train_cum_loss)
h_It <- It %% 600
r_It <- ceiling(It/600)
sum((Xtrain %*% all_h[h_It,]) * ytrain < 0 ) / nT

# run logistic regression and find cloest h to it
loglm <-  glm(as.numeric(y>0) ~ 0 + as.matrix(X), family = 'binomial')
loglm <-  glm(as.numeric(y>0) ~  as.matrix(X), family = 'binomial')
best_h <- loglm$coefficients
dist_to_besth <- apply(all_h - rep(best_h,each = nh), 1, function(x){sum(x^2)})
sum((predict(loglm,type = 'response') - 0.5) * y  <0) / nT

sum((Xtrain %*% best_h[-1] + best_h[1] ) * ytrain < 0 ) / nT

sum((Xtrain %*% best_h) * ytrain < 0 ) / nT
sum((Xtrain %*% all_h[318,]) * ytrain < 0 ) / nT
sum((Xtrain %*% all_h[307,]) * ytrain < 0 ) / nT
sum((Xtrain %*% all_h[428,]) * ytrain < 0 ) / nT

test_cum_loss <- rep(0,nh)
for (i in seq_len(ntest / batch_size)){
    Xbatch <- Xtest[seq(((i-1)*batch_size+1) , i*batch_size),] 
    ybatch <- ytest[seq(((i-1)*batch_size+1) , i*batch_size)]

    test_cum_loss <- test_cum_loss + 
    apply(all_h %*% t(Xbatch) * rep(ybatch, each = nh), 1, 
          function(x){sum(x<0)})
}

print(min(train_cum_loss/nT)); 
print(min(test_cum_loss/ntest)); 
which.min(train_cum_loss/nT);which.min(test_cum_loss/nT);
REP = rep
print(RET[rep== REP & rounds == 100000, list(trainerr,testerr)])
cat(rep)
}




realRET <- matrix(c(0.06552 ,0.0667 ,0.0656 ,0.0649
,0.065 ,0.065 ,0.06403 ,0.0639
,0.06537 ,0.0656 ,0.06446 ,0.0697
,0.06412 ,0.0617 ,0.06443 ,0.0634 ,0.06544
,0.0616 ,0.06475 ,0.0653), ncol = 2, byrow = TRUE)

apply(realRET,2,mean)
RET[rounds == 100000,list(mean(trainerr),mean(testerr))]

cbind(RET[rounds == 100000,list(trainerr,testerr)], realRET)



