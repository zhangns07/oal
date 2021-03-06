library(plyr)
# data
skin <- read.table('skin.txt')
y <- skin[,4]
y <- -3 + 2*y  # depends on which direction of hyperplane gives smaller error
X_raw <- skin[,-4]
# X <- scale(X_raw) 
maxX <- max(X_raw)
X <- X_raw / maxX
X <- as.matrix(X)

# experts
l <- 20
scales <- 2^c(0:4)
polar_base <- seq(0,pi,length.out = l)
polar <- expand.grid(polar_base, polar_base)


all_h_org  <- apply(polar,1,function(x){ # unscaled h
                      c(sin(x[1])*cos(x[2]), sin(x[1])*sin(x[2]), cos(x[1])) })
all_h <- matrix(apply(array(scales),1,function(x){all_h_org * x}),ncol = 3, byrow = TRUE)

all_thre <- unique(c(seq(0,0.1,0.01),seq(0.1, 0.3,0.1))) * max(scales)

