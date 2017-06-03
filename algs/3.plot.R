library(data.table)
library(ggplot2)
cost <- 0.5
cost <- 0.6

rep <- 1
filename <- paste0('results/bestexp_rep',rep,'_cost',cost,'.csv')
ret_bestexp <- read.table(filename, sep = ',')
for (rep in c(2:10)){
    filename <- paste0('results/bestexp_rep',rep,'_cost',cost,'.csv')
    ret_bestexp <- rbind(ret_bestexp, read.table(filename, sep = ','))
}


rep <- 1
filename <- paste0('results/ucbhp_rep',rep,'_cost',cost,'.csv')
ret_ucb <- read.table(filename, sep = ',')
for (rep in c(2:10)){
    filename <- paste0('results/ucbhp_rep',rep,'_cost',cost,'.csv')
    ret_ucb <- rbind(ret_ucb, read.table(filename, sep = ','))
}

ret_ucb <- data.table(ret_ucb, best_exp = ret_bestexp[,2])
ret_ucb <- ret_ucb[,list(req_m = mean(V2),req_sd = sd(V2),
                         loss_m = mean(V5), loss_sd = sd(V5),
                         reg_m = mean((V5 - best_exp )/V1),
                         reg_sd = sd((V5 - best_exp )/V1)),by=list(rounds = V1)]

rep <- 1
filename <- paste0('results/iwal_rep',rep,'_cost',cost,'.csv')
ret_iwal <- read.table(filename, sep = ',')

for (rep in c(2:10)){
    filename <- paste0('results/iwal_rep',rep,'_cost',cost,'.csv')
    ret_iwal <- rbind(ret_iwal , read.table(filename, sep = ','))
}

ret_iwal <- data.table(ret_iwal, best_exp = ret_bestexp[,2])
ret_iwal <- ret_iwal[,list(req_m = mean(V2),req_sd = sd(V2),
                         loss_m = mean(V4), loss_sd = sd(V4),
                         reg_m = mean((V4 - best_exp)/V1),
                         reg_sd = sd((V4 - best_exp)/V1)) ,by=list(rounds = V1)]

toplot <- ret_ucb[,list(rounds, y = req_m, ysd = req_sd, method = 'ucb', type = 'label request')]
toplot <- rbind(toplot, ret_iwal[,list(rounds, y = req_m, ysd = req_sd, method = 'iwal', type = 'label request')])
#toplot <- rbind(toplot, ret_ucb[,list(rounds, y = loss_m, ysd = loss_sd, method = 'ucb', type = 'cumulative loss')])
#toplot <- rbind(toplot, ret_iwal[,list(rounds, y = loss_m, ysd = loss_sd, method = 'iwal', type = 'cumulative loss')])
#toplot <- rbind(toplot, ret_ucb[,list(rounds, y = reg_m, ysd = reg_sd, method = 'ucb', type = 'regret per round')])
#toplot <- rbind(toplot, ret_iwal[,list(rounds, y = reg_m, ysd = reg_sd, method = 'iwal', type = 'regret per round')])
toplot <- rbind(toplot, ret_ucb[,list(rounds, y = reg_m, ysd = 0, method = 'ucb', type = 'pseudo-regret per round')])
toplot <- rbind(toplot, ret_iwal[,list(rounds, y = reg_m, ysd = 0, method = 'iwal', type = 'pseudo-regret per round')])


idx <- seq_len(round(nrow(toplot)/50)) * 50
toplot <- toplot[idx]

g <- ggplot(toplot, aes(x = rounds)) +  geom_line(aes(y = y, color = factor(method))) + 
    geom_ribbon(aes(ymin = y-ysd, ymax = y+ysd, fill= factor(method)), alpha = 0.3,)+
    facet_wrap( ~ type, scales = 'free_y',ncol = 1)
g <- g + theme_bw()+theme(legend.title=element_blank()) 
g <- g + xlab("Number of Rounds") + ylab("")  + labs(title = paste0("c = ",cost))

pdf(paste0('ucb_iwal_cost_',cost,'.pdf'))
plot(g)
dev.off()

