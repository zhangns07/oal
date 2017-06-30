# compare train error vs. test error per round
library(data.table)
library(ggplot2)
DATASET <- c('skin','shuttle','HTRU_2')

# 1. read data
for(dataset in DATASET){

    resultdir <- paste0('results/',dataset)
    train_err <- data.table()
    test_err <- data.table()


    for (rep in c(1:10)){
        for(req_cost in c(0.3,0.5)){

            filename <- paste0(resultdir,'/ucb_req',req_cost,'_rep',rep,'.csv')
            tmpdata <- fread(filename)
            train_err <- rbind(train_err, cbind(tmpdata, cost = req_cost, 
                                                alg = 'ucb', rep = rep))

            filename <- paste0(resultdir,'/otb_ucb_req',req_cost,'_rep',rep,'.csv')
            tmpdata <- fread(filename)
            test_err <- rbind(test_err, cbind(tmpdata, cost = req_cost, 
                                              alg = 'ucb', rep = rep))

            filename <- paste0(resultdir,'/ucblcb_req',req_cost,'_rep',rep,'.csv')
            tmpdata <- fread(filename)
            train_err <- rbind(train_err, cbind(tmpdata, cost = req_cost, 
                                                alg = 'ucblcb', rep = rep))

            filename <- paste0(resultdir,'/otb_ucblcb_req',req_cost,'_rep',rep,'.csv')
            tmpdata <- fread(filename)
            test_err <- rbind(test_err, cbind(tmpdata, cost = req_cost, 
                                              alg = 'ucblcb', rep = rep))

            test_err <- rbind(test_err, cbind(tmpdata, cost = req_cost, 
                                              alg = 'ucbfl', rep = rep))

            filename <- paste0('results/',dataset,'_ucbfl_req',req_cost,'_rep',rep,'.csv')
            tmpdata <- fread(filename)
            train_err <- rbind(train_err, cbind(tmpdata, cost = req_cost, 
                                                alg = 'ucbfl', rep = rep))


            filename <- paste0(dataset,'_iwal_req',req_cost,'_rep',rep,'.csv')
            tmpdata <- fread(filename)
            train_err <- rbindlist(list(train_err, 
                                        cbind(tmpdata[,c(1:3),with=F], dummy = 0, tmpdata[,c(4:7),with=F], cost = req_cost, 
                                              alg = 'iwal', rep = rep)))

            filename <- paste0(dataset,'_otb_iwal_req',req_cost,'_rep',rep,'.csv')
            tmpdata <- fread(filename)
            test_err <- rbind(test_err, cbind(tmpdata, cost = req_cost, 
                                              alg = 'iwal', rep = rep))


        }
    }



    colnames(train_err) <- c('round','num_label_per_round','expert_h','expert_r','minUCB',
                             'loss_per_round','misclass_per_round', 'logistic_per_round', 
                             'cost','alg','rep')
    colnames(test_err) <- c('round','err','cost','alg','rep')


    train_err <- train_err[round > 0]
    test_err <- test_err[round > 0]

    train_err[,num_label_per_round:= num_label_per_round - num_label_per_round %% 100]

    # 2. plot by rounds
    #  train err and test err
    toplot_err <- data.table()
    COLS <- c('num_label_per_round','loss_per_round',
              'misclass_per_round','logistic_per_round')
    for (col in COLS){
        toplot_err <- rbind(toplot_err,
                            train_err[,list(mean = mean(get(col)/round),
                                            sd = sd(get(col)/round),
                                            col = col ),by=list( alg,cost, round)])
    }


    toplot_err <- rbind(toplot_err, test_err[,list(mean = mean(err), sd = sd(err), col = 'otb_misclass_test'),by=list( alg,cost, round)])



    for (req_cost in c(0.3,0.5)){
        g <- ggplot(toplot_err[cost == req_cost], aes(x = round, color = alg)) + geom_line(aes(y = mean))
        g <- g + geom_ribbon(aes(ymin = mean-sd, ymax = mean+sd, fill= alg), alpha = 0.2)
        g <- g + facet_wrap( ~ col , scales = 'free_y', ncol = 1)
        g <- g + theme_bw()+theme(legend.title=element_blank()) 
        g <- g + xlab("Number of Rounds") + ylab("")  + labs(title = paste0("dataset: ", dataset, ", c = ",req_cost))

        pdf(paste0('plots/',dataset,'_ucb_cost_',req_cost,'.pdf'))
        plot(g)
        dev.off()
    }

    # 3. plot by num labels
    #  train err and test err
    toplot_err <- data.table()
    COLS <- c('loss_per_round',
              'misclass_per_round','logistic_per_round')

    for (col in COLS){
        toplot_err <- rbind(toplot_err,
                            train_err[,list(mean = mean(get(col)/round),
                                            col = col ),by=list( alg,cost, labels = num_label_per_round)])

    }

    toplot_err <- rbind(toplot_err, 
                        cbind(test_err, train_err[,list(num_label_per_round)])[,list(mean = mean(err), col = 'otb_misclass_test'),by=list( alg, cost, labels = num_label_per_round)])



    for (req_cost in c(0.3,0.5)){
        g <- ggplot(toplot_err[cost == req_cost], aes(x = labels , color = alg)) + geom_line(aes(y = mean))
        #g <- g + geom_ribbon(aes(ymin = mean-sd, ymax = mean+sd, fill= alg), alpha = 0.2)
        g <- g + facet_wrap( ~ col , scales = 'free_y', ncol = 1)
        g <- g + theme_bw()+theme(legend.title=element_blank()) 
        g <- g + xlab("Number of Rounds") + ylab("")  + labs(title = paste0("dataset: ", dataset, ", c = ",req_cost))

        pdf(paste0('plots/',dataset,'bylabels_ucb_cost_',req_cost,'.pdf'))
        plot(g)
        dev.off()
    }


}
