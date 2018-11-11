library(data.table)
library(latex2exp)
library(ggplot2)


load('rbret_noscale.RData'); RET_rb  <- RET
load('oneret_noscale.RData'); RET_one <-  RET
load('onepassiveret_noscale.RData'); RET_passive <-  RET
RET <- cbind(rbind(RET_rb,RET_one,RET_passive),slack='azuma', const=1)

load('rbret_bernstein.RData'); RET_rb_ber  <- RET; RET_rb_ber[,loss_softlogistic:=NULL]
load('rbret.RData'); RET_rb  <- RET
load('oneret.RData'); RET_one  <- RET
load('onepassiveret.RData'); RET_passive <-  RET; RET_passive[,loss_softlogistic:=NULL]
RET <- cbind(rbind(RET_rb,RET_one,RET_passive),slack='azuma', const=1)
RET <- RET[data!='w8a']

load('rbret_noscale.RData'); RET_rb  <- RET
load('rbret_bernstein_noscale.RData'); RET_rb_ber  <- RET

RET <- cbind(rbind(RET_rb),slack='azuma', const=1)
RET <- rbind(RET,cbind(RET_rb_ber,slack='bernstein'))
RET <- RET[data!='w8a']

#----------------------------------------
# optrb-eiwal , rb-iwal , noscale
# RESULT: NO DIFFERENCE
load('optrbret_bernstein_noscale.RData'); RET_rb_ber  <- RET; RET_rb_ber[,loss_softlogistic:=NULL]; RET_rb_ber[,const:=NULL]; 
load('rbret_noscale.RData'); RET_rb  <- RET; RET_rb <- RET_rb[alg=='rbiwal',]
RET <- rbind(RET_rb_ber,RET_rb,fill=TRUE)
RET <- RET[!data %in% c('phishing','egg','w8a')]

# by round
bylabel <- TRUE; loss <- 'loss_misclass'; logerr <- FALSE; nonaive <- TRUE; regiontype <- 'tree'
if(bylabel <- TRUE){ RET[,rounds:=labels] } else {RET[,rounds := accepts]}
toplot <- RET[rounds > 0 & rounds %%100 == 0 & rounds < 10^(3.5), list(err = mean(get(loss)),sd=sd(get(loss)),.N),by=list(rounds,data,alg,region,nr)]
toplot <- toplot[N==20 ]

toplot$alg <- as.factor(toplot$alg); levels(toplot$alg) <- apply(array(levels(toplot$alg)),1,function(x){switch(x,'oneiwal'='IWAL','optrb'='OPT-RB-IWAL','rbiwal'='RB-IWAL','onepassive'='PASSIVE')})
toplot$data <- factor(toplot$data, levels=c("magic04" ,"nomao" ,"shuttle" ,"a9a" ,"ijcnn1" ,"codrna" ,"skin" ,"covtype"))

toplot <- toplot[nr %in% c(20,40)]; toplot$nr <- ifelse(toplot$nr == 20, 10, 20) # because indeed the tree only has 10 and 20 regions roughly

g <- ggplot(toplot[region==regiontype & nr == 20],aes(x=log(rounds,10),y=err, color=as.factor(alg),linetype=as.factor(alg)))
g <- g + geom_line(size=0.7) 
g <- g + facet_wrap(~data,ncol=2,scales='free_y')
g <- g + theme_bw(base_size=15)+theme(legend.title=element_blank(), legend.position='top')
xlab_tex <- ifelse(bylabel,"$\\log_{10}$(Number of Labels)","$\\log_{10}$(Number of Queries to IWAL)")
lossname <- switch(loss,'loss_logistic'='Logistic Loss', 'loss_misclass'='Misclassification Loss','loss_softlogistic'='Soft Logistic Loss')
ylab_tex <- ifelse(logerr, paste0("$\\log_{10}$(",lossname,")"), lossname)
g <- g + xlab(TeX(xlab_tex)) + ylab(TeX(ylab_tex))
g <- g + scale_color_manual(values=c("red", "blue" ,"darkgreen","orange","black"))
print(g)

filename <- paste0(loss,"_vs_",ifelse(bylabel,'labels','accepts'),'_opteiwal_vs_rbiwal.pdf')
pdf(filename, width=7,height=10)
print(g)
dev.off()


#----------------------------------------
# E-iwal, iwal
# RESULT: NO DIFFERENCE
load('oneiwal_bernstein_noscale.RData');  RET[,alg:='eiwal']; RET_ber <- RET
load('onepassiveret_noscale.RData'); RET_pas <- RET
load('oneret_noscale.RData'); RET_iwal <- RET
RET <- rbind(RET_ber, RET_pas,RET_iwal,fill=TRUE)
RET <- RET[!data %in% c('phishing','egg','w8a')]

# by round
bylabel <- TRUE; loss <- 'loss_misclass';  regiontype <- 'tree'
if(bylabel <- TRUE){ RET[,rounds:=labels] } else {RET[,rounds := accepts]}
toplot <- RET[rounds > 0 & rounds %%100 == 0 & rounds < 10^(3.5), list(err = mean(get(loss)),sd=sd(get(loss)),.N),by=list(rounds,data,alg)]
toplot <- toplot[N==20 ]

toplot$alg <- as.factor(toplot$alg); levels(toplot$alg) <- apply(array(levels(toplot$alg)),1,function(x){switch(x,'oneiwal'='IWAL','eiwal'='E-IWAL','onepassive'='PASSIVE')})
toplot$data <- factor(toplot$data, levels=c("magic04" ,"nomao" ,"shuttle" ,"a9a" ,"ijcnn1" ,"codrna" ,"skin" ,"covtype"))
g <- ggplot(toplot,aes(x=log(rounds,10),y=err, color=as.factor(alg),linetype=as.factor(alg)))
g <- g + geom_line(size=0.7) 
g <- g + facet_wrap(~data,ncol=2,scales='free_y')
g <- g + theme_bw(base_size=15)+theme(legend.title=element_blank(), legend.position='top')
xlab_tex <- ifelse(bylabel,"$\\log_{10}$(Number of Labels)","$\\log_{10}$(Number of Queries to IWAL)")
lossname <- switch(loss,'loss_logistic'='Logistic Loss', 'loss_misclass'='Misclassification Loss','loss_softlogistic'='Soft Logistic Loss')
ylab_tex <- ifelse(logerr, paste0("$\\log_{10}$(",lossname,")"), lossname)
g <- g + xlab(TeX(xlab_tex)) + ylab(TeX(ylab_tex))
g <- g + scale_color_manual(values=c("red", "blue" ,"darkgreen","orange","black"))
print(g)
filename <- paste0(loss,"_vs_",ifelse(bylabel,'labels','accepts'),'_one.pdf')
pdf(filename, width=7,height=10)
print(g)
dev.off()

#----------------------------------------
# optrb, rb, noscale
load('rbret_noscale.RData'); 
RET <- RET[!data %in% c('phishing','egg','w8a')]
RET <- RET[alg!='naiveiwal']

# by round
bylabel <- TRUE; loss <- 'loss_misclass'; regiontype <- 'tree'; nregion <- 20
if(bylabel <- TRUE){ RET[,rounds:=labels] } else {RET[,rounds := accepts]}
toplot <- RET[rounds > 0 & rounds %%100 == 0 & rounds < 10^(3.5), list(err = mean(get(loss)),sd=sd(get(loss)),.N),by=list(rounds,data,alg,region,nr)]
toplot <- toplot[N==20 ]

toplot$alg <- as.factor(toplot$alg); levels(toplot$alg) <- apply(array(levels(toplot$alg)),1,function(x){switch(x,'oneiwal'='IWAL','optrb'='ORIWAL','rbiwal'='RIWAL','onepassive'='PASSIVE')})

toplot <- toplot[nr %in% c(20,40)]; toplot$nr <- ifelse(toplot$nr == 20, 10, 20) # because indeed the tree only has 10 and 20 regions roughly
for(DATA in c("magic04" ,"nomao" ,"shuttle" ,"a9a" ,"ijcnn1" ,"codrna" ,"skin" ,"covtype")){
g <- ggplot(toplot[region==regiontype & nr == nregion & data == DATA],aes(x=log(rounds,10),y=err, color=as.factor(alg),linetype=as.factor(alg)))
g <- g + geom_line(size=1.3) 
xlab_tex <- ifelse(bylabel,"$\\log_{10}$(Number of Labels)","$\\log_{10}$(Number of Queries to IWAL)")
lossname <- switch(loss,'loss_logistic'='Logistic Loss', 'loss_misclass'='Misclassification Loss','loss_softlogistic'='Soft Logistic Loss')
ylab_tex <- lossname
g <- g + xlab(TeX(xlab_tex)) + ylab(TeX(ylab_tex))
g <- g + scale_color_manual(values=c("red", "blue" ,"darkgreen","orange","black"))
g <- g + ggtitle(paste0(DATA))
g <- g + theme_bw(base_size=23)+theme(legend.title=element_blank(), 
                                      legend.position='top', #legend.position=c(0.6,0.9),
                                      legend.key.width = unit(1.5,"cm"),
                                      legend.direction = "horizontal",
                                      plot.title = element_text(hjust = 0.5,family ='Courier'))
filename <- paste0(loss,"_vs_",ifelse(bylabel,'labels','accepts'),'_optvsrb_',DATA,'_',nregion,'.pdf')
pdf(filename,width=8,height=6)
print(g)
dev.off()
}


#----------------------------------------
# optrb, iwal, passive
load('rbret_noscale.RData'); RET <- RET[alg == 'optrb' & nr %in% c(20,40),]; RET$nr <- ifelse(RET$nr == 20, 10, 20) ;RET_rb  <- RET
load('oneret_noscale.RData'); RET_one <-  RET
load('onepassiveret_noscale.RData'); RET_passive <-  RET
RET <- cbind(rbind(RET_rb,RET_one,RET_passive))


# by round
bylabel <- TRUE; loss <- 'loss_misclass';  regiontype <- 'tree'; nregion <- 10
if(bylabel <- TRUE){ RET[,rounds:=labels] } else {RET[,rounds := accepts]}
toplot <- RET[rounds > 0 & rounds %%100 == 0 & rounds < 10^(3.5), list(err = mean(get(loss)),sd=sd(get(loss)),.N),by=list(rounds,data,alg, region, nr)]
toplot <- toplot[N==20 ]
toplot <- rbind(toplot[!alg %in% c('oneiwal','onepassive'),],
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=nregion,region=regiontype),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='onepassive'),nr=nregion,region=regiontype))
toplot$alg <- as.factor(toplot$alg); levels(toplot$alg) <- apply(array(levels(toplot$alg)),1,function(x){switch(x,'oneiwal'='IWAL','optrb'='ORIWAL','onepassive'='PASSIVE')})
toplot$alg <- factor(toplot$alg,levels=c('ORIWAL','IWAL','PASSIVE'))

for(DATA in c("magic04" ,"nomao" ,"shuttle" ,"a9a" ,"ijcnn1" ,"codrna" ,"skin" ,"covtype")){
g <- ggplot(toplot[region==regiontype & nr == nregion & data == DATA],aes(x=log(rounds,10),y=err, color=as.factor(alg),linetype=as.factor(alg)))
g <- g + geom_line(size=1.3) 
xlab_tex <- ifelse(bylabel,"$\\log_{10}$(Number of Labels)","$\\log_{10}$(Number of Queries to IWAL)")
lossname <- switch(loss,'loss_logistic'='Logistic Loss', 'loss_misclass'='Misclassification Loss','loss_softlogistic'='Soft Logistic Loss')
ylab_tex <- lossname
g <- g + xlab(TeX(xlab_tex)) + ylab(TeX(ylab_tex))
g <- g + scale_color_manual(values=c("red", "blue" ,"darkgreen","orange","black"))
g <- g + ggtitle(paste0(DATA))
g <- g + theme_bw(base_size=23)+theme(legend.title=element_blank(), 
                                      legend.key.width = unit(1.2,"cm"),
                                      legend.direction = "horizontal",
                                      legend.position="top",#legend.position=c(0.6,0.9),
                                      plot.title = element_text(hjust = 0.5,family ='Courier'))

filename <- paste0(loss,"_vs_",ifelse(bylabel,'labels','accepts'),'_onevsopt_',DATA,'_',nregion,'.pdf')
pdf(filename,width=8,height=6)
print(g)
dev.off()
}



#----------------------------------------
# by round
bylabel <- TRUE; loss <- 'loss_misclass'; logerr <- FALSE; nonaive <- TRUE; regiontype <- 'tree'
if(bylabel <- TRUE){ RET[,rounds:=labels] } else {RET[,rounds := accepts]}
toplot <- RET[rounds > 0 & rounds %%100 == 0 & rounds < 10^(3.5), 
              list(err = mean(get(loss)),sd=sd(get(loss)),.N),by=list(rounds,data,alg,region,nr,slack,const)]
toplot <- toplot[N==20 ]
if(1==0){
toplot <- rbind(toplot[!alg %in% c('oneiwal','onepassive'),],
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=2,region='circle'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=5,region='circle'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=10,region='circle'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=20,region='circle'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=40,region='circle'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=2,region='hyper'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=5,region='hyper'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=10,region='hyper'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=20,region='hyper'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=40,region='hyper'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=2,region='tree'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=5,region='tree'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=10,region='tree'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=20,region='tree'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=40,region='tree'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=2,region='cluster'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=5,region='cluster'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=10,region='cluster'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=20,region='cluster'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=40,region='cluster'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='onepassive'),nr=2,region='tree'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='onepassive'),nr=5,region='tree'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='onepassive'),nr=10,region='tree'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='onepassive'),nr=20,region='tree'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='onepassive'),nr=40,region='tree'))

}

if(nonaive){toplot <- subset(toplot, alg != 'naiveiwal')}
toplot$alg <- as.factor(toplot$alg); levels(toplot$alg) <- apply(array(levels(toplot$alg)),1,function(x){switch(x,'oneiwal'='IWAL','optrb'='OPT-RB-IWAL','rbiwal'='RB-IWAL','onepassive'='PASSIVE')})
toplot$data <- factor(toplot$data, levels=c("phishing" ,"egg" ,"magic04" ,"nomao" ,"shuttle" ,"a9a" ,"ijcnn1" ,"codrna" ,"skin" ,"covtype"))

toplot <- toplot[nr %in% c(20,40)]; toplot$nr <- ifelse(toplot$nr == 20, 10, 20) # because indeed the tree only has 10 and 20 regions roughly
if(nonaive){ toplot$alg <- factor(toplot$alg,levels=c('OPT-RB-IWAL','RB-IWAL','IWAL','PASSIVE'))
} else { toplot$alg <- factor(toplot$alg,levels=c('OPT-RB-IWAL','RB-IWAL','NAIVE-IWAL','IWAL','PASSIVE')) }
#toplot[,alg := paste0(alg,'_',slack,'_',const)]

#toplot <- toplot[alg %in% c('RB-IWAL_bernstein_1','RB-IWAL_azuma_1') & nr==10,]
#toplot <- toplot[alg %in% c('OPT-RB-IWAL_bernstein_1','OPT-RB-IWAL_azuma_1') & nr==10,]
#toplot <- toplot[alg %in% c('OPT-RB-IWAL_azuma_1','RB-IWAL_azuma_1') & nr==10,]

toplot <- toplot[ !data %in% c('phishing','egg') & nr==10 & alg!='RB-IWAL',]
#toplot <- toplot[ !data %in% c('phishing','egg') & nr==10 ,]
if(logerr == TRUE){ g <- ggplot(toplot[region==regiontype  ],
                                aes(x=log(rounds,10),y=log(err,10), color=as.factor(alg),linetype=as.factor(alg)))
} else{ g <- ggplot(toplot[region==regiontype ],aes(x=log(rounds,10),y=err, color=as.factor(alg),linetype=as.factor(alg)))}
g <- g + geom_line(size=0.7) 
#g <- g + geom_errorbar(aes(x=log(rounds,10),ymin=err-sd,ymax=err+sd))
#g <- g + facet_grid(data~nr,scales='free_y')
g <- g + facet_wrap(~data,ncol=2,scales='free_y')
g <- g + theme_bw(base_size=15)+theme(legend.title=element_blank(), legend.position='top')
xlab_tex <- ifelse(bylabel,"$\\log_{10}$(Number of Labels)","$\\log_{10}$(Number of Queries to IWAL)")
lossname <- switch(loss,'loss_logistic'='Logistic Loss', 'loss_misclass'='Misclassification Loss','loss_softlogistic'='Soft Logistic Loss')
ylab_tex <- ifelse(logerr, paste0("$\\log_{10}$(",lossname,")"), lossname)
g <- g + xlab(TeX(xlab_tex)) + ylab(TeX(ylab_tex))
g <- g + scale_color_manual(values=c("red", "blue" ,"darkgreen","orange","black"))
print(g)

filename <- paste0(loss,"_vs_",ifelse(bylabel,'labels','accepts'),'_',regiontype,'.pdf')
pdf(filename, width=7,height=10)
print(g)
dev.off()

if(1==0){
#filename <- paste0(loss,"_vs_",ifelse(bylabel,'labels','accepts'),'_',regiontype,'_nr10_nh500_azuma.pdf')
#filename <- paste0(loss,"_vs_",ifelse(bylabel,'labels','accepts'),'_',regiontype,'_nr10_nh3000_azuma.pdf')
#filename <- paste0(loss,"_vs_",ifelse(bylabel,'labels','accepts'),'_',regiontype,'_nr10_nh3000_azuma_onevsopt.pdf')
filename <- paste0(loss,"_vs_",ifelse(bylabel,'labels','accepts'),'_',regiontype,'_nr10_nh3000_azuma_onevsopt.pdf')
pdf(filename, width=8,height=8)
print(g)
dev.off()

}

#----------------------------------------

# label per accept
toplot <- RET[accepts > 0 & accepts %%100 == 0 & accepts < 10^4,list(labels = mean(labels),.N),by=list(accepts,data,alg,region,nr)]
toplot <- rbind(toplot[alg!='oneiwal'],
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=2,region='circle'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=5,region='circle'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=10,region='circle'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=20,region='circle'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=40,region='circle'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=2,region='hyper'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=5,region='hyper'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=10,region='hyper'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=20,region='hyper'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=40,region='hyper'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=2,region='tree'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=5,region='tree'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=10,region='tree'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=20,region='tree'),
                cbind( subset(toplot[,-c('region','nr'),with=F],alg=='oneiwal'),nr=40,region='tree'))


toplot <- toplot[alg %in% c('rbiwal','naiveiwal')]
toplot$alg <- as.factor(toplot$alg); levels(toplot$alg) <- c('NAIVE-IWAL','RB-IWAL')
toplot$alg <- factor(toplot$alg,levels(toplot$alg)[c(2,1)])
#levels(toplot$alg) <- c('NAIVE-IWAL','IWAL','OPT-RB-IWAL','RB-IWAL') #toplot$alg <- factor(toplot$alg,levels(toplot$alg)[c(3,4,1,2)])
toplot$data <- factor(toplot$data, levels=c("phishing" ,"egg" ,"magic04" ,"nomao" ,"shuttle" ,"a9a" ,"ijcnn1" ,"codrna" ,"skin" ,"covtype"))
toplot <- toplot[nr %in% c(20,40)]; toplot$nr <- ifelse(toplot$nr == 20, 10, 20) 
g <- ggplot(toplot[region=='tree' ],aes(x=log(accepts,10),y=labels/accepts, linetype=as.factor(alg)))+geom_line(size=0.7)
g <- g + facet_grid(data~nr,scales='free_y')
g <- g + theme_bw(base_size=15)+theme(legend.title=element_blank(), legend.position='top')
g <- g + xlab(TeX("$\\log_{10}$(Number of Queries to IWAL)")) + ylab("Proportion of Labels Requested")

pdf('labels_vs_accepts.pdf', width=7,height=10)
print(g)
dev.off()

# error by type
bylabel <- TRUE; loss <- 'loss_misclass'; logerr <- FALSE; 
if(bylabel <- TRUE){ RET[,rounds:=labels] } else {RET[,rounds := accepts]}
toplot <- RET[rounds > 0 & rounds %%100 == 0 & rounds < 10^4, list(err = mean(get(loss)),.N),by=list(rounds,data,alg,region,nr)]
toplot <- toplot[N==20  & alg == 'optrb']
if(logerr == TRUE){ g <- ggplot(toplot[ ! nr %in% c(2,5) ],aes(x=log(rounds,10),y=log(err,10), color=as.factor(region)))+geom_line(size=0.7)
} else{ g <- ggplot(toplot[ ! nr %in% c(2,5) ],aes(x=log(rounds,10),y=err, color=as.factor(region)))+geom_line(size=0.7) }
g <- g + facet_grid(data~nr,scales='free_y')
g <- g + theme_bw(base_size=15)+theme(legend.title=element_blank(), legend.position='top')
xlab_tex <- ifelse(bylabel,"$\\log_{10}$(Number of Labels)","$\\log_{10}$(Number of Queries to IWAL)")
lossname <- switch(loss,'loss_logistic'='Logistic Loss', 'loss_misclass'='Misclassification Loss','loss_softlogistic'='Soft Logistic Loss')
ylab_tex <- ifelse(logerr, paste0("$\\log_{10}$(",lossname,")"), lossname)
g <- g + xlab(TeX(xlab_tex)) + ylab(TeX(ylab_tex))
print(g)


