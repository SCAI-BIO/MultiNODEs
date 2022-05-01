############# README
# This file runs the Bayesian network and saves out VPs
############# 

rm(list=ls())
library(tidyverse)
library(beepr)
library(arules)
library(mclust)
library(rpart)
library(bnlearn)
library(parallel)
# general helpers
source('/home/pwendland/Dokumente/GitHub/mnode/BN_prior/simulate_VP.R')

# Name output files
name<-'main'
data_out<-paste0('/home/pwendland/Dokumente/GitHub/mnode/BN_prior/',name)
scr<-"bic-cg" # BN score
pred-loglik-cg
aic-cg
loglik-cg
mth<-"mle" # BN method

longitudinal=read.csv('/home/pwendland/Dokumente/GitHub/mnode/BN_prior/longitudinal_latent_ppmi.csv')[,2:48]
static=read.csv('/home/pwendland/Dokumente/GitHub/mnode/BN_prior/static_latent_ppmi.csv')[,2:4]
mixed=read.csv('/home/pwendland/Dokumente/GitHub/mnode/BN_prior/mixture_latent_ppmi.csv')[,2:7]
colnames(longitudinal)=c('l1','l2','l3','l4','l5','l6','l7','l8','l9','l10','l11','l12','l13','l14','l15','l16','l17','l18','l19','l20','l21','l22','l23','l24','l25','l26','l27','l28','l29','l30','l31','l32','l33','l34','l35','l36','l37','l38','l39','l40','l41','l42','l43','l44','l45','l46','l47')
colnames(static)=c('s1','s2','s3')
colnames(mixed)=c('m1','m2','m3','m4','m5','m6')
mixed[mixed>0.5]=1
mixed[mixed<0.5]=0
mixed[] <- lapply(mixed, factor)

dat=as.data.frame(c(mixed,static,longitudinal))

############################
############################ Bnet
############################

# Make bl/wl
bl_1=tiers2blacklist(list(colnames(static),colnames(longitudinal)))
bl_2=tiers2blacklist(list(colnames(mixed),colnames(longitudinal)))
bl_3=tiers2blacklist(list(colnames(mixed),colnames(static)))
bl=do.call("rbind", list(bl_1, bl_2, bl_3))


# Final bayesian network
#maxp could be removed or set to inf it is the number of max parents per node
finalBN = tabu(dat, blacklist=bl,  score=scr)
saveRDS(finalBN,paste0(data_out,'_finalBN.rds'))
# save fitted network
real = dat
#real$SUBJID<-NULL
finalBN<-readRDS(paste0(data_out,'_finalBN.rds'))
mth='mle'
fitted = bn.fit(finalBN, real, method=mth)
saveRDS(fitted,paste0(data_out,'_finalBN_fitted.rds'))

# Virtual Patient Generation
virtual<-simulate_VPs(fitted,n=354)
virtual_mnode=cbind(virtual[,grep('l',colnames(virtual))],virtual[,grep('s',colnames(virtual))])

# save out virtual patients

saveRDS(virtual_mnode,paste0(data_out,'_VirtualPPts.rds'))
write.csv(virtual_mnode,paste0(data_out,'_VirtualPPts.csv'),row.names=FALSE)


