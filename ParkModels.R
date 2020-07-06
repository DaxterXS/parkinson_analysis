#import packages
#library(microbenchmark)
library(e1071) #per il tuning e le svm
library(FNN) #per il knn
library(randomForest) #se spiega da sola
require(caTools)
library(rpart) #tree
library(glmnet) #ridge e lasso
#FUNZIONI UTILI

rsq<- function(fitted,data){
  
  return(  1- sum((data-fitted)^2)/sum( (data - mean(data) )^2) )
}


stand <- function(x, v=var(x))
{
  m <- mean(x)
  st <- (x-m)/sqrt(v)
  return(st)
}

MyRfTune<- function(x,y,mtries,nsizes,maxtrees,NF=5)
{
  folds <- cut(seq(1,nrow(x)),breaks=NF,labels=FALSE)
  
  ngrid=length(mtries)*length(nsizes)
  
  rftune<-data.frame(expand.grid(mtries,nsizes),numeric(ngrid))
  names(rftune)=c("Mtry","Nodesize","r2.V")
  
  for (j in 1:ngrid){
    for(i in 1:NF)
    {
      
      indexes <- which(folds==i,arr.ind = TRUE)
      Testx <- x[indexes,]
      Testy <- y[indexes]
      Trainx <- x[-indexes,]
      Trainy <- y[-indexes]
      MT=rftune[j,1]
      NS=rftune[j,2]
      
      
      rf.CV.M <- randomForest(x=Trainx,y=Trainy,xtest=Testx,ytest=Testy,ntree=maxtrees,mtry=MT,nodesize=NS)
      
      
      rftune[j,3]  <- rftune[j,3]+ mean(rf.CV.M$test$rsq)/NF
    }
    
  }
  nbest=min(which(rftune$r2.V==max(rftune$r2.V)))
  MT=rftune[nbest,1]
  NS=rftune[nbest,2]
  bestrf<-randomForest(x,y,ntree=maxtrees,mtry=MT,nodesize=NS,do.trace = 100)
  return(list("tune"=rftune,
              "best.model"=bestrf,
              "best.params"=rftune[nbest,]))
}




#functions
stand <- function(x, v=var(x))
{
  m <- mean(x)
  st <- (x-m)/sqrt(v)
  return(st)
}


#setwd("~/Data Spaces/Tesina")
parkinsons = read.csv(file = "parkinsons_updrs.csv", header=T)

parkinsons <- parkinsons[,-c(11,17)] #elimino variabili perfettamente correlate

attach(parkinsons)



parkX <- parkinsons[,7:20]




parkX[,-c(11,13)] <- log10(parkinsons[,-c(1:6,17,19)]) #trasformazione logaritmica, per ridurre l'asimmetria
X<-cbind(parkinsons[,2:3],parkX)

for(h in 1:ncol(parkX))
{
  parkX[,h] <- stand(parkX[,h])
}


parkX<-parkX[,-13] #levo DFA

PCA <- prcomp(parkX,center = FALSE,scale. = FALSE)
plot(PCA, type="l")
s <- summary(PCA)
View(PCA$rotation)

autoplot(PCA,loadings=TRUE,loadings.colour="blue",loadings.label=TRUE)

parkPCA.tot <- cbind(parkinsons[,c(1:6,19)],s$x[,1:13]) #dataset con tutte le PCA

parkPCA4s <- cbind(parkinsons[,c(1:6,19)],s$x[,1:4]) #dataset con PCA fino a 5
parkPCA9s <- cbind(parkinsons[,c(1:6,19)],s$x[,1:9]) #dataset con PCA fino a 9

#detach(parkinsons)
#attach(parkPCA4s)
#attach(parkPCA9s)

#predittori compreso subject.
predictors <- parkPCA4s[,-(5:6)]
#predictors$subject. = as.factor(parkinsons$subject.) #subject ? un fattore

#standardizzazione predittori
for(h in c(1,2,4,5))
{
  predictors[,h] <- stand(predictors[,h])
}

for(h in 6:(length(predictors)))
{
  predictors[,h] <- stand(predictors[,h],v=var(parkPCA4s$PC1)) #componenti principali, si standardizza con la varianza pi? grande
}

#predittori senza subject e test time
predictors <- predictors[,-c(1,4)]



#datasets separati per la predizione di motor_UPDRS e total_UPDRS
Motor <- cbind(motor_UPDRS,predictors)
Total <- cbind(total_UPDRS,predictors)


ParkM <- cbind(motor_UPDRS,X)
ParkT <- cbind(total_UPDRS,X)




#IL VERO CODICE INIZIA QUI

InfoModels <- data.frame(Data=character(), Model=character(),Params=character(),Training=double(),Validation=double(),Test=double(),stringsAsFactors = FALSE)


set.seed(17)
#DATASET CON PCA
#{ShMotor <- Motor[sample(nrow(Motor)),]
#ShTotal <- Total[sample(nrow(Total)),]
#dtype<- "PCA"}

#dataset completo
{ShMotor <- ParkM[sample(nrow(ParkM)),]
ShTotal <- ParkT[sample(nrow(ParkT)),]
dtype <- "complete"}

dm<-paste(dtype,"M",sep="-")
dt<-paste(dtype,"T",sep="-")

FR=0.8 #FRAZIONE DI DATI PER TRAINING

#divido dataset in training e test
TrMotor <- ShMotor[1:ceiling(nrow(ShTotal)*FR),]
TsMotor <- ShMotor[(ceiling(nrow(ShTotal)*FR)+1):nrow(ShTotal),]

TrTotal <- ShTotal[1:ceiling(nrow(ShTotal)*FR),]
TsTotal <- ShTotal[(ceiling(nrow(ShTotal)*FR)+1):nrow(ShTotal),]

TrMotor.Y <- TrMotor[,1]
TrMotor.X <- TrMotor[,-1]

TsMotor.Y <- TsMotor[,1]
TsMotor.X <- TsMotor[,-1]

TrTotal.Y <- TrTotal[,1]
TrTotal.X <- TrTotal[,-1]

TsTotal.Y <- TsTotal[,1]
TsTotal.X <- TsTotal[,-1]

#per il dataframe finale
dm<-paste(dtype,"M",sep="-")
dt<-paste(dtype,"T",sep="-")

#SVR
mod<-"svr"
Costs = 2^(0:7)  #2^(5:8) 
Gammas = seq(0.2,1,0.2)
Epsilons = seq(0.4,1.4,0.2)

#tuning della svm
tsvm.m<-tune.svm(TrMotor[,-1],y=TrMotor[,1],cost=Costs,gamma=Gammas,epsilon=Epsilons,scale=TRUE,tunecontrol= tune.control(cross=NF)  )
tsvm.t<-tune.svm(TrTotal[,-1],y=TrTotal[,1],cost=Costs,gamma=Gammas,epsilon=Epsilons,scale=TRUE,tunecontrol= tune.control(cross=NF)  )

#r2 di training
svr.r2.Tr.M<-rsq(tsvm.m$best.model$fitted,TrMotor[,1])
svr.r2.Tr.T<-rsq(tsvm.t$best.model$fitted,TrTotal[,1])

#r2 di test
svr.r2.Ts.M<-rsq(predict(tsvm.m$best.model,TsMotor[,-1]),TsMotor$motor_UPDRS)
svr.r2.Ts.T<-rsq(predict(tsvm.t$best.model,TsTotal[,-1]),TsTotal$total_UPDRS)

#r2 di cross validation
svr.r2.V.M<-1-tsvm.m$best.performance/var(TrMotor$motor_UPDRS)
svr.r2.V.T<-1-tsvm.m$best.performance/var(TrTotal$total_UPDRS)
 
InfoModels[nrow(InfoModels) + 1,] = list(dt,mod,I(list(tsvm.t$best.parameters)),svr.r2.Tr.T,svr.r2.V.T,svr.r2.Ts.T)
InfoModels[nrow(InfoModels) + 1,] = list(dm,mod,I(list(tsvm.m$best.parameters)),svr.r2.Tr.M,svr.r2.V.M,svr.r2.Ts.M)

            
#KNN
mod="knn-reg"
krange=1:50

knnr.mse.V.T=numeric(length(krange))
knnr.mse.V.M=numeric(length(krange))
knnr.r2.V.T=numeric(length(krange))
knnr.r2.V.M=numeric(length(krange))

for(K in krange)
{
  
  for(i in 1:NF)
  {
    indexes <- which(folds==i,arr.ind = TRUE)
    TestTotal <- TrTotal[indexes,]
    TestMotor <- TrMotor[indexes,]
    TrainTotal <- TrTotal[-indexes,]    
    TrainMotor <- TrMotor[-indexes,]
    
    knnr.model.T <- knn.reg(train = TrainTotal[,-1],test = TestTotal[,-1], y=TrainTotal$total_UPDRS,k=K)
    knnr.model.M <- knn.reg(train = TrainMotor[,-1],test = TestMotor[,-1], y=TrainMotor$motor_UPDRS,k=K)
    
    knnr.mse.V.T[K] = knnr.mse.V.T[K]+sum((knnr.model.T$pred-TestTotal$total_UPDRS)^2)/(NF*nrow(TestTotal))
    knnr.mse.V.M[K] = knnr.mse.V.M[K]+sum((knnr.model.M$pred-TestMotor$motor_UPDRS)^2)/(NF*nrow(TestMotor))    
    knnr.r2.V.T[K] = knnr.r2.V.T[K]+rsq(knnr.model.T$pred,TestTotal$total_UPDRS)/NF
    knnr.r2.V.M[K] = knnr.r2.V.M[K]+rsq(knnr.model.M$pred,TestMotor$motor_UPDRS)/NF    
  }
  
}
             
KT=which(knnr.r2.V.T==max(knnr.r2.V.T))         
KM=which(knnr.r2.V.M==max(knnr.r2.V.M))             

knnr.T = knn.reg(train= TrTotal[,-1],y=TrTotal$total_UPDRS,k=KT)
knnr.M = knn.reg(train= TrMotor[,-1],y=TrMotor$motor_UPDRS,k=KM)

knnr.r2.Tr.T <- knnr.T$R2Pred
knnr.r2.Tr.M <- knnr.M$R2Pred

knnr.T = knn.reg(train= TrTotal[,-1],test = TsTotal[,-1],y=TrTotal$total_UPDRS,k=KT)
knnr.M = knn.reg(train= TrMotor[,-1],test= TsMotor[,-1],y=TrMotor$motor_UPDRS,k=KM)


knnr.r2.Ts.T = rsq(knnr.T$pred,TsTotal$total_UPDRS)
knnr.r2.Ts.M = rsq(knnr.M$pred,TsMotor$motor_UPDRS)


InfoModels[nrow(InfoModels) + 1,] = list(dt,mod,I(list(K=KT)),knnr.r2.Tr.T,knnr.r2.V.T[K],knnr.r2.Ts.T)
InfoModels[nrow(InfoModels) + 1,] = list(dm,mod,I(list(K=KM)),knnr.r2.Tr.M,knnr.r2.V.M[K],knnr.r2.Ts.M)




#TREE (PRUNED)
mod="reg-tree"
tree.r2.V.M = 0
tree.r2.V.T = 0

#albero che overfitti
tree.M <- rpart(motor_UPDRS ~ . , data = TrMotor, minbucket=1, cp=0,xval=5)
tree.T <- rpart(total_UPDRS ~ . , data = TrTotal, minbucket=1, cp=0,xval=5)


tree.CPT.T <- tree.T$cptable
tree.CPT.M <- tree.M$cptable

#tramite xerror compreso in rpart si calcola il prune ottimale
bestCP.T=tree.CPT.T[min(which(tree.CPT.T[,"xerror"]<min(tree.CPT.T[,"xerror"]+tree.CPT.T[,"xstd"]))),"CP"]
bestCP.M=tree.CPT.M[min(which(tree.CPT.M[,"xerror"]<min(tree.CPT.M[,"xerror"]+tree.CPT.M[,"xstd"]))),"CP"]

tree.pr.T <- prune(tree.T, bestCP.T)
tree.pr.M <- prune(tree.M, bestCP.M)

for(i in 1:NF)
{
  
  indexes<- which(folds==i,arr.ind = TRUE)
  
  TestMotor <- TrMotor[indexes,]
  TrainMotor <- TrMotor[-indexes,]
  
  TestTotal <- TrTotal[indexes,]
  TrainTotal <- TrTotal[-indexes,]
  
  #parms = list(split="information")
  tree.M <- rpart(motor_UPDRS ~ . , data = TrainMotor, minbucket=1, cp=0, xval=5)
  tree.T <- rpart(total_UPDRS ~ . , data = TrainTotal, minbucket=1, cp=0, xval=5)
  
  tree.pr.T <- prune(tree.T, bestCP.T)
  tree.pr.M <- prune(tree.M, bestCP.M)
  
  
  tree.r2.V.M = tree.r2.V.M+rsq(predict(tree.pr.M,TestMotor[,-1]),TestMotor$motor_UPDRS)/NF
  tree.r2.V.T = tree.r2.V.T+rsq(predict(tree.pr.T,TestTotal[,-1]),TestTotal$total_UPDRS)/NF
  
}

tree.r2.Tr.M = rsq(predict(tree.pr.M,TrMotor[,-1]),TrMotor$motor_UPDRS)
tree.r2.Tr.T = rsq(predict(tree.pr.T,TrTotal[,-1]),TrTotal$total_UPDRS)

tree.r2.Ts.M = rsq(predict(tree.pr.M,TsMotor[,-1]),TsMotor$motor_UPDRS)
tree.r2.Ts.T = rsq(predict(tree.pr.T,TsTotal[,-1]),TsTotal$total_UPDRS)

InfoModels[nrow(InfoModels) + 1,] = list(dt,mod,I(list(CP=bestCP.T)),tree.r2.Tr.T,tree.r2.V.T,tree.r2.Ts.T)
InfoModels[nrow(InfoModels) + 1,] = list(dm,mod,I(list(CP=bestCP.M)),tree.r2.Tr.M,tree.r2.V.M,tree.r2.Ts.M)




             
#RANDOM FOREST
mod="rf-reg"
Mtries<-c(3,5,7)
if(ncol(TrTotal)>7) Mtries<- c(Mtries,ncol(TrTotal.X))



#ho dovuto fare la mia funzione personale di tuning
trf.m <- MyRfTune(x=TrMotor.X,y=TrMotor.Y,mtries=Mtries,nsizes=c(1,2,5,10),maxtrees = 600)
trf.t <- MyRfTune(x=TrTotal.X,y=TrTotal.Y,mtries=Mtries,nsizes=c(1,2,5,10),maxtrees = 1100)
#si può risparmiare del tempo mettendo solo 2000 come numero di alberi e trovando il ginocchio usando l'out of bag error
NTM=600
NTT=1100

  rf.r2.V.M = trf.m$best.params[["r2.V"]]
  rf.r2.V.T = trf.t$best.params[["r2.V"]]



rf.r2.Tr.M = rsq(predict(trf.m$best.model,TrMotor[,-1]),TrMotor$motor_UPDRS)
rf.r2.Tr.T = rsq(predict(trf.t$best.model,TrTotal[,-1]),TrTotal$total_UPDRS)


rf.r2.Ts.M = rsq(predict(trf.m$best.model,TsMotor[,-1]),TsMotor$motor_UPDRS)
rf.r2.Ts.T = rsq(predict(trf.t$best.model,TsTotal[,-1]),TsTotal$total_UPDRS)

MTT=trf.m$best.params[["Mtry"]]
MTM=trf.m$best.params[["Mtry"]]

NST=trf.t$best.params[["Nodesize"]]
NSM=trf.m$best.params[["Nodesize"]]
  
InfoModels[nrow(InfoModels) + 1,] = list(dt,mod,I(list(data.frame(ntree=NTT,nvar=MTT,nodesize=NST))),rf.r2.Tr.T,rf.r2.V.T,rf.r2.Ts.T)
InfoModels[nrow(InfoModels) + 1,] = list(dm,mod,I(list(data.frame(ntree=NTM,nvar=MTM,nodesize=NSM))),rf.r2.Tr.M,rf.r2.V.M,rf.r2.Ts.M)





#RIDGE REGRESSION

#standardizzare variabili
TrMotorS.Y<-stand(TrMotor.Y)
TrMotorS.X<-apply(TrMotor.X,2,FUN=stand)
TsMotorS.Y<-stand(TsMotor.Y)
TsMotorS.X<-apply(TsMotor.X,2,FUN=stand)
TrTotalS.Y<-stand(TrTotal.Y)
TrTotalS.X<-apply(TrTotal.X,2,FUN=stand)
TsTotalS.Y<-stand(TsTotal.Y)
TsTotalS.X<-apply(TsTotal.X,2,FUN=stand)

mod="ridge-reg"
lambdas = 10^(seq(-4,0,0.1))
alphs=(0:10)*0.1
ridge.cvm.M=numeric(length(alphs))
ridge.cvm.T=numeric(length(alphs))

for(A in alphs){
ridge.cv.M <- cv.glmnet(as.matrix(TrMotorS.X),TrMotorS.Y,lambda=lambdas,alpha=A, nfolds = NF)
ridge.cv.T <- cv.glmnet(as.matrix(TrTotalS.X),TrTotalS.Y,lambda=lambdas,alpha=A, nfolds = NF)

ridge.cvm.M[A*10+1]=min(ridge.cv.M$cvm)
ridge.cvm.T[A*10+1]=min(ridge.cv.T$cvm)

}
AM=(which(ridge.cvm.M==min(ridge.cvm.M))-1)/10
AT=(which(ridge.cvm.T==min(ridge.cvm.T))-1)/10

ridge.cv.M <- cv.glmnet(as.matrix(TrMotorS.X),TrMotorS.Y,lambda=lambdas,alpha=AM, nfolds = NF)
ridge.cv.T <- cv.glmnet(as.matrix(TrTotalS.X),TrTotalS.Y,lambda=lambdas,alpha=AT, nfolds = NF)

lambda.M <- ridge.cv.M$lambda.min
lambda.T <- ridge.cv.T$lambda.min

ridge.M <- glmnet(as.matrix(TrMotorS.X),TrMotorS.Y, lambda=lambda.M)
ridge.T <- glmnet(as.matrix(TrTotalS.X),TrTotalS.Y, lambda=lambda.T)

rid.r2.Tr.M <-rsq(predict(ridge.M,s=lambda.M,newx=as.matrix(TrMotorS.X)),TrMotorS.Y)
rid.r2.Tr.T <-rsq(predict(ridge.T,s=lambda.T,newx=as.matrix(TrTotalS.X)),TrTotalS.Y)

rid.r2.Ts.M <-rsq(predict(ridge.M,s=lambda.M,newx=as.matrix(TsMotorS.X)),TsMotorS.Y)
rid.r2.Ts.T <-rsq(predict(ridge.T,s=lambda.T,newx=as.matrix(TsTotalS.X)),TsTotalS.Y)

rid.r2.V.M <- 1-min(ridge.cv.M$cvm)/var(TrMotorS.Y)                     
rid.r2.V.T <- 1-min(ridge.cv.T$cvm)/var(TrTotalS.Y)

InfoModels[nrow(InfoModels) + 1,] = list(dt,mod,I(list(data.frame(lambda=lambda.T,alpha=AT))),rid.r2.Tr.T,rid.r2.V.T,rid.r2.Ts.T)
InfoModels[nrow(InfoModels) + 1,] = list(dm,mod,I(list(data.frame(lambda=lambda.M,alpha=AM))),rid.r2.Tr.M,rid.r2.V.M,rid.r2.Ts.M)





#LINEAR REGRESSION
mod="lr"
reg.T <- lm(total_UPDRS ~ ., data=TrTotal)
reg.M <- lm(motor_UPDRS ~ ., data=TrMotor)

s.T <- summary(reg.T)
s.M <- summary(reg.M)

lr.r2.Tr.T <- s.T$r.squared
lr.r2.Tr.M <- s.M$r.squared

lr.r2.Ts.T <- rsq(predict(reg.T,TsTotal.X),TsTotal.Y)
lr.r2.Ts.M <- rsq(predict(reg.M,TsMotor.X),TsMotor.Y)

lr.r2.V.M = 0
lr.r2.V.T = 0

for(i in 1:NF)
{
  
  indexes<- which(folds==i,arr.ind = TRUE)
  
  TestMotor <- TrMotor[indexes,]
  TrainMotor <- TrMotor[-indexes,]
  
  TestTotal <- TrTotal[indexes,]
  TrainTotal <- TrTotal[-indexes,]
  
  lr.M <- lm(motor_UPDRS ~ . , data = TrainMotor)
  lr.T <- lm(total_UPDRS ~ . , data = TrainTotal)
  
  lr.r2.V.M = lr.r2.V.M+rsq(predict(lr.M,TestMotor[,-1]),TestMotor$motor_UPDRS)/NF
  lr.r2.V.T = lr.r2.V.T+rsq(predict(lr.T,TestTotal[,-1]),TestTotal$total_UPDRS)/NF
  
}

InfoModels[nrow(InfoModels) + 1,] = list(dt,mod,NA,lr.r2.Tr.T,lr.r2.V.T,lr.r2.Ts.T)
InfoModels[nrow(InfoModels) + 1,] = list(dm,mod,NA,lr.r2.Tr.M,lr.r2.V.M,lr.r2.Ts.M)
