library(dplyr,quietly = T,warn.conflicts = F)

library(lattice,quietly = T,warn.conflicts = F)

library(ggplot2,quietly = T,warn.conflicts = F)

library(corrplot,quietly = T,warn.conflicts = F)

library(MASS,quietly = T,warn.conflicts = F)

library(Matrix,quietly = T,warn.conflicts = F)

library(foreach,quietly = T,warn.conflicts = F)

library(glmnet,quietly = T,warn.conflicts = F)

library(rpart,quietly = T,warn.conflicts = F)

library(randomForest,quietly = T,warn.conflicts = F)

library(klaR,quietly = T,warn.conflicts = F)

library(kernlab,quietly = T,warn.conflicts = F)

library(lda,quietly = T,warn.conflicts = F)

library(caret,quietly = T,warn.conflicts = F)
trainset<-read.csv("../input/train.csv",header = T,sep = ",",na.strings = c("NA",""))

testset<-read.csv("../input/test.csv",header = T,sep = ",",na.strings = c("NA",""))
## combine the training and testing for the missing data treatment

data<-rbind(trainset[,-2],testset)

sapply(data,function(x){sum(is.na(x))})
table(data$Embarked)

## imputaion NA data for Embarked, 

data$Embarked[which(is.na(data$Embarked))]<-"S"

## imputaion NA data for Embarked

data$Fare[which(is.na(data$Fare))]<-mean(data$Fare[which(!is.na(data$Fare))])
table(data$Embarked)

## imputaion NA data for Embarked, 

data$Embarked[which(is.na(data$Embarked))]<-"S"

## imputaion NA data for Embarked

data$Fare[which(is.na(data$Fare))]<-mean(data$Fare[which(!is.na(data$Fare))])
data$Name<-as.character(data$Name)

## create a Title variable by picking-up the people's titles

data$Title<-gsub("^.+,","",data$Name)

data$Title<-gsub("\\..+","",data$Title)

## table for the people who have missing data in age_variable

table(data[is.na(data$Age),]$Title)
## calcul the mean for the 5 titles

data_sub<-data[!is.na(data$Age),]

meanDr<-mean(data_sub[data_sub$Title==" Dr",]$Age)

meanMaster<-mean(data_sub[data_sub$Title==" Master",]$Age)

meanMiss<-mean(data_sub[data_sub$Title==" Miss",]$Age)

meanMr<-mean(data_sub[data_sub$Title==" Mr",]$Age)

meanMrs<-mean(data_sub[data_sub$Title==" Mrs",]$Age)

meanMs<-mean(data_sub[data_sub$Title==" Ms",]$Age)

## imputation Missing Values to age

data$Age[is.na(data$Age) & data$Title==" Dr"]<-meanDr

data$Age[is.na(data$Age) & data$Title==" Master"]<-meanMaster

data$Age[is.na(data$Age) & data$Title==" Miss"]<-meanMiss

data$Age[is.na(data$Age) & data$Title==" Mr"]<-meanMr

data$Age[is.na(data$Age) & data$Title==" Mrs"]<-meanMrs

data$Age[is.na(data$Age) & data$Title==" Ms"]<-meanMs
trainset[,-2]<-data[1:891,-12]

testset<-data[892:1309,-12]



## train set treatment

trainset<-trainset[,c(-1,-4,-9,-11)]

testset<-testset[,c(-1,-3,-8,-10)]



trainset$Survived<-as.factor(trainset$Survived)
trainset_1<-trainset

## We do some transformations to have a better presentation

trainset_1$Survived<-as.character(trainset_1$Survived)

trainset_1$Survived<-gsub("0","Perished",trainset_1$Survived)

trainset_1$Survived<-gsub("1","Survived",trainset_1$Survived)

trainset_1$Survived<-as.factor(trainset_1$Survived)

percentage_survival<-round(sum(trainset$Survived==1)/nrow(trainset),3)

qplot(Survived,data=trainset_1,main="Passager Survival")
percentage_man<-round(sum(trainset_1$Sex=="male")/nrow(trainset),3)

qplot(Sex,data=trainset_1,fill=Survived)
qplot(Age,data=trainset_1,fill=Survived,binwidth=2)
qplot(Pclass,data=trainset_1,fill=Survived,binwidth=0.5)
qplot(Embarked,data=trainset_1,fill=Survived)
qplot(SibSp,data=trainset_1,fill=Survived,binwidth=0.5)
qplot(Parch,data=trainset_1,fill=Survived,binwidth=0.5)
## we will take off the variables : PassengerId, Name, Cabin, Title 

pairs(trainset)
library(corrplot)

trainset_2<-trainset

trainset_2$Sex<-as.numeric(trainset_2$Sex)

trainset_2$Embarked<-as.numeric(trainset_2$Embarked)

trainset_2$Survived<-as.numeric(trainset_2$Survived)



correlations<-cor(trainset_2)

corrplot(correlations,method="circle")
# Binary Classification machine learning

## Run algorithmes using 10-fole cross validation

set.seed(5)

trainControl<-trainControl(method = "repeatedcv",number=10,repeats = 3)

metric<-"Accuracy"

preprocess<-c("BoxCox")



## GLMNET regularized Logistic Regression

set.seed(5)

fit.glmnet<-train(Survived~., data=trainset, method="glmnet", preProc=preprocess, metric=metric, trControl=trainControl)

## LDA

set.seed(5)

fit.lda<-train(Survived~., data=trainset, method="lda", preProc=preprocess, metric=metric, trControl=trainControl)

## KNN

set.seed(5)

fit.knn<-train(Survived~., data=trainset, method="knn", preProc=preprocess,metric=metric, trControl=trainControl)

## SVM

set.seed(5)

fit.svm<-train(Survived~., data=trainset, method="svmRadial",preProc=preprocess, metric=metric, trControl=trainControl)

## CART

set.seed(5)

fit.cart<-train(Survived~., data=trainset, method="rpart", preProc=preprocess,metric=metric, trControl=trainControl)

## RF

set.seed(5)

fit.rf<-train(Survived~., data=trainset, method="rf",preProc=preprocess, metric=metric, trControl=trainControl)



results<-resamples(list(GLM=fit.glmnet, IDA=fit.lda, KNN=fit.knn, CART=fit.cart, SVM=fit.svm,RF=fit.rf))

summary(results)

dotplot(results)
set.seed(5)

grid<-expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1,10,by=1))

fit.svm.tune<-train(Survived~., data=trainset, method="svmRadial",preProc=preprocess, 

               tuneGrid=grid,metric=metric, trControl=trainControl)



print(fit.svm.tune)

plot(fit.svm.tune)
## prediction using SVM model

pred.svm<-predict(fit.svm, testset)
head(pred.svm)