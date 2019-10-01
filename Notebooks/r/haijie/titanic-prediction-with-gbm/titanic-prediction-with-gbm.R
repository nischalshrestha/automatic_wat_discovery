library(lattice,quietly = T)

library(ggplot2,quietly = T)

library(plyr,quietly = T)

library(dplyr,quietly = T)

library(foreach,quietly = T)

library(iterators,quietly = T)

library(parallel,quietly = T)

library(survival,quietly = T)

library(splines,quietly = T)



library(corrplot,quietly = T)

library(doParallel,quietly = T)

library(gbm,quietly = T)

library(pROC,quietly = T)

library(xgboost,quietly = T)

library(caret,quietly = T)
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
## training set treatment

trainset<-trainset[,c(-1,-4,-9,-11)]



trainset$Survived[trainset$Survived==0]<-"Perished"

trainset$Survived[trainset$Survived==1]<-"Survived"

levels(trainset$Survived)<-c("Perished","Survived")



Xtrainset<-trainset[,-1]

Ytrainset<-trainset[,1]



## test set treatment

Xtestset<-testset[,c(-1,-3,-8,-10)]



##  as.numeric pour les variables

Xtrainset$Sex<-as.numeric(Xtrainset$Sex)

Xtrainset$Embarked<-as.numeric(Xtrainset$Embarked)



Xtestset$Sex<-as.numeric(Xtestset$Sex)

Xtestset$Embarked<-as.numeric(Xtestset$Embarked)
### Generalized bosted regression medel (BGM)

# Set up training control

ctrl <- trainControl(method = "repeatedcv",   # 10fold cross validation

                     number = 5,	# do 5 repititions of cv

                     summaryFunction=twoClassSummary,	# Use AUC to pick the best model

                     classProbs=TRUE,

                     allowParallel = TRUE)

 



# Use the expand.grid to specify the search space	

# Note that the default search grid selects multiple values of each tuning parameter

 

grid <- expand.grid(interaction.depth=c(1,2), # Depth of variable interactions

                    n.trees=c(10,20),	        # Num trees to fit

                    shrinkage=c(0.01,0.1),		# Try 2 values for learning rate 

                    n.minobsinnode = 20)

#											

set.seed(1951)  # set the seed



# Set up to do parallel processing   

registerDoParallel(4)		# Registrer a parallel backend for train

getDoParWorkers()

 

gbm.tune <- train(x=Xtrainset,y=trainset$Survived,

                              method = "gbm",

                              metric = "ROC",

                              trControl = ctrl,

                              tuneGrid=grid,

                              verbose=FALSE)
# Look at the tuning results

# Note that ROC was the performance criterion used to select the optimal model.   



gbm.tune$bestTune

plot(gbm.tune)  		# Plot the performance of the training models
res <- gbm.tune$results

res
### GBM Model Predictions and Performance

# Make predictions using the test data set

gbm.pred <- predict(gbm.tune,Xtestset)

head(gbm.pred)
rValues <- resamples(gbm=gbm.tune)

rValues$values

summary(rValues)
bwplot(rValues,metric="ROC",main="GBM Accuracy")	# boxplot

dotplot(rValues,metric="ROC",main="GBM Accuracy")	# dotplot