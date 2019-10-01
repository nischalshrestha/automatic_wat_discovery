# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
train<- read.csv("../input/train.csv",stringsAsFactor=FALSE)

test<- read.csv("../input/test.csv",stringsAsFactor=FALSE)

train$Sex[which(train$Sex=="female")]<-1

train$Sex[which(train$Sex=="male")]<-0



for(i in c("Master.","Miss.","Mrs.","Mr.","Dr.")){

train$Name[grep(i,train$Name,fixed=TRUE)]<-i}



masterage<-mean(train$Age[which(train$Name=="Master.")],trim=.5,na.rm=TRUE)

missage<-mean(train$Age[which(train$Name=="Miss.")],trim=.5,na.rm=TRUE)

mrsage<-mean(train$Age[which(train$Name=="Mrs.")],trim=.5,na.rm=TRUE)

mrage<-mean(train$Age[which(train$Name=="Mr.")],trim=.5,na.rm=TRUE)

drage<-mean(train$Age[which(train$Name=="Dr.")],trim=.5,na.rm=TRUE)



train$Age[is.na(train$Age)&train$Name=="Master."]<-masterage

train$Age[is.na(train$Age)&train$Name=="Miss."]<-missage

train$Age[is.na(train$Age)&train$Name=="Mrs."]<-mrsage

train$Age[is.na(train$Age)&train$Name=="Mr."]<-mrage

train$Age[is.na(train$Age)&train$Name=="Dr."]<-drage



train$Child<- 0

train[which(train$Age<14),c("Child")]<-1



train$Family<- NA



for(i in 1:nrow(train)){

train$Family[i]<-train$SibSp[i]+train$Parch[i]+1

}



train$Mother<- 0

train[which(train$Parch>0 & train$Age>18 & train$Name=="Mrs."),c("Mother")]<-1



train$Cabin[which(!train$Cabin=="")]<-1

train$Cabin[which(train$Cabin=="")]<-0



#建模

train.glm<- glm(Survived~Age+Child+Family+Sex*Pclass+Cabin,family=binomial,data=train)



#test

test$Sex[which(test$Sex=="female")]<-1

test$Sex[which(test$Sex=="male")]<-0



for(i in c("Master.","Miss.","Mrs.","Mr.","Dr.")){

test$Name[grep(i,test$Name,fixed=TRUE)]<-i}



masterage<-mean(test$Age[which(test$Name=="Master.")],trim=.5,na.rm=TRUE)

missage<-mean(test$Age[which(test$Name=="Miss.")],trim=.5,na.rm=TRUE)

mrsage<-mean(test$Age[which(test$Name=="Mrs.")],trim=.5,na.rm=TRUE)

mrage<-mean(test$Age[which(test$Name=="Mr.")],trim=.5,na.rm=TRUE)

drage<-mean(test$Age[which(test$Name=="Dr.")],trim=.5,na.rm=TRUE)



test$Age[is.na(test$Age)&test$Name=="Master."]<-masterage

test$Age[is.na(test$Age)&test$Name=="Miss."]<-missage

test$Age[is.na(test$Age)&test$Name=="Mrs."]<-mrsage

test$Age[is.na(test$Age)&test$Name=="Mr."]<-mrage

test$Age[is.na(test$Age)&test$Name=="Dr."]<-drage

test$Age[which(is.na(test$Age))]<- 20



test$Child<- 0

test[which(test$Age<14),c("Child")]<-1



test$Family<- NA



for(i in 1:nrow(test)){

test$Family[i]<-test$SibSp[i]+test$Parch[i]+1

}



test$Mother<- 0

test[which(test$Parch>0 & test$Age>18 & test$Name=="Mrs."),c("Mother")]<-1



test$Cabin[which(!test$Cabin=="")]<-1

test$Cabin[which(test$Cabin=="")]<-0



#预测

p.hats<- predict.glm(train.glm,newdata=test,type="response")



survival <- NA

for(i in 1:length(p.hats)) {

  if(p.hats[i] > .5) {

    survival[i] <- 1

  } else {

    survival[i] <- 0

  }

}



kaggle.sub <- cbind(test$PassengerId,survival)

colnames(kaggle.sub) <- c("PassengerId", "Survived")

write.csv(kaggle.sub, file = "kaggle7.csv", row.names = FALSE)