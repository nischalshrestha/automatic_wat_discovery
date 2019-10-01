# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train<-read.csv("../input/train.csv")

summary(train)



# Any results you write to the current directory are saved as output.
#feature identification- Pclass

#No. of ppl survived/Total number of ppl in that class

aggregate(train[,2], list(train[,3]), mean)

aggregate(train[,2], list(train[,3]), length)


#feature identification- sex

#No. of ppl survived/Total number of ppl in that class

aggregate(train[,2], list(train[,5]), mean)

aggregate(train[,2], list(train[,5]), length)
#feature identification- age

#No. of ppl survived/Total number of ppl in that class

train$agegroup<-0

train$agegroup[which(train[,6]>20)]<-1



#train$agegroup<-train[,6]%/%10





aggregate(train[,2], list(train$agegroup), mean)

aggregate(train[,2], list(train$agegroup), length)
#feature identification- siblings

#No. of ppl survived/Total number of ppl in that class

train$siblingyn<-0

train$siblingyn[which(train[,7]>0)]<-1

aggregate(train[,2], list(train$siblingyn), mean)

aggregate(train[,2], list(train$siblingyn), length)
#feature identification- parent-child

#No. of ppl survived/Total number of ppl in that class

train$parentyn<-0

train$parentyn[which(train[,8]>0)]<-1

aggregate(train[,2], list(train$parentyn), mean)

aggregate(train[,2], list(train$parentyn), length)
library(modeest)



#feature identification- fare

#No. of ppl survived/Total number of ppl in that class

qplot(train$Fare, geom="histogram", binwidth=10) 

mlv(train$Fare, method = "mfv")


#feature identification- fare

#No. of ppl survived/Total number of ppl in that class

train$farediv<-0

train$farediv[which(train[,10]>10)]<-1

train$farediv[which(train[,10]>20)]<-2

train$farediv[which(train[,10]>30)]<-3







aggregate(train[,2], list(train$farediv), mean)

aggregate(train[,2], list(train$farediv), length)


#feature identification- embarked

#No. of ppl survived/Total number of ppl in that class





aggregate(train[,2], list(train[,12]), mean)

aggregate(train[,2], list(train[,12]), length)
#feature identification- embarked

#No. of ppl survived/Total number of ppl in that class



train$embdiv<-0

train$embdiv[which(train[,12]=='C')]<-1



aggregate(train[,2], list(train$embdiv), mean)

aggregate(train[,2], list(train$embdiv), length)
train$farediv<-0

train$farediv[which(train[,10]>10)]<-1

train$farediv[which(train[,10]>30)]<-2









aggregate(train[,2], list(train$farediv), mean)

aggregate(train[,2], list(train$farediv), length)

x<-read.csv("../input/train.csv")



x$psex<-0

x$psex[which(x$Sex=='female')]<-1



x$pfare<-0

x$pfare[which(x$Fare>10)]<-.6

x$pfare[which(x$Fare>30)]<-1





x$ppclass<-0

x$ppclass[which(x$Pclass=='2')]<-.6

x$ppclass[which(x$Pclass=='1')]<-1



x$pembarked<-0

x$pembarked[which(x$Embarked=='C')]<-1



x$score<-(x$psex*.56)+(x$pfare*.39)+(x$ppclass*.38)+(x$pembarked*.21)



#summary(x)

quantile(x$score,c(.25,.61,.75))

#write.csv(x,"/home/gauthaman/Downloads/soln.csv")

#aggregate(x[,2], list(x$pembarked), mean)
#feature identification- siblings

#No. of ppl survived/Total number of ppl in that class

sum(train[,2])
#predictor



library(readr) # CSV file I/O, e.g. the read_csv function

library(rpart)



x<-read.csv("../input/train.csv")



x$psex<-0

x$psex[which(x$Sex=='female')]<-1



x$pfare<-0

x$pfare[which(x$Fare>10)]<-.6

x$pfare[which(x$Fare>30)]<-1





x$ppclass<-0

x$ppclass[which(x$Pclass=='2')]<-.6

x$ppclass[which(x$Pclass=='1')]<-1



x$pembarked<-0

x$pembarked[which(x$Embarked=='C')]<-1



x$score<-(x$psex*.56)+(x$pfare*.39)+(x$ppclass*.38)+(x$pembarked*.21)

#survive if score>.77



#aggregate(x[,2], list(x$servpred), mean)

#write.csv(x, file="submissiontest.csv" ,row.names=FALSE)
#decision tree predictor



library(readr) # CSV file I/O, e.g. the read_csv function

library(rpart)



x<-read.csv("../input/train.csv")





x$pfare<-0

x$pfare[which(x$Fare>10)]<-.6

x$pfare[which(x$Fare>30)]<-1







#x$score<-(x$psex*.56)+(x$pfare*.39)+(x$ppclass*.38)+(x$pembarked*.21)

fit <- rpart(Survived ~ Sex + Fare + Pclass+SibSp+Parch + Embarked,

               data=x,

               method="class")

#survive if score>.77



#aggregate(x[,2], list(x$servpred), mean)

#write.csv(x, file="submissiontest.csv" ,row.names=FALSE)
library(rattle)

library(rpart.plot)

library(RColorBrewer)
fancyRpartPlot(fit)
x<-read.csv("../input/test.csv")





x$pfare<-0

x$pfare[which(x$Fare>10)]<-.6

x$pfare[which(x$Fare>30)]<-1







Prediction <- predict(fit, x, type = "class")

submit <- data.frame(PassengerId = x$PassengerId, Survived = Prediction)

write.csv(submit, file = "decisiontreeoutput1.csv", row.names = FALSE)
fancyRpartPlot(fit)#decision tree straightshoot



library(readr) # CSV file I/O, e.g. the read_csv function

library(rpart)



x<-read.csv("../input/train.csv")









fit <- rpart(Survived ~ Sex + Fare + Pclass+ SibSp + Parch + Embarked+ Age,

               data=x,

               method="class")



t<-read.csv("../input/test.csv")



fancyRpartPlot(fit)



Prediction <- predict(fit, t, type = "class")

submit <- data.frame(PassengerId = t$PassengerId, Survived = Prediction)

write.csv(submit, file = "decisiontreeoutput2.csv", row.names = FALSE)
