# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
library("stats")

library("base")
train = read.csv('../input/train.csv')

test = read.csv('../input/test.csv')

test$Survived <- NA 
combined <- rbind.data.frame(train,data=test) 

nrow(train)

nrow(test)

nrow(combined)
str(combined, give.attr = FALSE)

colSums(is.na(combined))

colnames(combined)

combined=combined[,-c(11,12,4,9)]
str(combined)

combined$Survived <- factor(combined$Survived)

combined$Pclass <- factor(combined$Pclass)

combined$Age=ifelse(is.na(combined$Age),median(na.omit(combined$Age)),combined$Age)

combined$Fare=ifelse(is.na(combined$Fare),median(na.omit(combined$Fare)),combined$Fare)
chiSqStat=NULL

for (i in 2:(ncol(combined))) 

{ if(is.factor(combined[,i]))

{

  ChiSqTest=chisq.test(x=combined$Survived,

                       y=combined[,i])

  chiSqStat=rbind.data.frame(chiSqStat,

                             cbind.data.frame(

                               variable.names=colnames(combined)[i],

                               chi_sq_value=ChiSqTest$statistic,

                               p_value=ChiSqTest$p.value))

  cat("\n",colnames(combined)[i],"\n","chi-sq value:",

      ChiSqTest$statistic,"pvalue:",ChiSqTest$p.value,"\n")

  cat("*************************")

}

}
train<-combined[1:891,]

test<-combined[892:1309,]

Logistic_Model_1=glm(Survived~.,family = binomial,data=train,maxit=100)

predict_Probs=predict(Logistic_Model_1,test)

Predict_Class = ifelse(predict_Probs >= 0.5,1,0)

submission = cbind("PassengerId"= test$PassengerId,"Survived"=Predict_Class)

write.csv(submission, "titanictestfinaldeb output1.csv", row.names = FALSE)