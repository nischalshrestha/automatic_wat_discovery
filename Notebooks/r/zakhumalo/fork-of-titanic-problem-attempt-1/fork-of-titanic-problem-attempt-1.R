# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



list.files("../input")



# Any results you write to the current directory are saved as output.
#read Training data

train<- read.csv("../input/train.csv")

#read Test data

test<-read.csv('../input/test.csv')

str(train)

dim(train)

dim(test)

str(test)

check<-colnames(train) %in% colnames(test)

colnames(train[check==F])

ts<-table(train$Survived)

ts
prop.table(ts)
tsg<-table(train$Sex,train$Survived)

tsg

prop.table(tsg, margin=1)
tf<- test

tf$Survived<-0

tf$Survived[tf$Sex=="female"]<- 1

#writing results to a csv file

solution<-data.frame(PassengerId=tf$PassengerId,Survived=tf$Survived)

#write.csv(solution, file =  "females.csv", row.names = FALSE)
colSums(is.na(train))

colSums(is.na(test))
Train<- train

Test<-test

Test$Survived<-NA

titanic<- rbind(Train,Test)

summary(titanic)
titanic[!complete.cases(titanic$Fare),]

titanic$Fare[1044]<- mean(titanic$Fare,na.rm=TRUE)

titanic[!complete.cases(titanic$Fare),]
library(rpart)

fit_age<-rpart(Age ~Pclass +Sex +SibSp +Parch +Fare + Embarked, data= titanic[is.na(titanic$Age),] ,method= "anova")

titanic$Age[is.na(titanic$Age)]<- predict(fit_age,titanic[is.na(titanic$Age),])
library(rpart)

fit_age<-rpart(Age ~Pclass +Sex +SibSp +Parch +Fare + Embarked, data= titanic[!is.na(titanic$Age),] ,method= "anova")

titanic$Age[is.na(titanic$Age)]<- predict(fit_age,titanic[is.na(titanic$Age),])

#summary(fit_age)
train2<- titanic[1:891,]

test2<- titanic[892:1309,]
library(rattle)

library(rpart.plot)

library(RColorBrewer)
fit <- rpart(Survived ~ Pclass + Sex + Fare + Age + SibSp + Parch + Embarked, data = train2, method = 'class')  

fancyRpartPlot(fit)
round(prop.table(table(train2$Survived)),2)

round(prop.table(table(train2$Sex, train2$Survived),margin = 1),2)
my_pred <- predict(fit, newdata = test2, type = "class")

#head(my_pred)

solution2<- data.frame(PassengerId = test2$PassengerId, Survived=my_pred)
write.csv(solution2, file =  "Tsolution.csv",row.names = FALSE)



