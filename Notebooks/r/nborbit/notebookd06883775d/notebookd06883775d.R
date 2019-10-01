# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.



library(caret)

library(rpart.plot)
train <- read.csv('../input/train.csv', stringsAsFactors = F)

test  <- read.csv('../input/test.csv', stringsAsFactors = F)



head(train)



#data preparation

train$Survived <- as.factor(train$Survived)

train$Pclass <- as.factor(train$Pclass)

train$Sex <- as.factor(train$Sex)

train$Embarked <- as.factor(train$Embarked)



#train only on complete cases

trainCompleteCases <- train[complete.cases(train),]



inTrain <- createDataPartition(y = trainCompleteCases$Survived, p = .85, list = FALSE)

training <- trainCompleteCases[ inTrain,]

testing <- trainCompleteCases[-inTrain,]



model <- train(

  Survived ~ Pclass + Sex + Age + Fare + SibSp + Parch + Embarked, 

  training,

  method="rpart2",

  na.action = na.pass)



testPred <- predict(model, testing, na.action = na.pass)

postResample(testPred, testing$Survived) 



rpart.plot(model$finalModel)



test$Pclass <- as.factor(test$Pclass)

test$Sex <- as.factor(test$Sex)

test$Embarked <- as.factor(test$Embarked)



test$Survived <- predict(model, test, na.action = na.pass)



write.csv(test[,c("PassengerId","Survived")], file = 'submission.csv', row.names = F)