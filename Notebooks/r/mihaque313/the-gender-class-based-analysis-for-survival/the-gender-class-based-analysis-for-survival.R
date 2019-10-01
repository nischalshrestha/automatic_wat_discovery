library(readr)

train <- read_csv("../input/train.csv") #importing the train.csv file
test <- read_csv("../input/test.csv")   #importing the test.csv file
train$fareRange<-'30+'

train$fareRange[train$Fare<30 & train$Fare>=20] <-'20-30'

train$fareRange[train$Fare<20 & train$Fare>=10] <-'10-20'

train$fareRange[train$Fare<10] <-'<10'
aggregate(Survived~Sex+Pclass+fareRange, data=train, FUN = function(x){sum(x)/length(x)})
test$Survived<-0

test$Survived[test$Sex=='female']<-1

test$Survived[test$Sex=='female' & test$Pclass==3 & test$Fare>=20]<-0

submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)

write.csv(submit, file = "ResultGC.csv", row.names = FALSE)