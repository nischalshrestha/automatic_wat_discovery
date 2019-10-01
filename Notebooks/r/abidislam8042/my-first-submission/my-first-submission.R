library(readr)

titanic <-   as.data.frame(read_csv("../input/titanic/train.csv"))



library(rpart)



fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,

             data=titanic,

             method="class")
fit
test <-   as.data.frame(read_csv("../input/titanic/test.csv"))
Prediction <- predict(fit, test, type = "class")

submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)

 write.csv(submit, file = "myfirstdtree.csv", row.names = FALSE)
Prediction