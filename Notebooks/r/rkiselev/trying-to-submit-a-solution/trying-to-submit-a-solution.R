library(ggplot2)

library(readr)

library(rpart)

library(rpart.plot)

library(RColorBrewer)



train <- read.csv("../input/train.csv")

test <- read.csv("../input/test.csv")
# Build the decision tree

my_tree <- rpart(Survived ~ Pclass + Sex +  Age + SibSp + Parch + Fare + Embarked, data = train, method = "class", cp=0, minsplit=50)

rpart.plot(my_tree, type=1)
# Make predictions on the test set

my_prediction <- predict(my_tree, newdata = test, type = "class")



# Finish the data.frame() call

my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)



# Use nrow() on my_solution

nrow(my_solution)



# Finish the write.csv() call

write.csv(my_solution, file = "my_solution.csv", row.names = FALSE)