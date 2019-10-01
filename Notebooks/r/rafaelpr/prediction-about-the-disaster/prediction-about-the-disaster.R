# loading packages

library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(rattle) 

library(rpart.plot)

library(RColorBrewer)
# Input data files are available in the "../input/" directory.

#First of all, We need to import train and test sets:

train <- read.csv("../input/train.csv") 

test <- read.csv("../input/test.csv")
#How many people survived the disaster with the Titanic? 

table(train$Survived)

prop.table(table(train$Survived))



#How many people survived the disaster with the Titanic by gender?

table(train$Sex, train$Survived)

prop.table(table(train$Sex, train$Survived), 1)
#Build the decision tree

library(rpart)

my_tree <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train, method = "class")

plot(my_tree)

text(my_tree)



#Plotting the decision tree

fancyRpartPlot(my_tree)
#Make predictions on the test set

prediction <- predict(my_tree, newdata = test, type = "class")



#Write the data.frame

my_solution <- data.frame(PassengerId = test$PassengerId, Survived = prediction)

write.csv(my_solution, file = "rribeiro1_titanic_solution_v2.csv", row.names = FALSE)