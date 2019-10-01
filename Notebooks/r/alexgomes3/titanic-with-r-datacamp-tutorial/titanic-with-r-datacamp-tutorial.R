# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



list.files("../input")



# Any results you write to the current directory are saved as output.



#########################################3

############## my code starts here



# decision trees library

library(rpart)



# Load in the packages to build a fancy plot

library(rpart.plot)

library(RColorBrewer)

train = read.csv("../input/train.csv")

test = read.csv("../input/test.csv")
# Your train and test set are still loaded in

str(train)



# Build the decision tree

my_tree_two <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train, method = "class")



# Visualize the decision tree using plot() and text()

plot(my_tree_two)

text(my_tree_two)



# Time to plot your fancy tree

fancyRpartPlot(my_tree_two)
my_tree_two
# Make predictions on the test set

my_prediction <- predict(my_tree_two, newdata = test, type = "class")



# Finish the data.frame() call

my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)



# Use nrow() on my_solution

nrow(my_solution)



# Finish the write.csv() call

write.csv(my_solution, file = "my_solution.csv", row.names = FALSE)
# Prediction 3

my_tree_three <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,

                     data = train, method = "class", control = rpart.control(minsplit = 50, cp = 0))



# Visualize my_tree_three

fancyRpartPlot(my_tree_three)



# predict

my_prediction3 <- predict(my_tree_three, newdata = test, type = "class")



# Finish the data.frame() call

my_solution3 <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction3)



# Finish the write.csv() call

write.csv(my_solution3, file = "my_solution3.csv", row.names = FALSE)