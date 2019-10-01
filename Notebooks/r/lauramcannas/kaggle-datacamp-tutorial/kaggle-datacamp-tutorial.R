# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")

# Any results you write to the current directory are saved as output.
#load train and test dataset

train <- read_csv ("../input/train.csv")

test <- read_csv ("../input/test.csv")

# understanding data

summary(train)



# decision tree as it is proposed in the Kaggle DataCamp tutorial

library(rpart)

#library(rattle)



my_tree <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train, method = "class")

#plot(my_tree)

#fancyRpartPlot(my_tree)



my_prediction <- predict(my_tree, test, type = "class")

my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)

#write.csv(my_solution, file = "my_solution.csv", row.names = FALSE)

write.csv(my_solution, file = "my_solution.csv", row.names = FALSE)