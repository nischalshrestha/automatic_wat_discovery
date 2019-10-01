# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
data <- read.csv("../input/train.csv", sep=",")

summary(data)

ggplot(data) + geom_bar(aes(x=Survived))
head(data)
summary(data$Age)

ggplot(data) + geom_density(aes(x=Age))

data <- data[!is.na(data$Age), ]
excludedVar <- c("PassengerId", "Name", "Ticket", "Cabin", "Embarked")

includedVar <- setdiff(names(data), excludedVar)

respondVar <- c("Survived")



trainingData <- data[, includedVar]

predictorVar <- setdiff(includedVar, respondVar)
summary(trainingData[, respondVar])

summary(trainingData[, predictorVar])
str(trainingData)
library(C50)

model <- C5.0(y = as.factor(trainingData[, respondVar]), x = as.matrix(trainingData[, predictorVar]))

predictions <- predict(model, as.matrix(trainingData[, predictorVar]), type="class")
summary(predictions)
library(caret)

result <- confusionMatrix(predictions, trainingData[, respondVar], positive="1")

result
testData <- read.csv("../input/test.csv")

testData <- testData[, predictorVar]

summary(testData)
testPredictions <- predict(model, as.matrix(testData[, predictorVar]), type="class")

summary(testPredictions)

str(testPredictions)

length(testPredictions)

testPredictions

as.matrix(testData[1:35, predictorVar])