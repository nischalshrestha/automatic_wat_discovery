# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
train <- read.csv('../input/train.csv', sep = ',', na.strings = '')

test <- read.csv('../input/test.csv', sep = ',', na.strings = '')
dim(train)

dim(test)
head(train)
tail(train)
str(train)
naNumbers <- colMeans(is.na(train))

naNumbers[naNumbers > 0]
model1 <- glm(Survived ~ Pclass, data = train)
predictTrain <- predict(model1, train)

predictedSurvival <- rep(0,dim(train)[1])

predictedSurvival[predictTrain > 0.5] = 1
table(predictedSurvival,train$Survived)
ErrorRate = (80 + 206) / (469+206+80+136)

print(ErrorRate)
predictTest <- predict(model1, test)

predictedTestSurvival <- rep(0,dim(test)[1])

predictedTestSurvival[predictTest > 0.5] = 1
pTable <- data.frame(test$PassengerId, as.integer(predictedTestSurvival), )
names(pTable) <- c("PassagerId","Survived")
write.csv(pTable,"mySubssion.csv")
pTable