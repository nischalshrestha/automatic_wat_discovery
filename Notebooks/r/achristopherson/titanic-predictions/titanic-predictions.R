library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(caret)

library(dplyr)



training <- read.csv("../input/train.csv", stringsAsFactors = F)

testing <- read.csv("../input/test.csv", stringsAsFactors = F)

allData <- bind_rows(training, testing)



str(allData)
allData$Survived <- as.factor(allData$Survived)

allData$Pclass <- as.factor(allData$Pclass)

allData$Sex <- as.factor(allData$Sex)

allData$Age <- as.numeric(allData$Age)

allData$SibSp <- as.numeric(allData$SibSp) 

allData$Parch <- as.numeric(allData$Parch)

allData$Cabin <- as.factor(allData$Cabin)

allData$Embarked <- as.factor(allData$Embarked)

str(allData)
allData$Title <- (gsub("(.*, )|(\\..*)", "", allData$Name ))

table(allData$Sex, allData$Title)
obscure <- c("Capt", "Col", "Don", "Dona", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir", "the Countess")

allData$Title[allData$Title == 'Ms'] <- 'Miss'

allData$Title[allData$Title == "Mlle"] <- "Miss"

allData$Title[allData$Title == "Mme"] <- "Mrs"

allData$Title[allData$Title %in% obscure] <- "Obscure"
allData$Fare[is.na(allData$Fare)] <- mean(allData$Fare, na.rm=TRUE)
agesIntact <- allData[!is.na(allData$Age),]

set.seed(799)

ageModel <- train(Age~Pclass, data=agesIntact, method="rpart")
agesToPredict <- allData[is.na(allData$Age),]

preds <- predict(ageModel, newdata=agesToPredict)
agesToPredict <- allData[is.na(allData$Age),]

preds <- predict(ageModel, newdata=agesToPredict)

allData$Age[is.na(allData$Age)] <- preds
finalTraining <- allData[1:891,]

finalTesting <- allData[892:1309,]

inCrossVal <- createDataPartition(y=finalTraining$Survived,p=0.75, list=FALSE)

realTraining <- finalTraining[inCrossVal,]

crossVal <- finalTraining[-inCrossVal,]
control <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

survivedModelFit <- train(Survived~Pclass+Title+Parch+SibSp+Fare, method="rf", data=realTraining, trControl=control)
nrow(allData)