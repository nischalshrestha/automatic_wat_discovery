library(caret)



trainData1 <- (read.csv("../input/train.csv"))

testData1 <- (read.csv("../input/test.csv"))



#check for NA

summary(trainData1)



#fixing NA

trainData1$Age[which(is.na(trainData1$Age))] <- mean(as.numeric(trainData1$Age[which(!is.na(trainData1$Age))]))

trainData1$Age <- as.numeric(trainData1$Age)



testData1$Age[which(is.na(testData1$Age))] <- mean(as.numeric(testData1$Age[which(!is.na(testData1$Age))]))

testData1$Age <- as.numeric(testData1$Age)





modelTitanic14 <- glm(Survived~(Pclass+Sex+Age+SibSp)^2, data = trainData1, family=binomial("logit"))

summary(modelTitanic14)



#prediction and conversion to 0/1

predTitanic <- as.data.frame(predict(modelTitanic14, testData1, type="response"))

colnames(predTitanic) <-c("Survived")

predTitanic$Survived[predTitanic$Survived>0.5] <- 1

predTitanic$Survived[predTitanic$Survived!=1] <- 0

predTitanic['PassengerId'] <- testData1$PassengerId



#output

write.csv(predTitanic, "predTitanic.csv", row.names = F, col.names = T)



#classifier testing code

predTitanicTrain <- as.data.frame(predict(modelTitanic14, trainData1, type="response"))

colnames(predTitanicTrain) <-c("Survived")

predTitanicTrain['PassengerId'] <- trainData1$PassengerId



predTitanicTrain01 <- predTitanicTrain



predTitanicTrain01$Survived <- ifelse(predTitanicTrain01$Survived>0.5,1,0)





conMatTitanic <- confusionMatrix(trainData1$Survived,predTitanicTrain01$Survived,positive = '1')

conMatTitanic




