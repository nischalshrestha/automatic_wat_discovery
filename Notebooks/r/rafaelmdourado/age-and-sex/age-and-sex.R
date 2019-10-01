try(library('ggplot2') , silent=TRUE)

try(library('dplyr') , silent=TRUE)
train <- read.csv('../input/train.csv', stringsAsFactors = F)

train$isTrain <- T

test  <- read.csv('../input/test.csv', stringsAsFactors = F)

test$isTrain <- F



full  <- bind_rows(train, test) # bind training & test data



# check data

str(full)
full$underTen <- F

full$underTen[full$Age < 10 & full$Age > 0] <- T



ggplot(full[full$isTrain,], aes(x = underTen, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') 
full$isFemale <- F

full$isFemale[full$Sex == "female"] <- T



ggplot(full[full$isTrain,], aes(x = isFemale, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') 
full$isAlone <- T

full$isAlone[full$Parch > 0] <- F

full$isAlone[full$SibSp > 0] <- F



ggplot(full[full$isTrain,], aes(x = isAlone, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') 
set.seed(42)



totalrows <- nrow(full[full$isTrain,])

trainrows <- sample(c(T,F), totalrows, replace = T, prob = c(0.8, 0.2))



trainningSet <- full[full$isTrain,][trainrows,]

testingSet <- full[full$isTrain,][!trainrows,]
model <- lm(Survived ~ isFemale + underTen + isAlone, trainningSet)



summary(model)
prediction <- predict(model, testingSet)



pdf <- data.frame(F = testingSet$Survived, P = as.integer(prediction > 0.3))



table(pdf)



print("Acc")

nrow(pdf[pdf$P == pdf$F,])/nrow(pdf)
test <- full[892:1309,]



prediction <- predict(model, test)



solution <- data.frame(PassengerID = test$PassengerId, Survived = as.integer(prediction > 0.3))



write.csv(solution, file = 'out.csv', row.names = F)