train <- read.csv(file = "../input/train.csv", stringsAsFactors = FALSE, header = TRUE)

test <- read.csv(file = "../input/test.csv", stringsAsFactors = FALSE, header = TRUE)



# EXPLORING THE DATA

str(train)

head(train, n=10)
# SURVIVAL RATE

# What is the percentage of those that had survived the titanic disaster?

prop.table(table(train$Survived)) * 100
# SURVIVAL RATE BY SEX

prop.table(table(train$Survived, train$Sex), 1) * 100
# SURVIVAL RATE BY PCLASS

prop.table(table(train$Survived, train$Pclass), 1) * 100
# SURVIVAL RATE BY SEX AND PCLASS

prop.table(table(train$Pclass, train$Sex, train$Survived), 1) * 100
# RESET COLUMN

test$Survived <- rep(0, 418)



# MAKE PREDICTION

test$Survived <- 0

test$Survived[test$Sex == 'female'] <- 1
# WRITE TO FILE

submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)

write.csv(submit, file = "TitanicPrediction.csv", row.names = FALSE)