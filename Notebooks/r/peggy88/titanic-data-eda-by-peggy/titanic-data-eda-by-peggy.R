# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



train <- read.csv("../input/train.csv")

test <- read.csv("../input/test.csv")

# View the structure of training dataset

str(train)

# View the structure of test dataset

str(test)
#We notice that some features are not of the type they are supposed to be in. 

#Hence we observe and transform the data type of the variables accordingly.



#Convert name feature to character

train$Name <- as.character(train$Name)

test$Name <- as.character(test$Name)



#Convert Survived and PClass to factor

train$Survived <- as.factor(train$Survived)

train$Pclass <- as.factor(train$Pclass)



test$Pclass <- as.factor(test$Pclass)



str(train)
#Checking how many survived in training data

table(train$Survived)

#Checking summary of dataset

summary(train)
plot(train$Sex, train$Survived, col=c("red","blue"))
#Check for missing values

colSums(is.na(train))

colSums(is.na(test))



#We can deal with missing values in many ways. 

#since Fare variable is missing only one value in test data, lets fill it

train2 <- train

test2 <- test

test2$Survived <- NA

full <- rbind(train2, test2)



full[!complete.cases(full$Fare),]

full$Fare[1044] <- median(full$Fare, na.rm = TRUE)

full[!complete.cases(full$Fare),]

#Fill in Age values now

train[is.na(train)] <- median(train$Age, na.rm = TRUE)

test[is.na(test)] <- median(test$Age, na.rm = TRUE)



#Lets split the full data into train and test data again

traindata <- full[1:891,]

testdata <- full[892:1309,]



dim(traindata)

dim(testdata)
#Building a classification model to predict survival status of test data

library(rpart)

dt <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=traindata, method= "class")

prediction <- predict(dt, newdata = testdata, type = "class")



submission <- data.frame(PassengerId = testdata$PassengerId, Survived = prediction)

write.csv(submission, file =  "gender_submission.csv", row.names = FALSE)


