# Loading packages

library(rpart) #Generate decision tree

library(class) #Generate K-Nearest Neighbor



#importing the data

train <- read.csv("../input/train.csv", stringsAsFactor = F) #importing training set

test <- read.csv("../input/test.csv", stringsAsFactor = F) #importing test set
head(train)

str(train)
#Calculate missing values in Pclass

sum(is.na(train$Pclass))



#Calculate missing values in Age

sum(is.na(train$Age))



#Calculate missing values in SibSp

sum(is.na(train$SibSp))



#Calculate missing values in Parch

sum(is.na(train$Parch))



#Calculate missing values in Fare

sum(is.na(train$Fare))



#Calculate missing values in Embarked

sum(is.na(train$Embarked))
clean_train <- na.omit(train)
# Turning Age into a factor.

clean_train$Age <- as.numeric(clean_train$Age)

test$Age <- as.numeric(test$Age)



#Turning Sex into a factor

clean_train$Sex <- as.factor(clean_train$Sex)

test$Sex <- as.factor(test$Sex)



#Turning Embarked into a factor.

clean_train$Embarked <- as.factor(clean_train$Embarked)

test$Embarked <- as.factor(test$Embarked)



#Check

str(clean_train)

str(test)
#Generating model

tree <- rpart(Survived ~ Sex + Pclass + Age + SibSp + Parch + Fare + Embarked, clean_train, method = "class")



#Making predictions on the test data

pred_dt = predict(tree, test, type = "class")
output = data.frame(PassengerId = test$PassengerId, Survived = pred_dt)

head(output, n = 20)
write.csv(output, file = "my_solution_dt.csv", row.names=FALSE)
survived <- clean_train$Survived

knn_train <- clean_train

knn_test <- test



# Normalize Pclass

min_class <- min(knn_train$Pclass)

max_class <- max(knn_train$Pclass)

knn_train$Pclass <- (knn_train$Pclass - min_class) / (max_class - min_class)

knn_test$Pclass <- (knn_test$Pclass - min_class) / (max_class - min_class)



# Normalize Age

min_age <- min(knn_train$Age)

max_age <- max(knn_train$Age)

knn_train$Age <- (knn_train$Age - min_age) / (max_age - min_age)

knn_test$Age <- (knn_test$Age - min_age) / (max_age - min_age)



# Normalize Fare

min_fare <- min(knn_train$Fare)

max_fare <- max(knn_train$Fare)

knn_train$Fare <- (knn_train$Fare - min_fare) / (max_fare - min_fare)

knn_test$Fare <- (knn_test$Fare - min_fare) / (max_fare - min_fare)
#Dropping tables in knn_train

knn_train$PassengerId = NULL

knn_train$Survived = NULL

knn_train$Name = NULL

knn_train$Ticket = NULL

knn_train$Cabin = NULL

knn_train$Sex = NULL

knn_train$Embarked = NULL



#Dropping tables in knn_test

knn_test$PassengerId = NULL

knn_test$Name = NULL

knn_test$Ticket = NULL

knn_test$Cabin = NULL

knn_test$Sex = NULL

knn_test$Embarked = NULL

for (i in seq_along(knn_test$Age)) {

  if (is.na(knn_test$Age[i])){

      knn_test$Age[i] <- mean(knn_test$Age, na.rm = TRUE)

      }

}



for (i in seq_along(knn_test$Fare)) {

  if (is.na(knn_test$Fare[i])){

      knn_test$Fare[i] <- mean(knn_test$Fare, na.rm = TRUE)

      }

}
pred_knn <- knn(train = knn_train, test = knn_test, cl = survived, k = 5)
output_knn = data.frame(PassengerId = test$PassengerId, Survived = pred_knn)

head(output_knn, n = 20)



write.csv(output_knn, file = "my_solution_knn.csv", row.names=FALSE)