library(rpart)

library(dplyr)



# Load in the packages to build a fancy plot

library(rattle)

library(rpart.plot)

library(RColorBrewer)



library(randomForest)
train <- read.csv("../input/train.csv")

test <- read.csv("../input/test.csv")


str(train)
table(train$Survived)
prop.table(table(train$Survived))
table(train$Sex, train$Survived)
prop.table(table(train$Sex,train$Survived), 1)
train$Child <- NA

train$Child[train$Age < 18] <- 1

train$Child[train$Age >= 18] <- 0
prop.table(table(train$Child, train$Survived), 1)
#make a copy of test

test_one <- test

# Initialize a Survived column to 0

test_one$Survived <- 0

# Set Survived to 1 if Sex equals "female"

test_one$Survived[test_one$Sex == "female"] <- 1

# Copy of test

test_one <- test



# Initialize a Survived column to 0

test_one$Survived <- 0



# Set Survived to 1 if Sex equals "female"

test_one$Survived[test_one$Sex == "female"] <- 1

my_tree <- rpart(Survived ~ Sex + Age,

                 data = train,

                 method ="class")
my_tree_two <- rpart(Survived ~ 

                     Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 

                     data = train, 

                     method = "class")
plot(my_tree)

text(my_tree)
plot(my_tree_two)

text(my_tree_two)
fancyRpartPlot(my_tree)
fancyRpartPlot(my_tree_two)
# Make predictions on the test set

my_prediction <- predict(my_tree_two, newdata = test, type = "class")

str(my_prediction)

# Finish the data.frame() call

my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)



# Use nrow() on my_solution

nrow(my_solution)
#modify the split

my_tree_three <- rpart(Survived ~ 

                       Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,

                       data = train, 

                       method = "class", 

                       control = rpart.control(minsplit = 50, cp = 0))



# Visualize my_tree_three

fancyRpartPlot(my_tree_three)
# Create train_two

train_two <- train

train_two$family_size <- train_two$SibSp + train_two$Parch + 1



# Finish the command

my_tree_four <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + family_size,

                      data = train_two, method = "class")



# Visualize your new decision tree

fancyRpartPlot(my_tree_four)
str(train)
train_two$family_size <- train_two$SibSp + train_two$Parch + 1



# Passenger on row 62 and 830 do not have a value for embarkment.

# Since many passengers embarked at Southampton, we give them the value S.

train$Embarked[c(62, 830)] <- "S"



# Factorize embarkment codes.

train$Embarked <- factor(train$Embarked)



# Passenger on row 1044(when combining train and test) has an NA Fare value. 

#Let's replace it with the median fare value.

test[153,]

test$Fare[153] <- median(train$Fare, na.rm = TRUE)

test[153,]

train$family_size <- train$SibSp + train$Parch + 1

test$family_size <- test$SibSp + test$Parch + 1



# How to fill in missing Age values?

# We make a prediction of a passengers Age using the other variables and a decision tree model.

# This time you give method = "anova" since you are predicting a continuous variable.



predicted_train_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + family_size,

                       data = train[!is.na(train$Age),], method = "anova")

predicted_test_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + family_size,

                       data = train[!is.na(test$Age),], method = "anova")
train$Age[is.na(train$Age)] <- predict(predicted_train_age, train[is.na(train$Age),])





test$Age[is.na(test$Age)] <- predict(predicted_test_age, train[is.na(test$Age),])

# Set seed for reproducibility

set.seed(111)

# Apply the Random Forest Algorithm

my_forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,

                          data = train, importance = TRUE, ntree = 100)



# Make your prediction using the test set

my_prediction <- predict(my_forest, test)



# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions

my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)



varImpPlot(my_forest)