#===========================================================#

# Program name: simple_titanic_ml

# Program purpose: simple ML models for predicting survival

#===========================================================#



library(dplyr)

library(caret)



# training data

train <- read.csv('../input/train.csv')

names(train) <- tolower(names(train))

print(head(train))



# create dummy features for pclass

pclass.vars <- dummyVars("~ pclass", data = train, fullRank = T)

print(class(pclass.vars))



# split training dataset into further train and test set; 

# use new training set for k-fold ridge regression

train_newIndex <- createDataPartition(train$survived, p = 0.8, list = F)



train_new <- train[train_newIndex,]

test <- train[-train_newIndex]



# run K-fold cross validation w/o lambda parameter (aka simple regression)


