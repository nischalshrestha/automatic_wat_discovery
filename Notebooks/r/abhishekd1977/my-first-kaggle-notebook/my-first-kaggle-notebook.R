# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



# Load packages

library(rpart)

library(tree)

library('ggplot2') # visualization

library('ggthemes') # visualization

library('scales') # visualization

library('dplyr') # data manipulation

library('mice') # imputation

library('randomForest') # classification algorithm

library('e1071') #Naive Bayes classification

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
train <- read.csv("../input/train.csv", header = TRUE,  stringsAsFactors = FALSE)

test <- read.csv("../input/test.csv", header = TRUE,  stringsAsFactors = FALSE)



###Add Family Size to dataset



train$FamilySize <- train$SibSp + train$Parch + 1

test$FamilySize <- test$SibSp + test$Parch + 1





### Make variables factors into factors

factor_variables <- c('PassengerId','Pclass','Sex','Embarked', 'FamilySize')



train[factor_variables] <- lapply(train[factor_variables], function(x) as.factor(x))

train$Survived = as.factor(train$Survived)



test[factor_variables] <- lapply(test[factor_variables], function(x) as.factor(x))



### Show number of missing Age values

sum(is.na(train$Age))

sum(is.na(test$Age))



### Perform predictive imputation on Age

mice_mod_train <- mice(train[, !names(train) %in% c('PassengerId','Name','Ticket','Cabin','Family','Survived')], method='rf', printFlag = FALSE)

mice_mod_test <- mice(test[, !names(test) %in% c('PassengerId','Name','Ticket','Cabin','Family')], method='rf', printFlag = FALSE)

# Save the complete output 

mice_output_train <- complete(mice_mod_train)

mice_output_test <- complete(mice_mod_test)



### Replace Age variable in training and test datasets from the predictive imputaion

train$Age <- mice_output_train$Age

test$Age <- mice_output_test$Age



# Show new number of missing Age values

sum(is.na(train$Age))

sum(is.na(test$Age))



# Set a random seed

set.seed(754)



###Visualize survival with respect to number of family members

ggplot(train[1:891,], aes(x = FamilySize, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  labs(x = 'Family Size') +

  theme_few()



## Build the model#1: Random Forest - START

rf_model <- randomForest(Survived ~ Pclass + Sex + SibSp + Parch + FamilySize + Age, data = train)



### Show model error

plot(rf_model, ylim=c(0,0.36))

legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)



###Determine which variable is impacting the model most

# Get importance

importance    <- importance(rf_model)

varImportance <- data.frame(Variables = row.names(importance), 

                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance

rankImportance <- varImportance %>%

  mutate(Rank = paste0('#',dense_rank(desc(Importance))))



# Use ggplot2 to visualize the relative importance of variables

ggplot(rankImportance, aes(x = reorder(Variables, Importance), 

    y = Importance, fill = Importance)) +

  geom_bar(stat='identity') + 

  geom_text(aes(x = Variables, y = 0.5, label = Rank),

    hjust=0, vjust=0.55, size = 4, colour = 'red') +

  labs(x = 'Variables') +

  coord_flip() + 

  theme_few()



### Predict using the test set. Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)

predict.randomForest <- predict(rf_model, test)

solution.randomForest <- data.frame(PassengerID = test$PassengerId, Survived = predict.randomForest)



## Build the model#1: Random Forest - END



## Build the model#2: Naive Bayes classification - START

naiveBayesModel <- naiveBayes(Survived ~ Pclass + Sex + SibSp + Parch + FamilySize + Age, data = train)

summary(naiveBayesModel)

predict.NaiveBayes <- predict(naiveBayesModel,test[,-1])



solution.naiveBayes <- data.frame(PassengerID = test$PassengerId, Survived = predict.NaiveBayes)



## Build the model#2: Naive Bayes classification - END