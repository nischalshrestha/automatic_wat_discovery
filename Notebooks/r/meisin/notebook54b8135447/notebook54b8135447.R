# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(ggthemes) # visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(Amelia) # for visualizing missing data

library(caret) # for using confusion matrix

library(ROCR) # for generating ROC curves

library(randomForest) # for generating Random Forest

library(stringr) # for manipulating string columns

library(rpart) # for generating Decision Trees

library(rpart.plot) # for visualizing Decision Trees

library(e1071) # for generating Support Vector Machines
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



### READING DATA ###

train <- read.csv("../input/train.csv", na.strings=c("NA", "")) ##important code to treat blank string as NA 

test <- read.csv("../input/test.csv", na.strings=c("NA", ""))



# Any results you write to the current directory are saved as output.
train$Survived = factor(train$Survived)

train$Pclass = factor(train$Pclass)

test$Pclass = factor(test$Pclass)

str(train)
percentage_missing_data <- sapply(train, function(df) {sum(is.na(df)==TRUE)/ length(df);})

percentage_missing_data

# results : return a percentage of missing value
missmap(train, main="Missing Map")
table(train$Embarked, useNA = "always")
train$Embarked[which(is.na(train$Embarked))] = 'S'
train$Name = as.character(train$Name)

table_words = table(unlist(strsplit(train$Name, "\\s+")))

sort(table_words [grep('\\.',names(table_words))], decreasing=TRUE)
train$Title <- str_match(train$Name, "[a-zA-Z]+\\.")

table(train$Title[which(is.na(train$Age))])
mean.mr <- mean(train$Age[train$Title == "Mr." & !is.na(train$Age)])

mean.dr <- mean(train$Age[train$Title == "Dr." & !is.na(train$Age)])

mean.miss <- mean(train$Age[train$Title == "Miss." & !is.na(train$Age)])

mean.master <- mean(train$Age[train$Title == "Master." & !is.na(train$Age)])

mean.mrs <- mean(train$Age[train$Title == "Mrs." & !is.na(train$Age)])



# assigning mean to rows missing Age value

train$Age[train$Title == "Mr." & is.na(train$Age)] <- mean.mr

train$Age[train$Title == "Dr." & is.na(train$Age)] <- mean.dr

train$Age[train$Title == "Miss." & is.na(train$Age)] <- mean.miss

train$Age[train$Title == "Master." & is.na(train$Age)] <- mean.master

train$Age[train$Title == "Mrs." & is.na(train$Age)] <- mean.mrs
## Training set has 891 rows, split by the following:

## New_Train = 623

## Validation = 268

new_train <- train[1:623,]

validation <- train[624:nrow(train),]
my_decision_tree <- rpart(Survived ~ Age + Sex + Pclass  + 

                          SibSp + Fare + Parch + Embarked, 

                          data = new_train, method = "class", control=rpart.control(cp=0.0001))

summary(my_decision_tree)

prp(my_decision_tree, type = 4, extra = 106)
my_dt_prediction <- predict(my_decision_tree, validation, type = "class")



# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)

dt_solution <- data.frame(PassengerID = validation$PassengerId, Survived = my_dt_prediction)



#Show model error

##!!plot(my_decision_tree, ylim=c(0,0.36))

##!!legend('topright', colnames(my_decision_tree$err.rate), col=1:3, fill=1:3)
my_svm = svm(Survived ~ Pclass + Sex + Age + SibSp + Fare + Parch + Embarked,

             data = train, probability = TRUE)

my_svm_prediction <- predict(my_svm, validation)
# Set a random seed

set.seed(754)



# Build the model (note: not all possible variables are used)

my_rf <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Fare + Parch + Embarked,

                                            data = train)

# Predict using the test set

my_rf_prediction <- predict(my_rf, validation)
class(my_dt_prediction)
confusionMatrix(my_dt_prediction, validation$Survived)

##Generating ROC curve 

my_dt.prob <- predict(my_decision_tree, validation, type = "prob")

#assessing performance with the use of ROC Curve

my_dt.scores <- prediction(my_dt.prob[,2], validation$Survived)

my_dt.perf <- performance(my_dt.scores, measure="tpr", x.measure="fpr")
confusionMatrix(my_rf_prediction, validation$Survived)



my_rf.prob <- predict(my_rf, validation, type = "prob")

#assessing performance with the use of ROC Curve

my_rf.scores <- prediction(my_rf.prob[,2], validation$Survived)

my_rf.perf <- performance(my_rf.scores, measure="tpr", x.measure="fpr")
confusionMatrix(my_svm_prediction, validation$Survived)



my_svm.prob <- predict(my_svm, validation, type = "prob")

#assessing performance with the use of ROC Curve

#my_svm.scores <- prediction(my_svm.prob[,2], validation$Survived)

#my_svm.perf <- performance(my_svm.scores, measure="tpr", x.measure="fpr")
# Plot the ROC curve

plot(my_dt.perf, col = "green", lwd = 1.5)

# Add the ROC curve of the logistic model and the diagonal line

plot(my_rf.perf, col = "red", lwd = 1, add = TRUE)

abline(0, 1, lty = 8, col = "grey")

legend("bottomright", legend = c("tree", "forest"), col = c("green", "red"), lwd = c(1.5,1))
#### more performance comparison codes

# AUC for the Decision Tree

my_dt.auc <- performance(my_dt.scores, "auc")      # AUC for the decision tree

my_rf.auc <- performance(my_rf.scores, "auc")



my_dt.auc@y.values

my_rf.auc@y.values
importance    <- importance(my_rf)
varImportance <- data.frame(Variables = row.names(importance), Importance = round(importance[ ,'MeanDecreaseGini'],2))



## sort by Importance

new_varImportance <- varImportance[order(-Importance),]



# Use ggplot2 to visualize the relative importance of variables

ggplot(new_varImportance, aes(x = reorder(Variables, Importance), 

    y = Importance, fill = Importance)) +

  geom_bar(stat='identity') + 

  labs(x = 'Variables') +

  coord_flip() + 

  theme_few()
importance = varImp(my_decision_tree, scale=FALSE)

importance

plot(importance)