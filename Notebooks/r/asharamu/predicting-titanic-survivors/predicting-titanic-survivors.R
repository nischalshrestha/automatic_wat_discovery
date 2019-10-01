

#######################--My First Kaggle Competition---######################



#**************************** Sinking of Titanic ****************************



# -------------------------  PART 1 -------------------------------------------------



########################## TRAINING A MODEL - xgboost ###############################



#----------------- Read the train file -----------------------------



#  this file will be split into training and testing and for building the classifier

database <- read.csv("train.csv", na.strings = c(""))

database <- database[c(3,5,6,7,8,10,12,2)] # removed passenger is, name, ticket and cabin number

head(database)

str(database)



#------------------------Data cleaning and formatting



# dealing with missing values

#------ replace missing age by mean

database$Age[is.na(database$Age)] <- round(mean(database$Age, na.rm = TRUE))



#-------------- remove two records who are missing Embarked(only two wont impact)

database <- na.omit(database)



#  Converting categorical values into numeric

database$Embarked = as.numeric(factor(database$Embarked,

                                       levels = c("C", "Q", "S"),

                                       labels = c(1,2,3))

)



database$Sex = as.numeric(factor(database$Sex,

                                 levels = c("female","male"),

                                 labels = c(1,2))

)



#------------- create training and testing data

library(caTools)

split <- sample.split(database$Survived, SplitRatio = 0.8)

training <- subset(database, split==TRUE)

testing <- subset(database, split==FALSE)



# -------------------Fitting XGBoost to the Training set

#install.packages('xgboost')

library(xgboost)

classifier = xgboost(data = as.matrix(training[-8]), label = training$Survived, nrounds = 10)



# Predicting the Test set results

y_pred = predict(classifier, newdata = as.matrix(testing[-8]))

y_pred = (y_pred >= 0.5)



# Making the Confusion Matrix

cm = table(testing[, 8], y_pred)

cm



# -------------------------  PART 2 -------------------------------------------------



########################## PREDICTING THE SURVIVORS - xgboost ###############################



# this file will be ran against the created model to predict the outcome

database_test <- read.csv("test.csv", na.strings = c(""))

database_output <- read.csv("test.csv", na.strings = c(""))

database_test <- database_test[c(2,4,5,6,7,9,11)] 



##dealing with missing values

#------ replace missing age by mean

database_test$Age[is.na(database_test$Age)] <- round(mean(database_test$Age, na.rm = TRUE))



#-------------- remove two records who are missing Embarked(only two wont impact)

#database_test <- na.omit(database_test)



#  Converting categorical values into numeric

database_test$Embarked = as.numeric(factor(database_test$Embarked,

                                      levels = c("C", "Q", "S"),

                                      labels = c(1,2,3))

)



database_test$Sex = as.numeric(factor(database_test$Sex,

                                 levels = c("female","male"),

                                 labels = c(1,2))

)



# --- Fun time - Predicting the outcome



y_pred = predict(classifier, newdata = as.matrix(database_test))

y_pred = ifelse(y_pred >= 0.5,1,0)

y_pred



#------------Add this predicted outcome to test data after renaming it as Survived



database_output$Survived <- y_pred

head(database_output)

database_output <- database_output[c(1,12)]



write.csv(database_output, file = "D:/Practice R and machine learning/Titanic Comp on kaggle/gender_submission.csv")
