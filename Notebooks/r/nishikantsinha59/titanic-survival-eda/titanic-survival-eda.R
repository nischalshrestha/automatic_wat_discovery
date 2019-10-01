# Disable warnings

options(warn=-1)



# Load Packages

library(tidyverse)     # collection of libraries like readr, dplyr, ggplot

library(rpart)         # classification algorithm

library(mice)          # imputation

library(scales)        # Visualization

library(ggthemes)      # Visualization

library(class)         # classification algorithm

library(e1071)         # classification algorithm

library(randomForest)  # classification algorithm

library(party)         # classification algorithm
# Read datasets

train <- read_csv("../input/train.csv")  # Load train data

test <- read_csv("../input/test.csv")    # Load test data

full <- bind_rows(train,test)     # combine training and test dataset 
head(full)  # View few records of loaded data
# Check Structure of data set

str(full)
# Apply sum(is.na()) to each variable to check missing values

sapply(full, function(x) sum(is.na(x)))

# sapply() is used to apply a function to each element of list, dataframes or vectors.
# Find which observation has missing fare value

which(is.na(full$Fare)==TRUE)   # which() will return indexes of the observation which has missing value for Fare

# 1044 is the index of missing fare value



full[1044,]   # View complete information of obbseravation which has missing Fare value
# Set the height and weight of the plot

options(repr.plot.width=6, repr.plot.height=4)



# Plot Fare distrfibution of class 3 passenger who embarked from port S

ggplot(full[full$Pclass == '3' & full$Embarked == 'S',], aes(x = Fare)) +   

geom_density(fill = 'royalblue', alpha ='0.7') +

geom_vline(aes(xintercept=median(Fare, na.rm=T)), colour='red', linetype='dashed', lwd=1) +

scale_x_continuous(labels=dollar) +

theme_few()
# Replace missing fare with median fare for class and embarkment.

full$Fare[1044] <- median(full[full$Pclass == 3 & full$Embarked == 'S', ]$Fare,na.rm = TRUE)
# Find which observation has missing Embarkation value

which(is.na(full$Embarked)==TRUE)

# Passengers with IDs 62 and 830 has missing embarkation value



# View complete information of passengers with ID 62 and 830

full[c(62,830),]
# Before visualization get rid of passenger Ids 62 and 830 having missing embarkment port

plot_data <- full[-c(62,830),]



# Use ggplot2 to visualize embarkment, passenger class, & median fare

ggplot(plot_data, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +

geom_boxplot() +

geom_hline(aes(yintercept=80), colour='red', linetype='dashed', lwd=2) +

scale_y_continuous(labels=dollar) +

theme_few()
# Replace NAs in Embarked with 'C'

full$Embarked[c(62,830)] <- 'C'
# Convert categorical variable into factors

factor_vars <- c('Pclass','Sex','Embarked')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))
# Set a random seed

set.seed(129)



# Build rpart model for age imputation 

age_pred <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked,

                  data = full[!is.na(full$Age),], method = "anova") 



# Use the rpart model to predict the missing age values

imputed_age <- predict(age_pred, full[is.na(full$Age),])

rpart_imputation <- full

missing_age_indexes <- which(is.na(full$Age)==TRUE)

rpart_imputation$Age[missing_age_indexes] <- imputed_age
# Perform mice imputation, excluding certain less-than-useful variables:

mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Survived')],

                 method='rf') 



# Save the complete output 

mice_output <- complete(mice_mod)
# Plot age distributions

par(mfrow=c(1,3))

hist(full$Age, freq=F, xlab ='Passengers Age',main='Age: Original Data', 

  col='turquoise4', ylim=c(0,0.06))

lines(density(full$Age,na.rm = TRUE), col="red2", lwd=1.5)



hist(rpart_imputation$Age, freq=F, xlab ='Passengers Age', main='Age: Rpart Output', 

  col='turquoise3', ylim=c(0,0.06))

lines(density(rpart_imputation$Age), col="red2", lwd=1.5)



hist(mice_output$Age, freq=F, xlab ='Passengers Age', main='Age: MICE Output', 

  col='turquoise1', ylim=c(0,0.06))

lines(density(mice_output$Age), col="red2", lwd=1.5)
# Replace Age variable from the mice model.

full$Age <- mice_output$Age



# Check if any missing Age values got replaced 

sum(is.na(full$Age))
# Extract title from passengers name

full$Title <- gsub('(.*, )|(\\..*)', '', full$Name) 

# gsub() function replaces all matches of a string, if the parameter is a string vector, returns a string 

# vector of the same length and with the same attributes



# Show title counts by sex

table(full$Sex, full$Title)
# Titles with very low cell counts to be combined to "rare" level

rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')



# Also reassign mlle, ms, and mme accordingly

full$Title[full$Title == 'Mlle']        <- 'Miss' 

full$Title[full$Title == 'Ms']          <- 'Miss'

full$Title[full$Title == 'Mme']         <- 'Mrs' 

full$Title[full$Title %in% rare_title]  <- 'Rare Title'



# Show title counts by sex again

table(full$Sex, full$Title)
# Extract surname from name variable

full$Surname <- sapply(full$Name, function(x) {strsplit(x, split = '[,.]')[[1]][1]})



# Check the number of unique Surnames

cat(paste('We have ', nlevels(factor(full$Surname)), ' unique surnames.'))
# Add familySize variable to our dataset

full$familySize <- full$SibSp + full$Parch + 1
# Combine familySize and Surname to make familyID 

full$familyID <- paste(as.character(full$familySize), full$Surname, sep = "")



#full$familyID[full$familySize == 1] <- 'Singleton'

full$familyID[full$familySize <= 2] <- 'Small'



# View the count of each category of familyID

table(full$familyID)
# Create a data frame having count of each category

fmlyIDs <- data.frame(table(full$familyID))



# Get the familyID which have count less than 3

smallFamId <- fmlyIDs[fmlyIDs$Freq <=2,]



# Replace the familyID which have count less than  3 with 'Small'

full$familyID[full$familyID %in% smallFamId$Var1] <- 'Small'



# Again check the count of each category of familyID

table(full$familyID)
# Use ggplot2 to visualize the relationship between family size & survival

ggplot(full[1:891,], aes(x = familySize, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  scale_x_continuous(breaks=c(1:11)) +

  labs(x = 'Family Size') +

  theme_few()
# Discretize family size

full$familySize[full$familySize > 4] <- 'large'

full$familySize[full$familySize == 1] <- 'singleton'

full$familySize[full$familySize < 5 & full$familySize > 1] <- 'small'



# Show family size by survival using a mosaic plot

mosaicplot(table(full$familySize, full$Survived), main='Family Size by Survival', shade=TRUE)
# First we'll look at the relationship between age, sex & survival

ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 

geom_histogram() + 

facet_grid(.~Sex) + 

theme_few()
# Copy the familyID to new variable familyID2

full$familyID2 <- full$familyID



# Covert to string

full$familyID2 <- as.character(full$familyID2)



# Create data frame having count of each familyID

fmlyIDs <- data.frame(table(full$familyID2))



# find the family ID having count equal to 3

smallFamId <- fmlyIDs[fmlyIDs$Freq ==3,]



# Replace the familyID having count 3 with 'Small'

full$familyID2[full$familyID %in% smallFamId$Var1] <- 'Small'



# Check the familyID2 variable

table(full$familyID2)
# First create the child variable and indicate whether child or adult

full$Child <- 'Adult'

full$Child[full$Age < 18] <- 'Child'



# Show the count 

table(full$Child, full$Survived)



# Use ggplot2 to visualize the relationship between being a child & survival

ggplot(as.data.frame(full[1:891,]), aes(x = Child, fill = factor(Survived))) +

geom_bar(position = "dodge") +

labs(x = 'Child or Adult') +

theme_few()
# Adding Mother variable 

full$Mother <- 'Not Mother'

full$Mother[full$Sex == 'female'  & full$Parch >0 & full$Age >= 18 & full$Title != 'Miss'] <- 'Mother'



# Show the count of mothers survival

table(full$Mother, full$Survived)



# Use ggplot2 to visualize the relationship between being a child & survival

ggplot(as.data.frame(full[1:891,]), aes(x = Mother, fill = factor(Survived))) +

  geom_bar(position = "dodge") +

  labs(x = '') +

  theme_few()
# Scaling of continuos variable

full[,c(6,10)] <- scale(full[,c(6,10)])
# Finish by factorizing our categorical variables

full$Title <- factor(full$Title)

full$Surname <- factor(full$Surname)

full$familySize <- factor(full$familySize)

full$familyID <- factor(full$familyID)

full$familyID2 <- factor(full$familyID2)

full$Child  <- factor(full$Child)

full$Mother <- factor(full$Mother)
train <- full[1:891,]

test <- full[892:1309,]
# build the model using all the important feature

glm_model <- glm(data = train, formula = Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked 

                 + Title + familySize + familyID2 + Child + Mother,

                 family = binomial(link = "logit"))



# Print annova (analysis of variance) table, this will give you variance and df of each vriable

anova(glm_model)



# Check the summary of the model, it will have residual and null diviance by which we can calculate pseudo R^2 

#summary(glm_model)



# Calculate pseudo R^2

psr <- 1 - (664.96/1186.7) # psr = 1 - (residual deviance / null deviance)



# Print  the value of pseudo R-square 

print(psr)

#This Logistic Regression model has Psuedo R-square of 0.4396562

# Create confusion matrix with train data set to check its accuracy in trainig environment

Prediction <- predict(glm_model, train[,-2], type = "response")

confusion_matrix_train <- table(train$Survived, (Prediction > 0.5))  

#  > 0.5 signifies that whatever comes greater than .5 will be considered 1 in confusion matrix



print(confusion_matrix_train)



#                    (Predicted Values)     

#                       FALSE TRUE

#  (Actual Values)   0   488   61

#                    1    82  260



# Calculate Precision and Recall based on the confusion matrix 

recall <- 260/(260+82)     # recall = True Positive(TP) / Actual Yes

# 0.7602339



precision <- 260/(260+61)   # precision = TP / Predicited Yes

# 0.8099688



# calculate the f1 score (measure of accuracy)

f1 <- (2*recall*precision)/(recall+precision)

print(paste0("Accuracy : ",f1))

#  accuracy 0.784314
# Do same prediction on test data set

Prediction <- predict(glm_model, test[,-2], type = "response")

Prediction <- as.integer(Prediction > 0.5)



# Create data frame having 2 columns PassengerId and Survived as in sample submission of this competition

submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)



# Write the data frame to a .csv file(in Rstudio)  and submit the file to kaggle 

# write.csv(submit, file = "glm_model.csv", row.names = FALSE)



# Kaggle score 0.77990
# Create Naive Bayes model using naiveBayes function

nb_model <- naiveBayes(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked 

                       + Title + familySize + familyID2 + Child + Mother,

                       data = train)

# Check summary of model

summary(nb_model)
#Creating confusion matrix for the model

train_pred <- predict(nb_model, train)

confusion_matrix_train <- table(train$Survived, train_pred)



print(confusion_matrix_train)

#                    (Predicted Values)     

#                       FALSE TRUE

#  (Actual Values)   0   470   79

#                    1    86  256





#Calculating Precision and recall

recall <- 256/(256+86)     # recall = TP / actual yes

# 0.748538



precision <- 256/(256+79)   # precision = TP / predicted yes

# 0.7641791



f1 <- (2*recall*precision)/(recall+precision)

print(paste0("Accuracy : ",f1))

# 0.7562777
#Do same prediction on test data set

test_pred <- predict(nb_model, test)

submit <- data.frame(PassengerId = test$PassengerId, Survived = test_pred)



# Kaggle score 0.76076
# Build model using function svm()

sv_model <- svm(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title 

                + familySize + familyID2 + Child + Mother,

                train[,-c(11)])

# Note always take care of the variable which have missing values for svm because if any row is having 

# missing feature then it will not give output for that particular observation

# That's why I have excluded the Cabin variable which is having missing values



# Check the information like gamma value, number of support vectors of the model built 

print(sv_model)
#Creating confusion matrix for the model

train_pred <- predict(sv_model, train[,-c(11)])

confusion_matrix_train <- table(train$Survived, train_pred)



print(confusion_matrix_train)

#                    (Predicted Values)     

#                       FALSE TRUE

#  (Actual Values)   0   491   58

#                    1    90  252



#Calculating Precision and recall

recall <- 252/(252+90)     # recall = TP / actual yes

# 0.7368421



precision <- 252/(252+58)   # precision = TP / predicted yes

# 0.8129032



f1 <- (2*recall*precision)/(recall+precision)

print(paste0("Accuracy : ",f1))

# 0.7730061
#Do same prediction on test data set

test_pred <- predict(sv_model, test[,-c(2,11)])

submit <- data.frame(PassengerId = test$PassengerId, Survived = test_pred)



# Kaggle score 0.78947
# Copy the survived label to a new variable

train_survived_labels <- train$Survived



# Convert labels of all categorical variable from characters to numeric codes

# make a new dataset and copy only the required features and then proceed with conversion

knn_train <- train[,-c(1,2,4,9,11,14,16)]

knn_train$Sex <- factor(knn_train$Sex, labels = c('0','1'))

knn_train$Embarked <- factor(knn_train$Embarked, labels = c('1','2','3'))

knn_train$Title <- factor(knn_train$Title, labels = c(1:5))

knn_train$familySize <- factor(knn_train$familySize, labels = c(1:3))

knn_train$familyID2 <- factor(knn_train$familyID2, labels = c(1:24))

knn_train$Child <- factor(knn_train$Child, labels = c('0','1'))

knn_train$Mother <- factor(knn_train$Mother, labels = c('0','1'))



# Repeat the above steps for test dataset as well 

knn_test <- test[,-c(1,2,4,9,11,14,16)]

knn_test$Sex <- factor(knn_test$Sex, labels = c('0','1'))

knn_test$Embarked <- factor(knn_test$Embarked, labels = c('1','2','3'))

knn_test$Title <- factor(knn_test$Title, labels = c(1:5))

knn_test$familySize <- factor(knn_test$familySize, labels = c(1:3))

knn_test$familyID2 <- factor(knn_test$familyID2, labels = c(1:20))

knn_test$Child <- factor(knn_test$Child, labels = c('0','1'))

knn_test$Mother <- factor(knn_test$Mother, labels = c('0','1'))
# Create the model using knn() function

knn_model <- knn(train = knn_train, test = knn_test, cl = train_survived_labels, k = 37)



# Write the result to .csv file 

submit <- data.frame(PassengerId = test$PassengerId, Survived = knn_model)



# Kaggle score 0.71291
# set seed

set.seed(415)



# Create model using rpart() function

dt_model <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + familySize 

             + familyID + Child + Mother,

      data = train,

     method = "class")
#Creating confusion matrix for the model

train_pred <- predict(dt_model, train, type = "class")

confusion_matrix_train <- table(train$Survived, train_pred)



print(confusion_matrix_train)

#                    (Predicted Values)     

#                       FALSE TRUE

#  (Actual Values)   0   502   47

#                    1    91  251



#Calculating Precision and recall

recall <- 251/(251+91)     # recall = TP / actual yes

# 0.7426901



precision <- 251/(251+47)   # precision = TP / predicted yes

# 0.8141026



f1 <- (2*recall*precision)/(recall+precision)

print(paste0("Accuracy : ",f1))

# 0.784375
Prediction <- predict(dt_model, test, type = "class")

submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)



# Kaggle Score 0.79904
# set seed

set.seed(415)



# Build the model (note: not all possible variables are used)

rf_model <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked 

                         + Title + familySize + familyID2 + Child + Mother,

                    data=train,

                    importance=TRUE,

                    ntree=2000)



# So let’s look at what variables were important:

varImpPlot(rf_model)
#Creating confusion matrix for the model

train_pred <- predict(rf_model, train)

confusion_matrix_train <- table(train$Survived, train_pred)



print(confusion_matrix_train)

#                    (Predicted Values)     

#                       FALSE TRUE

#  (Actual Values)   0   532   17

#                    1    62  280



#Calculating Precision and recall

recall <- 280/(280+62)     # recall = TP / actual yes

# 0.8187134



precision <- 280/(280+17)   # precision = TP / predicted yes

# 0.9427609



f1 <- (2*recall*precision)/(recall+precision)

print(paste0("Accuracy : ",f1))

# 0.876369327073553
Prediction <- predict(rf_model, test, type = "class")

submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)



# Kaggle Score 0.77511
# We again set the seed for consistent results and build a model in a similar way to our Random Forest

set.seed(415)



cf_model <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title 

                    + familySize + familyID + Child + Mother,

                    data = train,

                    controls=cforest_unbiased(ntree=2000, mtry=3))
#Creating confusion matrix for the model

train_pred <- predict(cf_model, train, OOB=TRUE, type = "response")

confusion_matrix_train <- table(train$Survived, train_pred)



print(confusion_matrix_train)

#                    (Predicted Values)     

#                       FALSE TRUE

#  (Actual Values)   0   508   41

#                    1    93  249



#Calculating Precision and recall

recall <- 249/(249+93)     # recall = TP / actual yes

# 0.7192982



precision <- 249/(249+41)   # precision = TP / predicted yes

# 0.8692579



f1 <- (2*recall*precision)/(recall+precision)

print(paste0("Accuracy : ",f1))

# 0.7879746
Prediction <- predict(cf_model, test, OOB=TRUE, type = "response")

submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)



#Kaggle score 0.80861