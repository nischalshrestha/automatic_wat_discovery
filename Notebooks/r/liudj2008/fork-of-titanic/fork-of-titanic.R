# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
#Load train data

train=read.csv('../input/train.csv')





train=train[,-c(1,4,9,11)]



#Fill the empty values

set.seed(1)

train$Age[is.na(train$Age)]=sample(train$Age[!is.na(train$Age)],177)



set.seed(2)

train$Embarked[train$Embarked=='']=sample(train$Embarked[train$Embarked!=''],2)

train$Embarked=as.character(train$Embarked)

train$Embarked=as.factor(train$Embarked)



train$Survived=as.factor(train$Survived)



summary(train)



# Modeling

library(randomForest)

set.seed(3)

rf.fit=randomForest(Survived~., data=train)

rf.pred=predict(rf.fit, newdata = train)



# Train error

table(true=train$Survived, pred=rf.pred)

mean(train$Survived==rf.pred)

# Load test sample

test_original=read.csv('../input/test.csv')



test=test_original



test=test[,-c(1,3,8,10)]

#summary(test)



# Initiate the survived factor and fill in the NAs

test=data.frame(Survived=as.factor(rep(0,418)), test)

set.seed(1)

test$Age[is.na(test$Age)]=sample(test$Age[!is.na(test$Age)],86)



set.seed(2)

test$Fare[is.na(test$Fare)]=sample(test$Fare[!is.na(test$Fare)],1)



# Making levels between training and testing data set is the same

names(test)

levels(test$Survived)=levels(train$Survived)

levels(test$Embarked)=levels(train$Embarked)



# Make prediction

test$Survived=predict(rf.fit, test)
# Export files

prediction=data.frame(PassengerId=test_original$PassengerId, Survived=test$Survived)





summary(prediction)

head(prediction)

# Output excel

write.csv(prediction,'gender_submission.csv')