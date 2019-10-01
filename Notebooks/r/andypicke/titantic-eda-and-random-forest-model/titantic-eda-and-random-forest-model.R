# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

suppressPackageStartupMessages(library(dplyr))

suppressPackageStartupMessages(library(caret))



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")
#Load the data look at it's structure:

train_data <- read.csv("../input/train.csv",stringsAsFactors=FALSE)

test_data <- read.csv("../input/test.csv",stringsAsFactors=FALSE)

#head(train_data)

#head(test_data)

full_data <- bind_rows(train_data,test_data)

head(full_data)

tail(full_data)
head(train_data)
head(test_data)
str(full_data)
#full_data$Survived <- factor(full_data$Survived,levels=c("no","yes"),labels=c(0,1))

full_data$Survived <- as.factor(full_data$Survived)

levels(full_data$Survived) <- make.names(levels(factor(full_data$Survived)))

#full_data$Survived <- factor(full_data$Survived,levels=c('no','yes'))

full_data$Pclass <- as.factor(full_data$Pclass)

full_data$Embarked <-as.factor(full_data$Embarked)

str(full_data)
#First, let's see how many passenegers did/didn't survive:

table(full_data$Survived)
#Is the average fare different for those that survived? Yes, the average fare for those that survived is about twice as large.



# Compare avg. fare for survived/not

# those who survived paid higher average fare

full_data %>%

        group_by(Survived) %>%

        summarize(avg_fare=mean(Fare))


#The next plot shows that the survivors tend to have paid a higher fare:

ggplot(filter(full_data,!is.na(Survived)),aes(x=Survived,y=Fare))+

        geom_boxplot()


#Look at relationship between P-class and fare

ggplot(full_data,aes(x=Pclass,y=Fare,group=Pclass))+

        geom_boxplot()


#Extract title from passenger names? Idea from <https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic> 



full_data$Title <- gsub('(.*, )|(\\..*)', '', full_data$Name)

table(full_data$Title)

full_data$Title<-as.factor(full_data$Title)





#Make new variables for family size/number of children

full_data <- full_data %>%

mutate(FamSize = SibSp + Parch)
# before building a model, re-split into training and test sets so we can evaluate it's performance?

#

#rm("train_data")

#rm("test_data")

full_data$Sex <- as.factor(full_data$Sex)

train_data <- full_data[1:891,]

test_data <- full_data[892:nrow(full_data),]
# impute missing values

ib <- which(is.na(train_data$Age))

train_data$Age[ib] <- median(train_data$Age,na.rm=TRUE)

ib <- which(is.na(test_data$Age))

test_data$Age[ib] <- median(train_data$Age,na.rm=TRUE)



ib <- which(is.na(train_data$Fare))

train_data$Fare[ib] <- median(train_data$Fare,na.rm=TRUE)

ib <- which(is.na(test_data$Fare))

test_data$Fare[ib] <- median(train_data$Fare,na.rm=TRUE)





str(train_data)




#Let's try a simple random forest model with some of the variables that I think would be important.

#dat_small <- select(dat,Pclass,Sex,Age,Fare,Survived,SibSp,Parch,FamSize,Title)

#head(dat_small)



# impute NAs

#dat_small <- preProcess(dat_small,method = "medianImpute")

#ib <- which(is.na(dat$Age))

#dat$Age[ib] <- median(dat$Age,na.rm=TRUE)



#dat$Survived <- factor(dat$Survived,levels = c(0,1),labels = c("no","yes"))



myFolds <- createFolds(train_data$Survived, k = 10)



# create a control object to pass to caret::train . If we fit multiple different models, this will allow us to fit them all in the same way so we can compare them easily.

myControl=trainControl(classProbs = TRUE, # IMPORTANT!

                       verboseIter = FALSE,

                       savePredictions = TRUE,

                       index=myFolds)





# fit the model using the caret::train function

mod_rf <- train( Survived~Pclass+Age+Sex+Fare+SibSp+Parch+FamSize+Title+Embarked,

              method="rf",

              data=train_data,

              trControl=myControl

              )
print(mod_rf)


#Output results to csv file for submission to Kaggle:'





# Predict using the test set

preds <- predict(mod_rf, newdata=test_data)

levels(preds)<-c(0,1)



# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)

solution <- data.frame(PassengerID = test_data$PassengerId, Survived = preds)

head(solution)

# Write the solution to file

write.csv(solution, file = 'Solution_rf.csv', row.names = F)
