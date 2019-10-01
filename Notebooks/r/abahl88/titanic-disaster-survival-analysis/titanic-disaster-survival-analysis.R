# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.


library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(randomForest) # classification algorithm

library(dplyr) # Data wrangling

train <- read.csv('../input/train.csv', stringsAsFactors = F)

test  <- read.csv('../input/test.csv', stringsAsFactors = F)



full  <- bind_rows(train, test) # bind training & test data



full[which(is.na(full$Survived)),2] <- 0 #addressing NA values due to combining of train and test datasets



full$Survived <- as.factor(full$Survived)


# check data

str(full)
head(full)
ggplot(train , aes(x = Survived)) + geom_bar()



table(train$Survived)
sapply(full, function(full) sum(is.na(full)))
which(is.na(full$Fare))
full[1044, ]



thirdclass <- full[full$Pclass == 3 & full$Embarked == 'S' ,  ]



ggplot(thirdclass , aes(Fare)) + geom_density()



median(thirdclass$Fare , na.rm = TRUE)



full$Fare[1044] <- median(thirdclass$Fare , na.rm = TRUE)
Male_Ages <- full[ which(is.na(full$Age)) & full$Sex == 'male' , ]



median(Male_Ages$Age , na.rm =  TRUE)



ggplot(Male_Ages , aes(Age)) + geom_density()



full$Age[is.na(full$Age) == TRUE & full$Sex == 'male'] <- median(Male_Ages$Age , na.rm =  TRUE)
Female_Ages <- full[ which(is.na(full$Age)) & full$Sex == 'female' , ]



median(Female_Ages$Age , na.rm =  TRUE)



ggplot(Female_Ages , aes(Age)) + geom_density()



full$Age[is.na(full$Age) == TRUE & full$Sex == 'female'] <- median(Female_Ages$Age , na.rm =  TRUE)
ggplot(train , aes(x = Pclass , fill = factor(Survived))) + geom_bar(width = 0.5)
ggplot(train, aes(Age,fill = factor(Survived))) +

    geom_histogram(binwidth = 5) + facet_grid(.~Sex)
train <- full[1:891,]

test <- full[892:1309,]
train$Sex <- as.factor(train$Sex)

train$Embarked <- as.factor(train$Embarked)





test$Sex <- as.factor(test$Sex)

test$Embarked <- as.factor(test$Embarked)
sapply(full, function(full) sum(is.na(full)))
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 

                                            Fare , data = train )
round(importance(rf_model), 2)
prediction <- predict(rf_model, test)



solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)