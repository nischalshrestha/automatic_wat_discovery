# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 

packageStartupMessage()

library(plyr)

library(dplyr)

library(ggplot2)

library(DT)

library(randomForest)

library(corrplot)

library(caret)

library(lattice)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.

mytheme_1 <- function() {

  return(

    theme(

      axis.text.x = element_text(

        angle = 90,

        size = 10,

        vjust = 0.4

      ),

      plot.title = element_text(size = 15, vjust = 2),

      axis.title.x = element_text(size = 12, vjust = -0.35)

    )

  )

}



mytheme_2 <- function() {

  return(

    theme(

      axis.text.x = element_text(size = 10, vjust = 0.4),

      plot.title = element_text(size = 15, vjust = 2),

      axis.title.x = element_text(size = 12, vjust = -0.35)

    )

  )

}

# Exploratory Analysis 



train <- read.csv("../input/train.csv")

test <- read.csv("../input/test.csv")





str(train)
datatable(train)

head(train)
# Univariate analysis - Categorical Variable analysis

train$Survived <- as.factor(train$Survived)

train$Pclass <- as.factor(train$Pclass)



histogram(train$Survived,  xlab = "Survived" )



prop.table(table(train$Survived))

## Only 38% of population survived in the training data



histogram(train$Pclass, xlab="Passenger Class")



prop.table(table(train$Pclass))

## 55% of the population travelled in 3rd Class



histogram(train$Sex,  xlab = "Passenger Sex" )



prop.table(table(train$Sex))

## 64% of the population are male and 35% of population or female



levels(train$Embarked)[1] <- "S"



histogram(train$Embarked,  xlab = "Passenger Embarked" )



prop.table(table(train$Embarked))

histogram(train$Sex,  xlab = "Passenger Sex" )



prop.table(table(train$Sex))

## 64% of the population are male and 35% of population or female

summary(train$Age)



histogram(train$Age)
summary(train$SibSp)

summary(train$Fare)



histogram(train$Fare, xlab="Fare")

test$Pclass <- as.factor(test$Pclass)



full <- bind_rows(train, test)



colSums(is.na(full))
full$Title <- sapply(full$Name, FUN = function(x) { strsplit(x, split = '[,.]')[[1]][2] } )



full$Title <- sub(' ', '', full$Title)



table(full$Title)



full$Title[full$Title %in% c("Mme", "Mlle") ] <- "Mlle"



full$Title[full$Title %in% c('Capt', 'Don', 'Major', 'Sir') ] <- "Sir"



full$Title[full$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'



full$Title <- as.factor(full$Title)



table(full$Title)

age_by_title <- full %>% group_by(Title) %>% summarise(Age = median(Age, na.rm=T ))



full[is.na(full$Age) & full$Title == "Mr", ]$Age <-age_by_title[age_by_title$Title== "Mr", ]$Age



full[is.na(full$Age) & full$Title == "Mrs", ]$Age <-age_by_title[age_by_title$Title== "Mrs", ]$Age



full[is.na(full$Age) & full$Title == "Miss", ]$Age <-age_by_title[age_by_title$Title== "Miss", ]$Age



full[is.na(full$Age) & full$Title == "Master", ]$Age <-age_by_title[age_by_title$Title== "Master", ]$Age



full[is.na(full$Age) & full$Title == "Dr", ]$Age <-age_by_title[age_by_title$Title== "Dr", ]$Age



full[is.na(full$Age) & full$Title == "Ms", ]$Age <-age_by_title[age_by_title$Title== "Ms", ]$Age



colSums(is.na(full))

Fare_by_Pclass <- full %>% group_by(Pclass) %>% summarise(median_Fare = median(Fare, na.rm=T) )



full[is.na(full$Fare),]$Fare <- Fare_by_Pclass[Fare_by_Pclass$Pclass=="3", ]$median_Fare



colSums(is.na(full))

full$FamilySize <- full$Parch + full$SibSp + 1



full$Surname <- sapply(full$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})



full$FamilyID <- paste(as.character(full$FamilySize), full$Surname, sep="")



famIDs <- data.frame(table(full$FamilyID ))



famIDs <- famIDs[famIDs$Freq <= 2,]



full$FamilyID <-as.character(full$FamilyID)



full$FamilyID[full$FamilyID %in% famIDs$Var1] <- 'Small'



full$FamilyID <- as.factor(full$FamilyID)



table(full$FamilyID)



full$isAlone <- ifelse(full$FamilySize == 1, 1,0 )



full$isCabin <- ifelse(full$Cabin == "", 0, 1)



full$Ticket <- as.character(full$Ticket)

full$Fare_Category <- NA



full$Fare_Category <- ifelse(full$Fare <= 50, "Low", full$Fare_Category )

full$Fare_Category <- ifelse(full$Fare > 50 & full$Fare <= 150, "Medium", full$Fare_Category )

full$Fare_Category <- ifelse(full$Fare > 150, "High", full$Fare_Category )



ggplot(full[1:nrow(train),], aes(Fare_Category, fill=Survived )) +

  geom_bar(stat = "count")

ggplot(full[!is.na(full$Survived),], aes(Age, fill=Survived)) +

  geom_histogram(stat  ="bin",bins = 12 )

full$Age_Category <- NA



full$Age_Category <- ifelse(full$Age <= 10, "Child", full$Age_Category)

full$Age_Category <- ifelse(full$Age > 10 & full$Age <= 22, "Tean", full$Age_Category)

full$Age_Category <- ifelse(full$Age > 22 & full$Age <= 35, "Adult", full$Age_Category)

full$Age_Category <- ifelse(full$Age > 35 , "Aged", full$Age_Category)



full$Age_Category <- as.factor(full$Age_Category)



full$FarePerPerson <- full$Fare/full$FamilySize





ggplot(full[!is.na(full$Survived),], aes(Age_Category, fill=Survived)) +

  geom_bar(stat="count")

ggplot(subset(full[1:nrow(train),], FamilyID != "Small" ), aes(FamilyID, fill=Survived)) +

  geom_bar(stat="count") +

  mytheme_1()

ggplot(full[1:nrow(train),], aes(FamilySize, fill=Survived)) +

  geom_bar(stat="count") +

  facet_wrap(~Sex+Pclass, nrow =2)+

  mytheme_1()

# Family Size greater than 6 not survived
ggplot(full[1:nrow(train),], aes(PassengerId, Age, color=Survived )  ) +

  geom_point() +

  facet_wrap(~Pclass+Title) +

  mytheme_2()



ggplot(full[1:nrow(train),], aes(PassengerId, Age, color=Survived )  ) +

  geom_point() +

  facet_wrap(~Pclass+Sex) +

  mytheme_2()

# Age above 20 people not survived

ggplot(full[1:nrow(train),], aes(PassengerId, Age, color=Survived )  ) +

  geom_point() +

  facet_wrap(~SibSp+Pclass) +

  mytheme_2()



# SibSp size more than 3 not survived

ggplot(full[1:nrow(train),], aes(PassengerId, Age, color=Survived )  ) +

  geom_point() +

  facet_wrap(~Parch+Pclass) +

  mytheme_2()



# Parch size more than 3 not survived
