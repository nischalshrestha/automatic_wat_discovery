# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
library(dplyr)

library(ggthemes)
##### Load the titanic dataset

Titanic_train <- read.csv('../input/train.csv')

Titanic_test <- read.csv('../input/test.csv')
##### Initial EDA

glimpse(Titanic_train)
### Transform Pclass to levels

Titanic_train$Pclass <- as.factor(Titanic_train$Pclass)

levels(Titanic_train$Pclass) <- c('Upper', 'Middle', 'Lower')

glimpse(Titanic_train)
### Transform Sex to male and female

Titanic_train$Sex <- as.factor(Titanic_train$Sex)

levels(Titanic_train$Sex) <- c('Female', 'Male')

glimpse(Titanic_train)
### 342 passengers are survived, 549 passengers are dead

sum(Titanic_train$Survived == 1)

sum(Titanic_train$Survived == 0)
### Sex distribution

Sex_Distribution <- ggplot(data = Titanic_train, aes(x=Sex)) + geom_bar(fill='blue', width = 0.7) + ylab('num_of_passenger') + ggtitle('Relationship between Sex and passenger numbers')

Sex_Distribution
### 233 Female and 109 Males are survived; 81 Female and 468 Males are dead

sum(Titanic_train$Survived == 1 & Titanic_train$Sex == 'Female')

sum(Titanic_train$Survived == 1 & Titanic_train$Sex == 'Male')

sum(Titanic_train$Survived == 0 & Titanic_train$Sex == 'Female')

sum(Titanic_train$Survived == 0 & Titanic_train$Sex == 'Male')
### Age distribution

Titanic_train1 <- Titanic_train %>%

  select(Survived, Pclass, Sex, Age) %>%

  filter(Age >= 0)

Age_Distribution <- ggplot() + geom_density(data = Titanic_train1, aes(x=Age), fill='blue', alpha=0.5)

Age_Distribution
### Data Visualization for Age vs Sex vs Survived

Age_Sex_Survived <- ggplot(Titanic_train1, aes(Age, fill = factor(Survived))) + geom_histogram(bins=30) + theme_few() + xlab("Age") + ylab("Count") + facet_grid(.~Sex)+ scale_fill_discrete(name = "Survived") + theme_few()+ ggtitle("Relationship among Age, Sex and Survived")

Age_Sex_Survived
### 61.29% passengers younger than 10 years old survived, 36.65% passengers between 10 and 30 years old survived, 41.80% passengers between 30 and 50 years old survived, 36.49% over 50 years old survived.

max(Titanic_train$Age, na.rm = TRUE)

min(Titanic_train$Age, na.rm = TRUE)

sum(Titanic_train1$Age < 10 & Titanic_train1$Survived == 1) / sum(Titanic_train1$Age < 10)

sum(Titanic_train1$Age >= 10 & Titanic_train1$Age < 30 & Titanic_train1$Survived == 1) / sum(Titanic_train1$Age >= 10 & Titanic_train1$Age < 30)

sum(Titanic_train1$Age >= 30 & Titanic_train1$Age < 50 & Titanic_train1$Survived == 1) / sum(Titanic_train1$Age >= 30 & Titanic_train1$Age < 50)

sum(Titanic_train1$Age >= 50 & Titanic_train1$Survived == 1) / sum(Titanic_train1$Age >= 50)
### Pclass distribution

Pclass_Distribution <- ggplot(data = Titanic_train, aes(x=Pclass)) + geom_bar(fill='blue', width = 0.7) + ylab('num_of_passenger') + ggtitle('Relationship between Pclass and passenger numbers')

Pclass_Distribution
### Data Visualization for Pclass vs Survived

Pclass_Survived <- ggplot(Titanic_train1, aes(Pclass, fill = factor(Survived))) + geom_bar() + theme_few() + xlab("Pclass") + ylab("num_of_passengers") + scale_fill_discrete(name = "Survived") + theme_few()+ ggtitle("Relationship between Pclass and Survived")

Pclass_Survived
### 62.96% passengers with Upper class survived, 47.28% passengers with Middle class Survived, 24.24% passengers with Lower class Survived

sum(Titanic_train$Pclass == 'Upper' & Titanic_train$Survived == 1) / sum(Titanic_train$Pclass == 'Upper')

sum(Titanic_train$Pclass == 'Middle' & Titanic_train$Survived == 1) / sum(Titanic_train$Pclass == 'Middle')

sum(Titanic_train$Pclass == 'Lower' & Titanic_train$Survived == 1) / sum(Titanic_train$Pclass == 'Lower')
### Family Size distribution

Titanic_train$FamilySize <- ifelse(Titanic_train$SibSp + Titanic_train$Parch + 1 == 1, "Single", ifelse(Titanic_train$SibSp + Titanic_train$Parch + 1 < 4, "Small", "Large"))

FamilySize_Distribution <- ggplot(Titanic_train, aes(FamilySize)) + geom_bar(position="dodge", fill = 'blue') +  scale_x_discrete(limits=c("Single", "Small", "Large")) + ylab('num_of_passenger') + ggtitle('Relationship between Family size and passenger numbers')

FamilySize_Distribution
### Data Visualization for Familysize vs Survived

Familysize_Survived <- ggplot(Titanic_train, aes(FamilySize, fill = factor(Survived))) + geom_bar() + theme_few() + xlab("Familysize") + ylab("num_of_passengers") + scale_fill_discrete(name = "Survived") + theme_few()+ ggtitle("Relationship between Familysize and Survived")

Familysize_Survived
### 30.35% passengers with Single family size survived, 56.27% passengers with Small family size survived, 34.07% passengers with Large family size survived

sum(Titanic_train$FamilySize == 'Single' & Titanic_train$Survived == 1) / sum(Titanic_train$FamilySize == 'Single')

sum(Titanic_train$FamilySize == 'Small' & Titanic_train$Survived == 1) / sum(Titanic_train$FamilySize == 'Small')

sum(Titanic_train$FamilySize == 'Large' & Titanic_train$Survived == 1) / sum(Titanic_train$FamilySize == 'Large')
##### Linear Model: Survived = 1.014245 - 0.489949*SexMale - 0.006252*Age - 0.209941*PclassMiddle - 0.400391*PclassLower + 0.148970*FamilySizeSingle + 0.166486*FamilySizeSmall

Titanic_train_model <- Titanic_train %>%

  filter(Age >= 0) %>%

  select(Sex, Age, Pclass, FamilySize, Survived)

Titanic_model <- lm(Survived ~ Sex + Age + Pclass + FamilySize, data = Titanic_train_model)

summary(Titanic_model)
Titanic_test$FamilySize <- ifelse(Titanic_test$SibSp + Titanic_test$Parch + 1 == 1, "Single", ifelse(Titanic_test$SibSp + Titanic_test$Parch + 1 < 4, "Small", "Large"))

Titanic_test$Pclass <- as.factor(Titanic_test$Pclass)

levels(Titanic_test$Pclass) <- c('Upper', 'Middle', 'Lower')

Titanic_test$Sex <- as.factor(Titanic_test$Sex)

levels(Titanic_test$Sex) <- c('Female', 'Male')
submission <- Titanic_test %>%

mutate(Survived = round(predict(Titanic_model, Titanic_test))) %>%

select(PassengerId, Survived)

submission[is.na(submission)] <- 0

submission
write.csv(submission, file = 'submission.csv', row.names = F)