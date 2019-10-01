# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2)

library(readr)

library(plyr)

library(dplyr)

#library(mice) # imputation

library(randomForest) # classification algorithm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")

# Any results you write to the current directory are saved as output.
train <- read.csv("../input/train.csv", stringsAsFactors=F)

test <- read.csv("../input/test.csv", stringsAsFactors=F)



str(train)

str(test)
#First, make a full set from train and test



#Check that there are no NAs in train$Survive

train %>%

summarise(numSurviveNAs = sum(is.na(Survived)))



#Create a test$Survived column so we can combine train and test

test$Survived <- NA

full <- rbind(train, test)



str(full)
#Which variables have missing data?

varMissings <- c()

for(column in names(full)) {

    misses <- sapply(full[[column]], function(x) {return(x=='' | is.na(x))})

    varMissings <- c(varMissings, sum(misses))

}

names(varMissings) <- names(full)



varMissings


makeBarPlot <- function(df, column) {

    print(column)

    column <- as.name(column)

    #tabulatedDf <- df %>%

    #    group_by_(column) %>%

    #    summarise(count = n())

    tabulatedDf <- count(df, column) 

    head(tabulatedDf)

    ggplot(tabulatedDf, aes(x=column, y=count, fill=factor(column), label=count)) +

        geom_bar(stat="identity") +

        labs(title="Bar Graph of Survival", fill=column)  +

       geom_text(aes(y = count + 10), position = position_dodge(0.9))

}



makeBarPlot(full, "Survived")
#Categorical variables



survived <- full %>%

    group_by(Survived) %>%

    summarise(count = n())

ggplot(survived, aes(x=Survived, y=count, fill=factor(Survived), label=count)) +

    geom_bar(stat="identity") +

    labs(title="Bar Graph of Survival", fill="Survived") +

    geom_text(aes(y = count + 10), position = position_dodge(0.9))



pclass <- full %>%

    group_by(Pclass) %>%

    summarise(count = n())

ggplot(pclass, aes(x=Pclass, y=count, fill=factor(Pclass), label=count)) +

    geom_bar(stat="identity") +

    labs(title="Bar Graph of Class", fill="Survived") +

    geom_text(aes(y = count + 10), position = position_dodge(0.9))



sex <- full %>%

    group_by(Sex) %>%

    summarise(count = n())

ggplot(sex, aes(x=Sex, y=count, fill=Sex, label=count)) +

    geom_bar(stat="identity") +

    labs(title="Bar Graph of Gender", fill="Sex")  +

   geom_text(aes(y = count + 10), position = position_dodge(0.9))



embarked <- full %>%

    group_by(Embarked) %>%

    summarise(count = n())

ggplot(embarked, aes(x=Embarked, y=count, fill=factor(Embarked), label=count)) +

    geom_bar(stat="identity") +

    labs(title="Bar Graph of Embarkment Location", fill="Survived") +

    geom_text(aes(y = count + 10), position = position_dodge(0.9))
#continuous variables



ggplot(full, aes(x=Age)) +

    geom_histogram() +

    labs(title="Histogram of Age")



ggplot(full, aes(x=Sibsp)) +

    geom_histogram() +

    labs(title="Histogram of Siblings and Spouses")



ggplot(full, aes(x=Parch)) +

    geom_histogram() +

    labs(title="Histogram of Parents and Children")



ggplot(full, aes(x=Fare)) +

    geom_histogram() +

    labs(title="Histogram of Fare")



summary(full$Age)

#Sibsp

#Parch

#fare

#Continuous



ggplot(full, aes(Age)) +

    geom_histogram() +

    labs(title="Histogram of Age")

ggplot(full, aes(Sibsp)) +

    geom_histogram() +

    labs(title="Histogram of Number of Siblings and Spouses")

ggplot(full, aes(Parch)) +

    geom_histogram() +

    labs(title="Histogram of Number of Parents and Children")

ggplot(full, aes(Fare)) +

    geom_histogram() +

    labs(title="Histogram of Fares")

#Check out the person who is missing fare

full[is.na(full$Fare),]
#How much do people of his class and age tend to pay?

thirdClass <- full[full$Pclass==3,]

print(paste("The median fare of third class passengers is:", median(thirdClass$Fare, na.rm=T)))



par(mfrow=c(1, 2))

ggplot(thirdClass, aes(x=Age, y=Fare)) +

    geom_point() +

    labs(title="Scatterplot of Fares and Ages")

ggplot(thirdClass, aes(Fare)) +

    geom_density() +

    labs(title="Density of Third Class Fares")
#Fill in missing fare

full$Fare[is.na(full$Fare)] <- median(thirdClass$Fare, na.rm=T)
#Have a look at who is missing embarkment

train[train$Embarked=='',]
#Where would people of their class and fare would likely embark?



# Get rid of our missing values

embarkFare <- train[train$Embarked!='',]



# Use ggplot2 to visualize embarkment, passenger class, & median fare

ggplot(embarkFare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +

  geom_boxplot() +

  geom_hline(aes(yintercept=80), 

    colour='red', linetype='dashed', lwd=1) +

  labs(title="Fares of Embarkment Locations by Class", fill="Passenger Class")
#fill in missing embarkment value

train$Embarked[train$Embarked==''] <- 'C'




# Make variables factors into factors

factor_vars <- c('PassengerId','Pclass','Sex','Embarked')



full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))



# Perform mice imputation, excluding certain less-than-useful variables:

mice_mod <- mice(full[, c("Pclass", "Fare", "Embarked", "SibSp", "Parch", "Survived")], method='rf') 

    

# Save the complete output 

mice_output <- complete(mice_mod)
# Plot age distributions

par(mfrow=c(1,2))

hist(full$Age, freq=F, main='Age: Original Data', 

  col='darkgreen', ylim=c(0,0.04))

hist(mice_output$Age, freq=F, main='Age: MICE Output', 

  col='lightgreen', ylim=c(0,0.04))
# Get passenger surnames

train$Surname <- gsub(',.*', '', train$Name)



freqs <- table(train$Surname)

head(freqs[order(-freqs)])

# Most common surnames
head(train$Cabin)
library(corrplot)

nums <- sapply(train, is.numeric)

numericSubset <- train[, nums]

numericSubset <- numericSubset[!is.na(numericSubset$Age),]

c <- cor(numericSubset)

corrplot(c)
library(randomForest)

# Split the data back into a train set and a test set

train <- full[1:891,]

test <- full[892:1309,]



# Build the model (note: not all possible variables are used)

rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 

                                            Fare + Embarked,

                                            data = train)



# Show model error

plot(rf_model, ylim=c(0,0.36))

legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)