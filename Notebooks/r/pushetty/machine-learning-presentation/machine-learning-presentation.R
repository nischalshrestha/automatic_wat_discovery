library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(rpart)

library(randomForest)



# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
#Descriptive Statistics with histogram and Boxplots

train_titanic<- read.table("../input/train.csv", header=T,sep=",")

test_titanic<- read.table("../input/test.csv", header=T,sep=",")



str(train_titanic)

str(test_titanic)

#boxplot(Age ~ Pclass,data=train_titanic)

#boxplot(Age ~ Survived,data=train_titanic)

#hist(train_titanic$Age)
#Inferential Statistics with Chi-square  [Tends to 0 -- Heavily Depends , Otherwise independent]

chisq.test(train_titanic$Survived,train_titanic$Sex)

chisq.test(train_titanic$Survived,train_titanic$Age)



#ScatterPlot to check if there is a correlation 

plot(train_titanic$Fare, train_titanic$Age, xlab = 'Fare', ylab = 'Age')



cor.test(train_titanic$Age,train_titanic$Fare, method='pearson')
#Logistic regression to show the relation for Binary Variable with the rest of the Dependent VAriables

# This is an ugly Fit



fit <- glm(Survived ~ Age + Pclass + Sex + SibSp + Parch + Fare + Embarked,

           data = train_titanic, family = binomial(link = 'logit'))

summary(fit)
# FEATURE ENGINEERING AND RANDOM FOREST APPLIED

# SHow your Data manipulation skills here pushpu



#Attach element 'Survived' to the test set and combine test and train dataset

test_titanic$Survived <- NA

titanic <- rbind(train_titanic,test_titanic)



##########################            FEATURE:: 1   EXPLORATORY        #########################



#Strip off titles from Names [Make em string first]

titanic$Name <- as.character(titanic$Name)

titanic$Title <- sapply(titanic$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})

titanic$Title <- sub(' ', '', titanic$Title)



#See varied titles

table(titanic$Title)



# Combine small title groups

titanic$Title[titanic$Title %in% c('Mme', 'Mlle')] <- 'Mlle'

titanic$Title[titanic$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'

titanic$Title[titanic$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

# Convert to a factor

titanic$Title <- factor(titanic$Title)



############################             FEATURE:: 2    NORMALIZE(Reduce Outliers)           #####################

#Check All features having NAs 

summary(titanic)

# fixup Age with NAs

summary(titanic$Age)  

Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title, 

                data=titanic[!is.na(titanic$Age),], method="anova")

titanic$Age[is.na(titanic$Age)] <- predict(Agefit, titanic[is.na(titanic$Age),])

#check if the Age NAs have been fixed

summary(titanic$Age)



# fixup blanks in Embarked

summary(titanic$Embarked)

which(titanic$Embarked == '')

titanic$Embarked[c(62,830)] = "S"

titanic$Embarked <- factor(titanic$Embarked)

# Fill in Fare NAs

summary(titanic$Fare)

which(is.na(titanic$Fare))

titanic$Fare[1044] <- median(titanic$Fare, na.rm=TRUE)



#All NAs have been gone now

summary(titanic)

# Predicitive Analytics start here, force your seatbelts  -- Pilot Pushparaj :)  -- Random Forests ahead

#Split back to train and test

train_titanic <- titanic[1:891,]

test_titanic <- titanic[892:1309,]

summary(test_titanic)

# Build Random Forest Ensemble

set.seed(465)

model <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title,data=train_titanic, importance=TRUE, ntree=2000)

# Look at variable importance

varImpPlot(model)

# Now let's make a prediction and write a submission file

Prediction <- predict(model, test_titanic)

summary(Prediction)



submit <- data.frame(PassengerId = test_titanic$PassengerId, Survived = Prediction)

write.csv(submit, file = "random_forest_prediction.csv", row.names = FALSE)


