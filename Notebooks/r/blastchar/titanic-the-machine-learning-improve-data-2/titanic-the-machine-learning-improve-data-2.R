# supress anoing warnings for now

options(warn=-1)



# Necessary Libraries 

library(ggplot2)

library(cowplot)



library(lattice)

library(caret)

library(MASS)



# for describe the data

library(psych)



# need for the vim library 

library(colorspace)

library(grid)

library(data.table)



# for validate missing data 

library(mice) 

library(VIM)
# load train.csv

DstTrain <- read.csv('../input/train.csv', stringsAsFactors = FALSE)

# load test.csv

DstTest  <- read.csv('../input/test.csv', stringsAsFactors = FALSE)
# check data sample 

head(DstTrain, n=5L)
# dimensions of dataset

dim(DstTrain)
# check column type

sapply(DstTrain, typeof)
# show columns info

str(DstTrain)
# Describe the load data only numerics

describeBy(DstTrain[,sapply(DstTrain, is.numeric)], na.rm = TRUE)
# unique to validate if "" exist

unique(DstTrain$Sex)

unique(DstTrain$Embarked)
# check for the pattern of missing data

md.pattern(DstTrain)
# let check who have missing data

pMiss <- function(x){sum(is.na(x))/length(x)*100}

apply(DstTrain,2,pMiss)

#apply(DstTrain,1,pMiss)
# more helpful visual representation

aggr_plot <- aggr(DstTrain, col=c('#8cb3d9','#4d0000'), 

                  numbers=TRUE, sortVars=TRUE, labels=names(DstTrain), 

                  cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
# more helpful visual representation

marginplot(DstTrain[c(1,6)], col=c('#8cb3d9','#4d0000'))
# let focus on the age column for now

summary(DstTrain$Age)
# Imputing the missing data

# The mice() function takes care of the imputing process

newAgeData <- mice(DstTrain,m=5,maxit=50,meth='pmm',seed=500)

summary(newAgeData)
# the complete clean dataset

DstTrainClean <- complete(newAgeData,1)



#check for missing

apply(DstTrainClean,2,pMiss)
# Compute the largest y value used in the Age remove NA

NoNAAge <- DstTrain[!is.na(DstTrain$Age),c("Age")]

hist(NoNAAge, col=heat.colors(10), breaks=10, main="Original ages")
#Clean Ages dataset

hist(DstTrainClean$Age, col=heat.colors(10), breaks=10, main="Clean NA ages")
# check DstTest

apply(DstTest,2,pMiss)
newData <- mice(DstTest,m=5,maxit=50,meth='pmm',seed=500)
# the complete clean dataset

DstTestClean <- complete(newData,1)



# check DstTestClean

apply(DstTestClean,2,pMiss)
# get the title from name

DstTrainClean$Title <- gsub('(.*, )|(\\..*)', '', DstTrainClean$Name)

DstTestClean$Title <- gsub('(.*, )|(\\..*)', '', DstTestClean$Name)
# histogram of Title

ggplot(DstTrainClean, aes(Title,fill = factor(Survived))) + 

geom_histogram(stat="count") + 

theme(axis.text.x = element_text(angle = 60, hjust = 1))
# Titles by Sex

table(DstTrainClean$Sex, DstTrainClean$Title)
# 'Mr', 'Mrs', 'Miss', 'Mme', 'Ms', 'Mlle'

# 'Master', 'Major', 'Capt', 'Col', 'Rev', 'Dr'

# 'the Countess', 'Jonkheer', 'Lady', 'Sir', 'Don', 'Dona'



Nobility.woman <- c('the Countess', 'Lady', 'Dona')

Nobility.man <- c('Jonkheer', 'Sir', 'Don')

Crew <- c('Master', 'Major', 'Capt', 'Col', 'Rev', 'Dr')

Passenger.woman <- c('Mrs', 'Miss', 'Mme', 'Ms', 'Mlle')

Passenger.man <- c('Mr')



DstTestClean$TitleType <- ifelse(DstTestClean$Title %in% Nobility.woman, 'Nobility.woman','')

DstTestClean$TitleType <- ifelse(DstTestClean$Title %in% Nobility.man, 'Nobility.man',DstTestClean$TitleType)

DstTestClean$TitleType <- ifelse(DstTestClean$Title %in% Crew, 'Crew',DstTestClean$TitleType)

DstTestClean$TitleType <- ifelse(DstTestClean$Title %in% Passenger.woman, 'Passenger.woman',DstTestClean$TitleType)

DstTestClean$TitleType <- ifelse(DstTestClean$Title %in% Passenger.man, 'Passenger.man',DstTestClean$TitleType)



DstTrainClean$TitleType <- ifelse(DstTrainClean$Title %in% Nobility.woman, 'Nobility.woman','')

DstTrainClean$TitleType <- ifelse(DstTrainClean$Title %in% Nobility.man, 'Nobility.man',DstTrainClean$TitleType)

DstTrainClean$TitleType <- ifelse(DstTrainClean$Title %in% Crew, 'Crew',DstTrainClean$TitleType)

DstTrainClean$TitleType <- ifelse(DstTrainClean$Title %in% Passenger.woman, 'Passenger.woman',DstTrainClean$TitleType)

DstTrainClean$TitleType <- ifelse(DstTrainClean$Title %in% Passenger.man, 'Passenger.man',DstTrainClean$TitleType)
# clean NA values

#DstTrain[is.na(DstTrain)] <- 0

#DstTest[is.na(DstTest)] <- 0
# histogram of SibSp

plot.SibSp <- ggplot(DstTrainClean, aes(SibSp,fill = factor(Survived))) +

    geom_histogram(stat = "count")



# histogram of Pclass

plot.Pclass <- ggplot(DstTrainClean, aes(Pclass,fill = factor(Survived))) +

    geom_histogram(stat = "count")



# histogram of Sex

plot.Sex <- ggplot(DstTrainClean, aes(Sex,fill = factor(Survived))) +

    geom_histogram(stat = "count")



# histogram of Embarked

plot.Embarked <- ggplot(DstTrainClean, aes(Embarked,fill = factor(Survived))) +

    geom_histogram(stat = "count")



# create the plot grid with all

plot_grid(plot.SibSp, plot.Pclass, plot.Sex, plot.Embarked )
# histogram of Title Type

ggplot(DstTrainClean, aes(TitleType,fill = factor(Survived))) + 

geom_histogram(stat="count") + 

theme(axis.text.x = element_text(angle = 60, hjust = 1))
# histogram of Age

ggplot(DstTrainClean, aes(Age,fill = factor(Survived))) + 

geom_histogram(bins = 40)



# histogram of Fare

ggplot(DstTrainClean, aes(Fare,fill = factor(Survived))) + 

geom_histogram(bins = 40)
# Run algorithms using 10-fold cross validation

control <- trainControl(method="cv", number=10)

metric <- "Accuracy"
# create the test dataset with only the testing columns

varsToKeep <- c("Survived", "TitleType", "Sex", "Age", "Pclass", "Embarked", "Fare", "SibSp")

DstTrainTest <- DstTrainClean[varsToKeep]



# convert Survived to factor

DstTrainTest$Survived <- as.factor(DstTrainTest$Survived)



# logistic regression

set.seed(7)

fit.glm <- train(Survived ~ ., data=DstTrainTest, method="glm", metric=metric, trControl=control)



# linear algorithms

set.seed(7)

fit.lda <- train(Survived ~ ., data=DstTrainTest, method="lda", metric=metric, trControl=control)



# CART

set.seed(7)

fit.cart <- train(Survived ~ ., data=DstTrainTest, method="rpart", metric=metric, trControl=control)



# kNN

set.seed(7)

fit.knn <- train(Survived ~ ., data=DstTrainTest, method="knn", metric=metric, trControl=control)



# SVM

set.seed(7)

fit.svm <- train(Survived ~ ., data=DstTrainTest, method="svmRadial", metric=metric, trControl=control)



# Random Forest

set.seed(7)

fit.rf <- train(Survived ~ ., data=DstTrainTest, method="rf", metric=metric, trControl=control)

# summarize accuracy of models

results <- resamples(list(

    glm=fit.glm, 

    lda=fit.lda, 

    cart=fit.cart, 

    knn=fit.knn, 

    svm=fit.svm, 

    rf=fit.rf

))

summary(results)
# compare accuracy of models

dotplot(results)
# summarize Best Model

print(fit.rf)
# prediction 

predictedval <- predict(fit.rf, newdata=DstTrainTest)



# summarize results with confusion matrix

cm <- confusionMatrix(predictedval, DstTrainClean$Survived)

cm$table



# calculate accuracy of the model

Accuracy<-round(cm$overall[1],2)

Accuracy

# prediction 

predictedval <- predict(fit.rf, newdata=DstTestClean)
# create a csv file for submittion

TitanicResult <- data.frame(PassengerId = DstTestClean$PassengerId, Survived = predictedval)

table(TitanicResult$Survived)

write.csv(TitanicResult,file = "TitanicResult.csv",row.names = FALSE)