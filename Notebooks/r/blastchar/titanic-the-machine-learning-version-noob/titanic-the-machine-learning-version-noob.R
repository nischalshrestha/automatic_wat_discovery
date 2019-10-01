# supress anoing warnings for now

options(warn=-1)



# Necessary Libraries 

library(ggplot2)

library(cowplot)

library(lattice)

library(caret)

library(MASS)
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
# show dataframe info

str(DstTrain)
# clean NA values

DstTrain[is.na(DstTrain)] <- 0

DstTest[is.na(DstTest)] <- 0
# histogram of SibSp

plot.SibSp <- ggplot(DstTrain, aes(SibSp,fill = factor(Survived))) +

    geom_histogram(stat = "count")



# histogram of Pclass

plot.Pclass <- ggplot(DstTrain, aes(Pclass,fill = factor(Survived))) +

    geom_histogram(stat = "count")



# histogram of Sex

plot.Sex <- ggplot(DstTrain, aes(Sex,fill = factor(Survived))) +

    geom_histogram(stat = "count")



# histogram of Embarked

plot.Embarked <- ggplot(DstTrain, aes(Embarked,fill = factor(Survived))) +

    geom_histogram(stat = "count")



# create the plot grid with all

plot_grid(plot.SibSp, plot.Pclass, plot.Sex, plot.Embarked )
# histogram of Age

ggplot(DstTrain, aes(Age,fill = factor(Survived))) + 

geom_histogram(bins = 40)



# histogram of Fare

ggplot(DstTrain, aes(Fare,fill = factor(Survived))) + 

geom_histogram(bins = 40)
# Run algorithms using 10-fold cross validation

control <- trainControl(method="cv", number=10)

metric <- "Accuracy"
# create the test dataset with only the testing columns

varsToKeep <- c("Survived", "Age", "Fare", "Sex", "Embarked", "SibSp", "Pclass")

DstTrainTest <- DstTrain[varsToKeep]



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



# set the Survived variable

#predict.Survived <- ifelse(predictedval > 0.5, 1, 0)

#predict.Survived <- as.integer(predict.Survived)



# summarize results with confusion matrix

cm <- confusionMatrix(predictedval, DstTrain$Survived)

cm$table



# calculate accuracy of the model

Accuracy<-round(cm$overall[1],2)

Accuracy

# prediction 

predictedval <- predict(fit.rf, newdata=DstTest)
# create a csv file for submittion

TitanicResult <- data.frame(PassengerId = DstTest$PassengerId, Survived = predictedval)

table(TitanicResult$Survived)

write.csv(TitanicResult,file = "TitanicResult.csv",row.names = FALSE)