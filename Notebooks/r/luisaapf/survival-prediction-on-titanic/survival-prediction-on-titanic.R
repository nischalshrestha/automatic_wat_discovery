# Loading the libraries

library(VIM) # visualize missing values

library(mice) # impute data

library(lattice) # plot multiple graphs

library(ggplot2) # plot graphs

library(scales) # change ggplot scale

library(randomForest) # model prediction

library(repr) # resize graphs



# Loading the data

train <- read.csv('../input/train.csv', na.strings='')

test  <- read.csv('../input/test.csv', na.strings='')
format(summary(train))
# set plot size

options(repr.plot.width=6, repr.plot.height=4)



aggr(train, sortVars=TRUE, numbers=TRUE, col=c("dodgerblue1","firebrick1"),

     labels=names(train), gap=4, cex.lab=.8, cex.axis=.6, cex.numbers=.5)
train <- train[ , -11]
train$Embarked[is.na(train$Embarked)] <- 'S'
# building the matrix of predictors using only SibSp and Parch to predict Age

m <- matrix(rep(0,55), nrow=5, ncol=11)

v <- c(0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0)

predictorMatrix <- rbind(m, v, m)



impTrain <- mice(train, method='pmm', seed=1000, predictorMatrix=predictorMatrix)
format(impTrain$predictorMatrix)
completeTrain <- complete(impTrain)
d1 <- density(train$Age, na.rm=TRUE)

d2 <- density(completeTrain$Age)



par(mfrow=c(1,2))

plot(d1, main="Before imputation", xlab="Age")

polygon(d1, col="paleturquoise", border="dodgerblue1")

plot(d2, main="After imputation", xlab="Age")

polygon(d2, col="mistyrose", border="firebrick1")
completeTrain$Survived <- as.factor(completeTrain$Survived)



ggplot(completeTrain, aes(x=Sex, fill=Survived)) + 

  geom_bar(width=0.5, aes(y = (..count..)/sum(..count..)), position="fill") +

  scale_y_continuous(labels=percent) +

  scale_fill_discrete(labels=c("No", "Yes")) +

  ylab("Percentage")
completeTrain$AgeRange[completeTrain$Age <= 16] <- "kid"

completeTrain$AgeRange[completeTrain$Age > 16 & completeTrain$Age <= 60] <- "adult"

completeTrain$AgeRange[completeTrain$Age > 60] <- "elderly"



completeTrain$AgeRange <- factor(completeTrain$AgeRange, levels=c("kid", "adult", "elderly"))



summary(completeTrain$AgeRange)
survivalByAge <- ggplot(completeTrain, aes(x=AgeRange, fill=Survived)) + 

  geom_bar(width=0.5, aes(y=(..count..)/sum(..count..)), position="fill") +

  scale_y_continuous(labels=percent) +

  scale_fill_discrete(labels=c("No", "Yes")) +

  labs(x="Age group", y="Percentage")



print(survivalByAge)
survivalByAge + facet_grid(~Sex)
hist(completeTrain$Fare, col="lightblue", xlab="Fare", main="Histogram of Fare",

     breaks=30)

abline(v=median(completeTrain$Fare), col="red", lwd=2)
medianFare <- median(completeTrain$Fare)

completeTrain$FareRange[completeTrain$Fare <= medianFare] <- "low"

completeTrain$FareRange[completeTrain$Fare > medianFare] <- "high"



ggplot(completeTrain, aes(x=FareRange, fill=Survived)) + 

  geom_bar(width=0.5) + labs(x="Fare", y="Counts") +

  scale_fill_discrete(labels=c("No", "Yes"))
completeTrain <- completeTrain[ , -c(12,13)]
survivalModel <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch

                              + Fare + Embarked, data=completeTrain)

print(survivalModel)
importance <- importance(survivalModel)

bluePallete <- colorRampPalette(c("steelblue2", "steelblue4"))

barchart(sort(reorder(rownames(importance), importance)) ~ sort(importance), 

         col=bluePallete(7), xlab="Importance")
format(summary(test))
aggr(test, sortVars=TRUE, numbers=TRUE, col=c("dodgerblue1","firebrick1"),

     labels=names(test), gap=4, cex.lab=.8, cex.axis=.6, cex.numbers=.5)
test <- test[ , -10]

test$Fare[is.na(test$Fare)] <- median(train$Fare)
m1 <- matrix(rep(0, 40), nrow=4, ncol=10)

v <- c(0, 0, 0, 0, 0, 1, 1, 0, 0, 0)

m2 <- matrix(rep(0, 50), nrow=5, ncol=10)



predictorMatrix <- rbind(m1, v, m2)



impTest <- mice(test, method='pmm', predictorMatrix=predictorMatrix, seed=1000)
testComplete <- complete(impTest)
predicted <- predict(survivalModel, testComplete)

result <- data.frame(PassengerId=test$PassengerId, Survived=predicted)

write.csv(result, "prediction.csv", row.names=FALSE)