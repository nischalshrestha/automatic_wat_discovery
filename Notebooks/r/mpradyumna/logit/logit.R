# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 

library(readr)

library(ggplot2)

library(ROCR)

library(ggplot2)

library(caret)

library(mice)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



rawdata <- read.csv('../input/train.csv',stringsAsFactors = FALSE)

names(rawdata)

str(rawdata)



mi.data <- mice(rawdata, seed = 1234, printFlag = FALSE)



mice:::print.mids(mi.data)



plot(mi.data)



cleandata <- complete(mi.data, action = 3)

sum(is.na(cleandata))



set.seed(3456)

trainIndex <- sample(1:nrow(cleandata),floor(0.7*nrow(cleandata)), replace = TRUE)



train_data <- cleandata[trainIndex,]

test_data <- cleandata[-trainIndex,]





model <- glm(Survived~Age+Pclass+Sex+Parch, family ="binomial", data=train_data)

summary(model)



predictions <- predict(model, type="response") 

pred_class <- ifelse(predictions> 0.5, 1, 0)





confusion.mat_train <- table(train_data$Survived,pred_class)

accuracy_train <- sum(diag(confusion.mat_train))/sum(confusion.mat_train)

precision_train <- confusion.mat_train[2,2]/sum(confusion.mat_train[,2])

recall_train <- confusion.mat_train[2,2]/sum(confusion.mat_train[2,])





fitted.results <- predict(model,test_data,type='response')

fitted.class <- ifelse(fitted.results > 0.5,1,0)





confusion.mat_test = table(test_data$Survived,fitted.class)

accuracy_test = sum(diag(confusion.mat_test))/sum(confusion.mat_test)

precision_test = confusion.mat_test[2,2]/sum(confusion.mat_test[,2])

recall_test = confusion.mat_test[2,2]/sum(confusion.mat_test[2,])





predicted <- predict(model,type="response")

prob <- prediction(predicted, train_data$Survived)

tprfpr <- performance(prob, "tpr", "fpr")

plot(tprfpr)



cutoffs <- data.frame(cut=tprfpr@alpha.values[[1]], fpr=tprfpr@x.values[[1]], 

                      tpr=tprfpr@y.values[[1]])

cutoffs <- cutoffs[order(cutoffs$tpr, decreasing=TRUE),]



head(subset(cutoffs, fpr < 0.2))



plot(tprfpr, colorize = TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))



tpr <- unlist(slot(tprfpr, "y.values"))

fpr <- unlist(slot(tprfpr, "x.values"))

roc <- data.frame(tpr, fpr)



ggplot(roc) + geom_line(aes(x = fpr, y = tpr)) + 

  geom_abline(intercept=0,slope=1,colour="blue") + 

  ylab("Sensitivity") +    xlab("1 - Specificity")





submit <- data.frame(PassengerId = test_data$PassengerId, Survived = test_data$Survived)

write.csv(submit, file = "theyallperish.csv", row.names = FALSE)