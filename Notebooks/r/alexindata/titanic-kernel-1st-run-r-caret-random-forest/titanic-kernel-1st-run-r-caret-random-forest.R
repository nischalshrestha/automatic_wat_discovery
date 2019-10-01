rm(list=ls())

set.seed(12345)

train <- read.csv("../input/train.csv", stringsAsFactors=F, na.strings=c("NA", ""))

test <- read.csv("../input/test.csv", stringsAsFactors=F, na.strings=c("NA", ""))

test$Survived <- NA

full <- rbind(train, test)



dim(train); dim(test); dim(full)



colnames(train)



str(train)



# line 27
library(dplyr)



# convert Sex to a factor variable, gender

full$gender <- as.factor(full$Sex)

full$gender <- relevel(full$gender, ref="female")



# impute the Cabin NAs

full$cabin_deck <- toupper(substring(full$Cabin, 1, 1))

table(full$Survived, full$cabin_deck, useNA="ifany")

full[full$cabin_deck %in% c('A', 'G', 'T'), ]$cabin_deck <- 'AGT'

full[full$cabin_deck %in% c('B', 'D', 'E'), ]$cabin_deck <- 'BDE'

full[full$cabin_deck %in% c('C', 'F'), ]$cabin_deck <- 'CF'

full[is.na(full$Cabin), ]$cabin_deck <- "unknown"

full$cabin_deck <- as.factor(full$cabin_deck)



# social status, title in names

title <- unique(gsub("^.+, (.+?)\\. .+$", "\\1", full$Name))

title



noble <- c("Dona", "Jonkheer", "the Countess", "Sir", "Lady", "Don")

pros <- c("Col", "Capt", "Major", "Dr")



full$title <- gsub("^.+, (.+?)\\. .+$", "\\1", full$Name)

full[full$title == "Mlle", ]$title <- "Miss"

full[full$title == "Mme" | full$title == "Ms", ]$title <- "Mrs"

full[full$title %in% noble, ]$title <- "noble"

full[full$title %in% pros, ]$title <- "pros"

full$title <- as.factor(full$title)



# impute NA in Fare, convert Fare to a factor variable, fare_type

median_fare <- full %>% group_by(Pclass) %>% summarize(medians=median(Fare, na.rm=T))

full[is.na(full$Fare), ]$Fare <- 

      median_fare[median_fare$Pclass==full[is.na(full$Fare), ]$Pclass ,]$medians



quantile <- quantile(full$Fare, probs=seq(0, 1, 0.2), na.rm=T)

full$fare_grade <- as.factor(cut(full$Fare, breaks=quantile, include.lowest=T, 

                                 labels=c('low', 'low_mid', 'mid', 'mid_hi', 'hi')))



# split Ticket into ticket string and ticket numbers

full$ticket_str <- gsub("(\\D*)\\d+", "\\1", full$Ticket)

full[full$ticket_str == "", ]$ticket_str <- "unavailable"

full$ticket_str <- as.factor(toupper(substring(full$ticket_str, 1, 1)))



full$ticket_num <- sapply(full$Ticket, function(x) 

      unlist(strsplit(x, split=" "))[length(unlist(strsplit(x, split=" ")))])

full[full$ticket_num == "LINE", ]$ticket_num <- 1

full$ticket_num <- as.numeric(full$ticket_num)

full$ticket_num <- as.factor(round(log10(full$ticket_num)))



# factorize Embarked variable

full$embarked <- as.factor(full$Embarked)

full[is.na(full$embarked), ]$embarked <- 'S'



# factorize Pclass

full$pclass <- as.factor(full$Pclass)



# combine SibSp and Parch

full$family <- full$SibSp + full$Parch + 1



# impute the Age NAs, using median age by "title"

library(dplyr)

full$age <- full$Age

medians <- full %>% group_by(title) %>% summarize(medians=median(Age, na.rm=T))

full <- inner_join(full, medians, by='title')

full[is.na(full$age), ]$age <- full[is.na(full$age), ]$medians



# line 100
# model training on training dataset

library(caret, quietly=TRUE, warn.conflicts=FALSE)

library(randomForest, quietly=TRUE, warn.conflicts=FALSE)



library(parallel, quietly=TRUE, warn.conflicts=FALSE)

library(doParallel, quietly=TRUE, warn.conflicts=FALSE)

cluster <- makeCluster(detectCores() - 1) # 1 core for the OS

registerDoParallel(cluster)



# partition full dataset back to training, devset, and test set

train_dev_idx <- 1:dim(train)[1]

test_idx <- (dim(train)[1]+1):dim(full)[1] # dont forget to use () before the :

inTrain_idx <- sample(train_dev_idx, replace=F, round(0.7*length(train_dev_idx)))

inDev_idx <- train_dev_idx[-inTrain_idx]



# line 125
# loop over the list var

colnames(full)



var <- list(c("age", "gender"), 

            c("age", "gender", "Fare"),

            c("age", "gender", "fare_grade"),

            c("age", "gender", "Fare", "family", "title", "pclass", "embarked", "ticket_str", "cabin_deck")

)



data <- list()

accuracy <- list()

rfmod <- list()



for (i in 1:length(var)) {

      data[[i]] <- full[train_dev_idx, var[[i]], drop=F]

      data[[i]] <- cbind(train$Survived, data[[i]])

      colnames(data[[i]])[1] <- 'Survived'

      data[[i]]$Survived <- as.factor(data[[i]]$Survived)

      

      # model training

      

      trControl <- trainControl(method="cv", number=10, allowParallel=TRUE)

      

      training <- data[[i]][inTrain_idx, , drop=F]

      rfmod[[i]] <- train(Survived ~., method="rf", data=training, 

                          trControl=trControl, metric="Accuracy")

      

      # model evaluation with devset

      prediction <- as.data.frame(matrix(0, nrow=length(inDev_idx), ncol=length(var)))



      devset <- data[[i]][inDev_idx, , drop=F]

      prediction[, i] <- predict(rfmod[[i]], newdata=devset[, -1, drop=F])

      

      accuracy[[i]] <- (sum(prediction[, i] == data[[1]][inDev_idx, ][, 1]) 

                                          / length(data[[1]][inDev_idx, ][, 1])) * 100

      print(paste("Variable set", as.character(i), 

                  "of total", as.character(length(var)), "for the var list."))

}



max(unlist(accuracy))

max_acc_idx <- which(unlist(accuracy) == max(unlist(accuracy)))

max_acc_idx



# line 178
best_var <- c("age", "gender", "Fare", "family", "title", "pclass", "embarked", "ticket_str", "cabin_deck")



# re-build the model on the full original training set, train

data <- full[train_dev_idx, best_var]

data <- cbind(train$Survived, data)

colnames(data)[1] <- "Survived"

data$Survived <- as.factor(data$Survived)



# model training 

trControl <- trainControl(method="cv", number=10, allowParallel=TRUE)

rfmod <- train(Survived ~., method="rf", data=data, 

                          trControl=trControl, metric="Accuracy")



rfmod$results



# model evaluation with devset

testset <- full[test_idx, best_var]

prediction <- predict(rfmod, newdata=testset)



# write to file

# df <- data.frame(PassengerId=test$PassengerId, Survived=prediction)

# write.csv(df, file="./data/kaggleTitanicRKernel.csv", row.names=F)



# using random forest in R does not give a good score.



# line 214
stopCluster(cluster)

date()
