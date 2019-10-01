# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
setwd("../input")

train <- read.csv("train.csv", na.strings = "")

test <- read.csv("test.csv")

str(train)

train$Survived <- as.factor(train$Survived)

train$Pclass <-  as.factor(train$Pclass)
#missing values

library(Amelia)

missmap(train)
sum(is.na(train$Age) == TRUE ) / length(train$Age)
#all the attributes in % of missing

sapply(train, function(df) {

  sum(is.na(df) == TRUE) / length(df);

})
#Input missing values



table(train$Embarked, useNA = "always")

#function (..., exclude = if (useNA == "no") c(NA, NaN), 

#useNA = c("no", "ifany", "always"), 

#dnn = list.names(...), deparse.level = 1)



#For Embarkmed we will assign the most probable port "S"



train$Embarked[which(is.na(train$Embarked))] = "S"

table(train$Embarked, useNA = "always")#done!



#Names....first we will see what kind of words are in "name"



train$Name = as.character(train$Name)

table_words = table(unlist(strsplit(train$Name, "\\s+"))) #separar por blanks

sort(table_words [grep("\\.", names(table_words))], 

     decreasing = TRUE) #lista por el tratamiento (que acaba en punto)



table_words



#obtain which titles contain missing values



library(stringr)

tb = cbind(train$Age, str_match(train$Name, "[a-zA-Z]+\\."))

table(tb[is.na(tb[,1]),2])



#we will assign the mean of the age for the title examples 

#with no missing data



mean.mr <- mean(train$Age[grepl("Mr\\.", train$Name) & 

                            !is.na(train$Age)])

mean.mrs <- mean(train$Age[grepl("Mrs\\.", train$Name) & 

                            !is.na(train$Age)] )

mean.dr <- mean(train$Age[grepl("Dr\\.", train$Name) & 

                             !is.na(train$Age)] )

mean.miss <- mean(train$Age[grepl("Miss\\.", train$Name) & 

                            !is.na(train$Age)] )

mean.master <- mean(train$Age[grepl("Master\\.", train$Name) & 

                            !is.na(train$Age)] )



#assign

#

train$Age[grepl("Mr\\.", train$Name) & is.na(train$Age)] = mean.mr

train$Age[grepl("Mrs\\.", train$Name) & is.na(train$Age)] = mean.mrs

train$Age[grepl("Dr\\.", train$Name) & is.na(train$Age)] = mean.dr

train$Age[grepl("Master\\.", train$Name) & is.na(train$Age)] = mean.master

train$Age[grepl("Miss\\.", train$Name) & is.na(train$Age)] = mean.miss



missmap(train)
#Exploring and visualizing data



barplot(table(train$Survived), main = "Passenger survival", 

        names = c("Perished", "Survived"))

barplot(table(train$Pclass), main = "Passenger Class", 

        names = c("First", "Second", "Third"))

barplot(table(train$Sex), main = "Passenger Gender")

hist(train$Age, main = "Passenger age", xlab = "Age")

barplot(table(train$SibSp), main = "Passenger Siblings")

barplot(table(train$Parch), main = "Passenger Parch")

hist(train$Fare, main = "Passenger Fare", xlab = "Fare")

barplot(table(train$Embarked), main = "Passenger Embarkation")



counts <- table(train$Survived, train$Sex)

counts

col <- c("darkblue", "red")
boxplot(train$Age~train$Survived, main = "Passenger survival by Age")
#separate passengers by bin age

#subsets

#

train_child <- train$Survived[train$Age < 13]

length(train_child[which(train_child == 1)])/length(train_child)



train_youth <- train$Survived[train$Age >= 13 & train$Age < 25]

length(train_youth[which(train_youth == 1)])/length(train_youth)



train_adult <- train$Survived[train$Age >= 25 & train$Age < 65]

length(train_youth[which(train_adult == 1)])/length(train_adult)



train_senior<- train$Survived[train$Age >= 65]

length(train_youth[which(train_senior == 1)])/length(train_senior)

#mosaic plot



mosaicplot(train$Pclass ~ train$Survived, 

           main = "Pass. survival Class", color = TRUE,

           xlab = "Pclass", ylab = "Survived")
#PREDICTION

#DECISION TREE



set.seed(666)

trainSize <- round(nrow(train)*0.70)

testSize <- nrow(train) - trainSize

index <- sample(seq_len(nrow(train)), size = trainSize)

train1 <- train[index,]

test1 <-  train[-index,] 



library(party)

#condition tree



train_ctree <- ctree(Survived ~ Pclass + Sex + Age + 

                        SibSp + Fare + Parch + Embarked,

                      data = train1)

plot(train_ctree)



train_ctree



#svm

library(e1071)

svm_model <- svm(Survived ~ Pclass + Sex + Age + 

  SibSp + Fare + Parch + Embarked,

data = train1, probability = TRUE)
#validation



ctree_prediction <- predict(train_ctree, test1)

ctree_prediction



plot(ctree_prediction, col= "blue")+  

plot(test1$Survived, col= "red", add = T) 



library(caret)

#confusion matrix

confusionMatrix(ctree_prediction, test1$Survived)



#probability matrix

train_ctree_pred <- predict(train_ctree, test1)

train_ctree_prob <- 1-unlist(treeresponse(train_ctree, test1,

                                          use.names = T)[seq(1,

                                                             nrow(test1)*2,2)])

library(ROCR)

train_ctree_prob_rocr <- prediction(train_ctree_prob, 

                                    test1$Survived)