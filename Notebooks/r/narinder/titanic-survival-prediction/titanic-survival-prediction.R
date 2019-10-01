# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(pROC)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
my.train.data <- read.csv("../input/train.csv")



#head(my.train.data)
my.test.data <- read.csv("../input/test.csv")



#head(my.test.data)
#my.train.data[,c("PassengerId", "Name", "Ticket", "Cabin")]



my.train.data
my.train.data$Cabin <- ifelse(is.na(my.train.data$Cabin), "NA", my.train.data$Cabin)





#my.train.data
drop.cols <- c("PassengerId", "Name", "Ticket", "Cabin")

my.train.data <- my.train.data[, !names(my.train.data) %in% drop.cols]



#head(my.train.data)
for(i in 1:ncol(my.train.data)){

    if(is.numeric(my.train.data[,i])){

        my.train.data[is.na(my.train.data[,i]), i] <- mean(my.train.data[,i], na.rm = TRUE)

    }

}



#my.train.data
#my.test.data$Cabin <- ifelse(is.na(my.test.data$Cabin), "NA", my.test.data$Cabin)



for(i in 1:ncol(my.test.data)){

    if(is.numeric(my.test.data[,i])){

        my.test.data[is.na(my.test.data[,i]), i] <- mean(my.test.data[,i], na.rm = TRUE)

        }

}





#my.test.data
set.seed(1313)

analysis <- glm(Survived ~ ., data = my.train.data, family = binomial(link = "logit"))
summary(analysis)
#my.test.data <- my.test.data[complete.cases(my.test.data),]
score <- data.frame(Survived = predict(analysis, newdata = my.test.data, type = "response"))

score_train <- data.frame(Prediction = predict(analysis, newdata = my.train.data, type = "response"))
auc(my.train.data$Survived, score_train$Prediction)
score$Survived <- ifelse(score$Survived > 0.5,1,0)
complete <- cbind(my.test.data, score)
write_csv(complete[,c("PassengerId", "Survived")], path = "myPredictions.csv")