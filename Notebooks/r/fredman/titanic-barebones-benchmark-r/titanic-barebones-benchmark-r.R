# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
# Set seed and read input data

set.seed(1)

train <- read.csv("../input/train.csv", stringsAsFactors=FALSE)

test  <- read.csv("../input/test.csv",  stringsAsFactors=FALSE)
#inspect data

head(train)

summary(train)
selected_features <- c("Pclass","Age","Sex","Parch","SibSp","Fare","Embarked")



extractFeatures <- function(data) {

  features <- c("Pclass",

                "Age",

                "Sex",

                "Parch",

                "SibSp",

                "Fare",

                "Embarked")

  fea <- data[,features]

  fea$Age[is.na(fea$Age)] <- -1

  fea$Fare[is.na(fea$Fare)] <- median(fea$Fare, na.rm=TRUE)

  fea$Embarked[fea$Embarked==""] = "S"

  fea$Sex      <- as.factor(fea$Sex)

  fea$Embarked <- as.factor(fea$Embarked)

  return(fea)

}
# standard random forest

library(randomForest)

rf <- randomForest(extractFeatures(train), as.factor(train$Survived), ntree=100, importance=TRUE)



submission <- data.frame(PassengerId = test$PassengerId)

submission$Survived <- predict(rf, extractFeatures(test))
#export predictions

write.csv(submission, file = "1_random_forest_r_submission.csv", row.names=FALSE)