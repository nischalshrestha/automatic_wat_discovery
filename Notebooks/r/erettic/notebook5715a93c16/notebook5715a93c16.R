# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
list.files("../input")
train <- read_csv("../input/train.csv")
head(train)
summary(train)
hist(train$Age, breaks = length(unique(train$Age)))
hist(train$Age[train$Survived == 0], breaks = length(unique(train$Age)))
hist(train$Age[train$Pclass == 1 & train$Survived == 0], breaks = length(unique(train$Age)))
hist(train$Age[train$Pclass == 2], breaks = length(unique(train$Age)))
hist(train$Age[train$Pclass == 3], breaks = length(unique(train$Age)))
train$Sex["Sex" == "male"] <- 1
train$Sex["Sex" == "female"] <- 0
summary(train)
head(train)
summary(as.factor(train$Sex))
female <- which(train$Sex == "female")
male <- which(train$Sex == "male")
head(female)
train$Sex[female] <- 0
train$Sex[male] <- 1
head(train)
