# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(repr)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
train_data = read.csv("../input/train.csv")

nrow(train_data)
# get the age and replace NAs with zeros

na_age <- is.na(train_data$Age)



survived_age <- train_data$Age[train_data$Survived == 1]

survived_age[na_age] <- 0

not_survived_age <- train_data$Age[train_data$Survived == 0]

not_survived_age[na_age] <- 0
# Histogram 

options(repr.plot.width=6, repr.plot.height=4)

par(mfrow=c(1,2))

hist(survived_age, breaks = 15, main='Survived', 

  col='green', xlab = "Age")

hist(not_survived_age, breaks = 15, main = 'NOT Survived', 

  col ='red', xlab = "Age")
survived_age <- survived_age[survived_age != 0]

not_survived_age <- not_survived_age[not_survived_age != 0]
# Histogram 



par(mfrow=c(1,2))

hist(survived_age, main='Survived', col='green', xlab = "Age")

hist(not_survived_age, main = 'NOT Survived', col ='red', xlab = "Age")
survived_gender = train_data$Sex[train_data$Survived == 1]

table(survived_gender)