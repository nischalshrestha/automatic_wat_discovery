# Load libraries

require(dplyr)

require(ggplot2)

require(data.table)

require(stringr)

require(tidyr)
train <- data.table(read.csv("../input/train.csv"))

test <- data.table(read.csv("../input/test.csv"))
str(train)
train_x <- train[,3:12]

train_y <- train[2]
summary(train_x)




# Grab title from passenger names

train$Cabin_Letter <- as.factor(substr(train$Cabin, 1, 1))



# Show title counts by sex

table(train$Cabin_Letter)



summary(train$Cabin_Letter)
train
train_x[, Cabin_Letter := NULL]