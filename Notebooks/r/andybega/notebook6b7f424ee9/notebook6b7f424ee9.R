library("rio")

library("ggplot2")

library("randomForest")



train <- import("../input/train.csv")

test  <- import("../input/test.csv")



str(train)
list.files("../input")