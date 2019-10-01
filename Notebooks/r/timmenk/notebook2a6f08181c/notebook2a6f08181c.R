# Load packages

library('ggplot2') # visualization

library('ggthemes') # visualization

library('scales') # visualization

library('dplyr') # data manipulation

library('mice') # imputation

library('randomForest') # classification algorithm
train <- read.csv('../input/train.csv', stringAsFactors = F)

test  <- read.csv('../input/test.csv', stringsAsFactors = F)



full <- bindrows(train, test)



str(full)


