# Load libraries

library(tidyverse)

library(randomForest)



# load the data files

train <-   read_csv("../input/train.csv")

test <- read_csv("../input/test.csv")