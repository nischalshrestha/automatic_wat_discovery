# Import the required libraries

library(dplyr)

library(Hmisc)

library(magrittr)



options(stringsAsFactors = FALSE)
train <- read.csv("../input/train.csv")

test <- read.csv("../input/test.csv")

test$Survived <- NA

dataset <- c("train", "test")

n <- c(nrow(train), nrow(test))

full <- rbind(train, test) %>%

    mutate(Dataset = as.factor(rep(dataset, n)),

           Sex = as.factor(Sex),

           Embarked = as.factor(gsub("^$", "S", Embarked)))
full %<>% mutate(AgeImp = impute(Age))
full %<>% mutate(CabinKnown = as.numeric(nchar(Cabin) > 0))

xtabs(~ Survived + CabinKnown, full, subset = Dataset == "train")