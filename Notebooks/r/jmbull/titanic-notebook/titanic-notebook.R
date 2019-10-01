# Load packages

library('ggplot2') # visualization
train <- read.csv('../input/train.csv', stringsAsFactors = F)

#test  <- read.csv('../input/test.csv', stringsAsFactors = F)



str(train)



message("

big ol' gap

")



# multiple linear regression

lm(formula = Survived ~ Age + Sex, data = train)