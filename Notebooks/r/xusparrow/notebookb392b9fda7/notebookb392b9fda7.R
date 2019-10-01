library(grid)

library(Matrix)

library(foreach)

library(vcd)

library(ggplot2)

library(stringi)

library(gridExtra)

library(car)

library(glmnet)

library(tree)

library(rpart)

library(gbm)
#Importing data

train <- read.csv('../input/train.csv', header = TRUE, stringsAsFactors = FALSE)

test  <- read.csv('../input/test.csv', header = TRUE, stringsAsFactors = FALSE)



#Seeing the details of train

str(train)
#changing variables to right form

train$Survived <- factor(train$Survived)

train$Pclass   <- factor(train$Pclass)

train$Sex      <- factor(train$Sex)

train$Embarked <- factor(train$Embarked )

train <- data.frame(train)
#check missing values

which(is.na(train$Pclass))#No missing values



library(vcd)

library(grid)

#mosaic plot

mosaic(Survived~Pclass,data=train,shade=T)



library(ggplot2)

#barplot

ggplot(train,

       aes(x=Pclass,fill=Survived))+

       geom_bar()+

       guides(fill=guide_legend(reverse = T))
#checking missing values

which(is.na(train$Sex))



#mosaic

mosaic(Survived~Sex,data=train,shade=T)



#barplot

ggplot(train,

       aes(x=Sex,fill=Survived))+

       geom_bar()+

       guides(fill=guide_legend(reverse = T))