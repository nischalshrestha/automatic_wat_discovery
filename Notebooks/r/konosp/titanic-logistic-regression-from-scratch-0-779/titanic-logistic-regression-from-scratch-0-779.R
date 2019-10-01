# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



test_data <- read_csv("../input/test.csv")

train_data <- read_csv("../input/train.csv")



library(ggplot2)

library(dplyr)

library(tidyr)

library(readr)



# Any results you write to the current directory are saved as output.
# Generic function to format both training and test set

format.tables <- function(df){

  

  #df$Survived <- as.factor(df$Survived)

  df$Pclass <- as.factor(df$Pclass)

  df$Sex <- as.factor(df$Sex)

  df$Embarked <- as.factor(df$Embarked)

  

  # Dummy variable for Sex

  df$Sex.int <- 0

  df[df$Sex == 'male',]$Sex.int <- 1

  

  # Dummy variable for Pclass

  #df$Pclass <- as.integer(df$Pclass)

  tmp <- model.matrix(PassengerId ~ Pclass,df[,c('PassengerId','Pclass')])

  df <- cbind(df,tmp[,-1])

  

  # Dummy variable if passenger has cabin

  df$has_cabin <- 0

  df[!is.na(df$Cabin),]$has_cabin <- 1

  

  tmp <- model.matrix(PassengerId ~ Embarked,df[,c('PassengerId','Embarked')])

  df <- cbind(df,tmp[,-1])

  

  #df$embarked.C <- 0

  #df$embarked.Q <- 0

  #df[df$Embarked == 'C',]$embarked.C <- 1

  #df[df$Embarked == 'Q',]$embarked.Q <- 1

  

  df$age_missing <- 0

  df[is.na(df$Age),]$age_missing <- 1

  

  # Titles

  titles <- df %>% select('Name') %>% 

    separate(col = Name,into = c('A','B'), sep = ',') %>% 

    separate(col = B,into = c('A2','B2'), sep = '. ') %>% 

    select('A2')

  

  df$title <-trimws(titles$A2)

  # df$title <- as.factor(df$title)

  tmp <- table(df$title) < 15

  df[tmp[df$title] == TRUE,]$title <- 'Other'

  

  tmp <- model.matrix(Name ~ title,df[,c('title','Name')])

  df <- cbind(df,tmp[,-1])

  

  return(df)

}
# Sigmoid function

G <- function(z) {

  return(1 / (1 + exp(-z)))

}



# Defaut definition of Z

# Not used in the actual code

# Z <- t(theta) %*% X



# Cost function

J <- function(theta, X, y){

  z <- X %*% theta

  sigmoid <- G(z)

  m <- length(y)

  res <- (-t(y) %*% log(sigmoid) - (t(1 - y) %*% log(1 - sigmoid)))/m

  return(res)

}



# Hypothesis function sample

# Not used in the actual code

# H <- G(Z)

table(is.na(train_data$Embarked))


# Diclaimer: Idea was found on different kernel

# Filling missing Embarked data

p1 <- ggplot(train_data)

p1 + aes(x = Embarked, y = Fare, color = as.factor(Pclass)) + 

  geom_boxplot() + 

  geom_hline(yintercept = median(train_data[is.na(train_data$Embarked),]$Fare), 

             linetype = 'dashed', color = 'blue')  + theme(legend.position="bottom")
# Fill missing embarked lines based on median

train_data[is.na(train_data$Embarked),]$Embarked <- 'C'
# Format coloumns to 0/1 dummy format

test_data <- format.tables(test_data)

train_data <- format.tables(train_data)
coloumns.to.fit <- c('Pclass2','Pclass3','Sex.int',

                     'SibSp','Parch',

                     'has_cabin','age_missing',

                     'EmbarkedQ','EmbarkedS', 'titleMiss', 'titleMr', 'titleMrs','titleOther')



head(train_data[,coloumns.to.fit],10)
# Vector of zeros: [0,0,0, ... ,0] equal to the number of columns

theta.initial <- as.vector(rep(0,length(coloumns.to.fit)))
# Select the coloums that will be used

X <- as.matrix(train_data[,coloumns.to.fit])

y <- train_data$Survived

# Find parameters theta that optimize (minimize) the cost function J.

# The cost function measures the difference between predicted and actual class

theta.optimized <- optim(theta.initial ,

                         fn = J, 

                         X = X, 

                         y = y,

                         control = list(maxit = 1000000))$par
# Based on the optimized theta, generate the output values of the sigmoid function

sigmoid.result <- G(X %*% theta.optimized)
# The logistic regretion requires to define the threshold that splits the two classes.

# The threshold is a value in the set (0,1) - not inclusive

# Find the thershold that maximises F1

F1.max = 0

F1 = 0

threshold <- 0

for(i in seq(from=0.01, to=1, by=0.005)){

  # For each possible value of the threhold

  # update predictions and evaluate F1

  y_predict <- sigmoid.result > i

  y_predict[y_predict == T] <- 1

  y_predict[y_predict == F] <- 0

  # Performance

  results.table <- table(y_predict, train_data$Survived)

  if (sum(dim(results.table)) == 4) {

    precision <- results.table[2,2] / (results.table[2,2] + results.table[2,1])

    recal <- results.table[2,2] / (results.table[2,2] + results.table[1,2])

    F1 <- 2 * precision * recal / (precision + recal)   

    # print(paste('Threshold: ', threshold, ' - ', 'F1: ', F1))

    if (F1 >= F1.max){

      F1.max <- F1

      threshold <- i

      # print(paste('Threshold: ', threshold, ' - ', 'F1: ', F1.max))

    }

  } 

}
print(theta.optimized)
# Final optimum prediction

y_predict <- sigmoid.result > threshold

y_predict[y_predict == T] <- 1

y_predict[y_predict == F] <- 0



# Performance on train set

results.table <- table(y_predict, train_data$Survived,dnn = c('Predicted','Actual'))

print(results.table)
precision <- results.table[2,2] / (results.table[2,2] + results.table[2,1])

recal <- results.table[2,2] / (results.table[2,2] + results.table[1,2])

F1 <- 2 * precision * recal / (precision + recal)   

print(paste('Threshold calculated: ', threshold, ' - ', 'F1-score: ', F1))


# Visualization

tmp <- cbind(as.data.frame(sigmoid.result), train_data$Survived,train_data$Sex)

names(tmp) <- c('sigmoid','Actual Outcome','Sex')

tmp$'Actual Outcome' <- as.character(tmp$'Actual Outcome')

tmp[tmp$`Actual Outcome` == 1,]$`Actual Outcome` <- 'Survived'

tmp[tmp$`Actual Outcome`== 0,]$`Actual Outcome` <- 'Died'

tmp$`Actual Outcome` <- as.factor(tmp$`Actual Outcome`)

p1 <- ggplot(tmp)

p1 + aes(x = sigmoid, fill = `Actual Outcome`) + geom_histogram(bins = 75) + 

  ggtitle(paste('Threshold: ', threshold, ' - ', 'F1-score: ', round(F1.max,3) )) +

  geom_vline(xintercept= threshold, color = 'blue') + facet_grid(Sex ~ .) +

  geom_text(aes(x=threshold - 0.10, label="Predicted:Died", y=100), colour="brown3", angle=0) +

  geom_text(aes(x=threshold + 0.15, label="Predicted:Survived", y=100), colour="darkolivegreen4", angle=0) +

  geom_text(aes(x=threshold + 0.02, label="Threshold", y = 65), colour="blue", angle=90) +

  xlab('Predicted probability of survival - Result of sigmoid function') +

  ylab('Number of passengers') + theme(legend.position="bottom")