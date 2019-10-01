# Dataset: Titanic (Kaggle)

# Purpose: Data Exploration 

# Jessica Leighton

# 12/31/16 - 1/2/16



#install.packages("ggplot2")

library('ggplot2')



#Step 1: read in data set, check out variables



#to find pathway, go in terminal

#could also used setwd("~/Downloads") to set working directory

train <- read.csv("~/Downloads/train.csv")

test <- read.csv("~/Downloads/test.csv")

variable.names(train)

str(train)

variable.names(test)



# train has extra binary column "Survive"

# create empty vector of int 0's as dummy to make columns equal

# stitch empty vector onto test

empty <- as.integer(matrix(0, nrow = 418))

test["Survived"] <- empty

str(test)



# alternatively, test$Survived <- rep(0, 418) does the same thing

?rep # rep repeats(element, many times)



# set-up if doing feature engineering later

# source: https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

# combine test and train, changes must be consistent across both 

full <- rbind(test, train)



# a few ways to look at variables:

str(full) # method that displays structure of an object str=structure

summary(full) #only shows 5 number summary, not useful for qualitative

variable.names(full) # lists only variable names

# Variables include: class, name, sex, age, # sib, # parent, ticket, fare, cabin



# how many survived? as count and percent

?prop.table

table(train$Survived) #calculates count of 0's and 1's as a table

prop.table(table(train$Survived)) # converts above to decimal



# how many females survived vrs males?

table(train$Sex, train$Survived)

# note: cor(train$Survived, train$Sex) does not work; inputs must be numeric

prop.table(table(train$Sex, train$Survived))

# shows 52% of passengers were males who did not survive

# shows 26% of passengers were females who did survive



# source: trevorstephens github

# https://github.com/trevorstephens/titanic/blob/master/Tutorial2.R

summary(train$Age)

train$Child <- 0

train$Child[train$Age < 18] <- 1

  #creates a binary for child/not.child, set to 1 if child =true

?aggregate 

aggregate(Survived ~ Child + Sex, data= train, FUN = sum)

195+38+86+23

  #Since total survived = 342, breaks down survivors by child/not.child, M/F

  #FUN details how to compute summary statistics

aggregate(Survived ~ Child + Sex, data= train, FUN = length)

259+55+519+58

  #Since total data set = 891, breaks down whole passenger list child/not, M/F

aggregate(Survived ~ Child + Sex, data=train, FUN = function(x) {sum(x)/length(x)})

  # Using sum (survivors) and length (total), finds percent of each





# understanding aggregate FUN=length, source: 

# https://www.r-bloggers.com/aggregate-a-powerful-tool-for-data-frame-in-r/

values <- data.frame(value = c("a", "a", "a", "a", "a", "b", "b", "b", 

                               "c", "c", "c", "c"))

aggregate(x = values, by = list(unique.values = values$value), 

                               FUN = length)

# using data frame "values", lists number of occurances of each unique value





savehistory()

quit()


