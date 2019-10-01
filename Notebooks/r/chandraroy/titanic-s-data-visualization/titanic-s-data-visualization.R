library(ggplot2)

#library(tidyverse) ## A smart alternative to use ggplot functions

#library(dplyr)

#library(ggvis)# Web friendly plots.



train.df <- read.csv("../input/train.csv", header = TRUE)

test.df <- read.csv("../input/test.csv", header = TRUE)
names(train.df)

names(test.df)
test.df2 <- data.frame(Survived = rep(0, nrow(test.df)), test.df[,])
combined.df <- rbind(train.df, test.df2)



# Check the fields wit NAs

colnames(combined.df)[ apply(combined.df, 2, anyNA) ]



# Impute missing Age with mean

summary(combined.df)



combined.df$Age[is.na(combined.df$Age)] <-28.00



# Replace NA in Embarked with most common Embarked point

table (combined.df$Embarked)

# Replace NAs in Embarked column with 'S'; it is the most common point of boarding

train.df$Embarked[is.na(train.df$Embarked)] <- 'S'

combined.df$Embarked[is.na(combined.df$Embarked)] <- 'S'



# Visualize Age and Gender distribution.



ggplot(data = combined.df)+

  geom_histogram(mapping = aes(x = Age, fill = Sex))



ggplot(data = combined.df)+

  geom_bar(mapping = aes(x = Pclass, fill = Sex))

# Plot Correlation between survivors and Gender

train.df$Survived <- as.factor(train.df$Survived)

ggplot(data = train.df )+

    geom_point( mapping = aes( x = Age, y = Fare, fill = Survived, color = Survived))

    #geom_smooth()
ggplot(data = train.df)+

    geom_bar(mapping = aes(x = Age, fill = Sex))
#Survived is int; convert it into categorical variable i.e. factor

train.df$Survived <- as.factor(train.df$Survived)

#Plot: A bar chart with Embarked on X axis.

ggplot(data = train.df)+

    geom_bar(mapping = aes( x = Embarked, fill = Survived))
ggplot( data = combined.df)+

    geom_bar(mapping = aes( x = Fare, fill = Pclass))
  boxplot( combined.df$Fare, col = "palevioletred1")

# Alternatively one can use summary() function.

# It displays Minimum, Maximum, Median, and Mean value.

# Further, it also displays the 1st and 3rd Quratile for detecting outliers.

# Outlier > Median + 1.5 * 3rd IQR

summary(combined.df$Fare)