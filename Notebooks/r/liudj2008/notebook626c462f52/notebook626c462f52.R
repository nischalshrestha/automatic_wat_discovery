# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
setwd('../input/titanic')
# Read train data to titanic file

setwd('../input/titanic')

dir()



titanic<-read.csv('train.csv',header=TRUE)



# first and last 3 lines of info

head(titanic,3)

tail(titanic,3)
#data frame structure of titanic

str(titanic)

#number of survived

num_survived<-length(which(titanic$Survived==1))

num_survived

#number of loss

num_loss<-length(which(titanic$Survived==0))

num_loss

# numbers of missing values in each column 

missing_values<-c(length(which(is.na(titanic$PassengerId))),length(which(is.na(titanic$Pclass))),length(which(is.na(titanic$Sex))),length(which(is.na(titanic$Age))),length(which(is.na(titanic$SibSp))),length(which(is.na(titanic$Parch))),length(which(is.na(titanic$Ticket))), length(which(is.na(titanic$Fare))),length(which(titanic$Cabin=='')),length(which(titanic$Embarked=='')))

missing_values
par(mfrow=c(4,2))



# Sex V.S. Survival

Sex_survival<-table(titanic$Survived,titanic$Sex)

barplot(Sex_survival,xlab='Sex', ylab='Headcounts', col = c('grey','green'),legend=rownames(Sex_survival),beside =T)

# Pclass V.S. Survival

Pclass_survival<-table(titanic$Survived,titanic$Pclass)

barplot(Pclass_survival,xlab='Pclass', ylab='Headcounts', col = c('grey','green'),legend=rownames(Pclass_survival),beside =T)



#Parch V.S. Survival

Parch_survival<-table(titanic$Survived,titanic$Parch)

barplot(Parch_survival,xlab='Parch', ylab='Headcounts', col = c('grey','green'),legend=rownames(Parch_survival),beside =T)



#Age V.S. Survival

survival<-titanic[which(titanic$Survived==1),]

loss<-titanic[which(titanic$Survived==0),]

hist(survival$Age, col=rgb(1,0,0,0.5),main='Histogram of Age',xlab='Age')

hist(loss$Age, col=rgb(0,0,1,0.5),add=T)



#SibSp V.S. Survival

SibSp_survival<-table(titanic$Survived,titanic$SibSp)

barplot(SibSp_survival,xlab='SibSp', ylab='Headcounts', col = c('grey','green'),legend=rownames(SibSp_survival),beside =T)



#Fare V.S. Survival

survival<-titanic[which(titanic$Survived==1),]

loss<-titanic[which(titanic$Survived==0),]

hist(survival$Fare, col=rgb(1,0,0,0.5),main='Histogram of Fare',xlab='Fare')

hist(loss$Fare, col=rgb(0,0,1,0.5),add=T)



#Embarked V.S. Survival

Embarked_survival<-table(titanic$Survived,titanic$Embarked)

barplot(Embarked_survival,xlab='Embarked', ylab='Headcounts', col = c('grey','green'),legend=rownames(Embarked_survival),beside =T)
# Transfer the useful data to new_titanic;

new_titanic<-titanic[,c(2,3,5,6,7,8,10,12)]



# Info of new_titanic

str(new_titanic)
# convert the 'sex' data to numeric data

# Convert the levels of sex from 'female','male' to '1' and '2' respectfully

levels(new_titanic$Sex)<-c(1,2)



# Convert the sex column first to factor and then to numeric

new_titanic$Sex<-as.numeric(factor(new_titanic$Sex))



# Convert the Embarked column to numeric

levels(new_titanic$Embarked)<-c(NA,1,2,3)

new_titanic$Embarked<-as.numeric(factor(new_titanic$Embarked))



# Show head and tail of new_titanic

head(new_titanic)

tail(new_titanic)

str(new_titanic)
# Remove the NA in the data

nrow(new_titanic)

new_titanic<-na.omit(new_titanic)

nrow(new_titanic)

write.csv(new_titanic,'clean_file.csv')