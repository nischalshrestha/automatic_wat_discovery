library('dplyr') # data manipulation

library('ggplot2') # Data Visualization

library('ggthemes') # Data Visualization



options(warn = -1)

# load train.csv

train <- read.csv('../input/train.csv', stringsAsFactors = F)

# load test.csv

test  <- read.csv('../input/test.csv', stringsAsFactors = F)

# combine them as a whole

test$Survived <- NA

full <- rbind(train,test)
# show first several rows of the data

head(full)


# check the data

str(full)
# Process Age Column



    # create a new data set age

age <- full$Age

n = length(age)

    # replace missing value with a random sample from raw data

set.seed(123)

for(i in 1:n){

  if(is.na(age[i])){

    age[i] = sample(na.omit(full$Age),1)

  }

}

    # check effect

par(mfrow=c(1,2))

hist(full$Age, freq=F, main='Before Replacement', 

  col='lightblue', ylim=c(0,0.04),xlab = "age")

hist(age, freq=F, main='After Replacement', 

  col='darkblue', ylim=c(0,0.04))
# Process Cabin Column to show number of cabins passenger has

cabin <- full$Cabin

n = length(cabin)

for(i in 1:n){

  if(nchar(cabin[i]) == 0){

    cabin[i] = 0

  } else{

    s = strsplit(cabin[i]," ")

    cabin[i] = length(s[[1]])

  }

} 

table(cabin)
# number of survivals and nonsurvivals across different age

d <- data.frame(Age = age[1:891], Survived = train$Survived)

ggplot(d, aes(Age,fill = factor(Survived))) +

    geom_histogram()
# create bar chart to show relationship between survival rate and age intervals

cuts <- cut(d$Age,hist(d$Age,10,plot = F)$breaks)

rate <- tapply(d$Survived,cuts,mean)

d2 <- data.frame(age = names(rate),rate)

barplot(d2$rate, xlab = "age",ylab = "survival rate")
# create histgram to show effect of Sex on survival

ggplot(train, aes(Sex,fill = factor(Survived))) +

    geom_histogram(stat = "count")
# calculate survival rate

tapply(train$Survived,train$Sex,mean)
# extract title from Name

n = length(full$Survived)

title = rep(NA,n)

for (i in 1:n){

  lastname = strsplit(train$Name[i],", ")[[1]][2]

  title[i] = strsplit(lastname,". ")[[1]][1]

}



# make a histogram of title v.s survival

d <- data.frame(title = title[1:891],Survived = train$Survived)

ggplot(d, aes(title,fill = factor(Survived))) +

    geom_histogram(stat = "count")
# count of title

table(title)
# survival rate

tapply(d$Survived,d$title,mean)
# replace rare titles to 'Rare'

title[title != 'Mr' & title != 'Miss' & title != 'Mrs' & title != 'Master'] <- 'Rare'

table(title)
# make a histogram

ggplot(train, aes(Pclass,fill = factor(Survived))) +

    geom_histogram(stat = "count")
# calculate survival rate

tapply(train$Survived,train$Pclass,mean)
# histogram of Parch

ggplot(train, aes(Parch,fill = factor(Survived))) +

    geom_histogram(stat = "count")
# histogram of SibSp

ggplot(train, aes(SibSp,fill = factor(Survived))) +

    geom_histogram(stat = "count")
# combine SibSp and Parch 

family <- full$SibSp + full$Parch

d <- data.frame(family = family[1:891],Survived = train$Survived)

ggplot(d, aes(family,fill = factor(Survived))) +

    geom_histogram(stat = "count")
tapply(d$Survived,d$family,mean)
# create histogram

d <- data.frame(Cabin = cabin[1:891],Survived = train$Survived)

ggplot(d, aes(Cabin,fill = factor(Survived))) +

    geom_histogram(stat = "count")
# calculate survival rate

tapply(d$Survived,d$Cabin,mean)
# make a histogram

ggplot(train, aes(Fare,fill = factor(Survived))) +

    geom_histogram()
# calculate

cuts <- cut(train$Fare,hist(train$Fare,50,plot = F)$breaks)

rate <- tapply(train$Survived,cuts,mean)

d <- data.frame(fare = names(rate),rate)

barplot(d$rate, xlab = "fare",ylab = "survival rate")
# make histogram

ggplot(train, aes(Embarked,fill = factor(Survived))) +

    geom_histogram(stat = "count")
# make table

tapply(train$Survived,train$Embarked,mean)