library(ggplot2) # We shall use this for data visualisation

library(readr) # the read_csv function to load our data into the script

test<-read.csv("../input/test.csv",stringsAsFactors=FALSE) #load the test.csv file





#factoring columns in the test.csv

test$Pclass <- as.factor(test$Pclass)

test$Sex <-as.factor(test$Sex)



#gender distribution on the ship

ggplot(test,aes(x=Sex))+

  geom_bar()
ggplot(test,aes(Age,fill=Sex))+

  geom_histogram(bins = 10)
#distrubtion of sex by age faceted by class

ggplot(test,aes(Age,fill=Sex))+

  facet_grid(~Pclass)+

  geom_histogram(bins = 10)

train <- read.csv("../input/train.csv") #load data set

#factoring

train$Survived<- as.factor(train$Survived)

train$Pclass<- as.factor(train$Pclass)

train$Sex<- as.factor(train$Sex)

#how many people survived

ggplot(train,aes(Survived))+

  geom_bar()



ggplot(train,aes(x= Age,y=Survived,colour= Sex))+

  geom_point() 
ggplot(train,aes(x= Age,y=Survived,colour= Sex))+

  facet_grid(~Pclass)+

  geom_point()

  