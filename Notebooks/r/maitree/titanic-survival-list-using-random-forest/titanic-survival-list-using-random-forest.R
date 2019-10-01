# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
train <- read.csv('../input/train.csv')

test  <- read.csv('../input/test.csv')
## find the NA fields in the dataset



sapply(train,function(x) sum(is.na(x)))



sapply(train,function(x) summary(is.na(x)))

       ## There is one value in Fare which is NA and many values in Age which has NA values
## getting a backup of train dataset



train1 <- train 
## since some of the data are missing, lets combine the two datasets to apply my logic to get values

## for the NA fields



train$Survived<-NULL



## join the two datasets to get a new combined dataset



combo<-rbind(train,test)
## I will generate a function to get the missing values. 



## to get the age values, lets do some analysis on age column



## age vs Class



library(ggplot2)



ggplot(data=combo,aes(x=factor(Pclass),y=Age,fill=factor(Pclass)))+

  geom_bar(stat="identity",position = position_dodge())
## age vs Class grouped dodge



ggplot(combo,aes(Age,fill=factor(Pclass)))+

  geom_bar(binwidth=1,position = position_dodge())
## the plot shows the class depend on different age group people. we will generate the 

## NA age columns as depending on the class

## Function to get the mean of the age for the Pclass

## for a missing age, get the Pclass and get the mean of age for that Pclass and replace this 

## mean value for the missing age





mean_class <- function(class){

  classvec<-subset(combo,Pclass==class)

  mean_age<-mean(classvec$Age,na.rm=TRUE)

  return(mean_age)

}



l_age <- length(combo$Age)

library(dplyr)



i<-1

for(i in 1:l_age) {

  if (is.na(combo$Age[i])==TRUE){

    class_value<-combo$Pclass[i]

    combo$Age[i]<-mean_class(class_value)

  }



  }
## Class vs fare

  

  ggplot(combo,aes(Fare,fill=factor(Pclass)))+

    geom_bar(stat = "identity",position = position_dodge(),binwidth = 5)
## class 3 has the highest fare

  

  fare_na_index <- which(is.na(combo$Fare))

  

 df<-subset(combo,Pclass=3)

 vec<-df$Fare

 

 

 mean(vec,na.rm = TRUE)

 combo$Fare[fare_na_index]<-mean(vec,na.rm = TRUE)
## Calculate the number of cabins per passenger id

 cabin_no<-function(string){

   return(length(strsplit(string," ")[[1]]))

 }

 

 l_cabin<-length(combo$Cabin)

 combo$Cabin<-as.character(combo$Cabin)

 

 l<-0

 for(l in 1:l_cabin){

     

   cabin_string<-combo$Cabin[l]

   if (cabin_string==" "){

     combo$Cabin[l]<-0

   }

   else{

   cabin_count<-cabin_no(cabin_string)

   combo$Cabin[l]<-cabin_count

   }

   

 }

 

 combo$Cabin<-as.numeric(combo$Cabin)

 
## converting the factor values to numeric

 combo$Sex <- factor(x=combo$Sex,labels = c(1,2))

 combo$Embarked<-replace(combo$Embarked,combo$Embarked=="","S")

 combo$Embarked<-factor(x=combo$Embarked,labels=c(1,2,3))

 
## getting the family member number

 

 names(combo)

 combo$family<-combo$SibSp + combo$Parch

 combo$SibSp<-NULL

 combo$Parch<-NULL
## Ignoring the name and Ticket

 

 combo$Name<-NULL

 combo$Ticket<-NULL

 

 
names(combo)
## At this point I have got my entire data, from here I can apply any model I want on my data. 



## I would want to go in the conventional process and will apply a simple classification on this data.



## for classification, I first need to divide my data again



nrow(train)

nrow(test)
## Let me create another training dataset



train <- combo[1:891,]

test <- combo[892:1309,]

train$Survived <- train1$Survived

names(train)

## Random Forest formula

library(randomForest)

train$Survived<-as.character(train$Survived)

train$Survived<-as.factor(train$Survived)

classifier<- randomForest(x=train[-8],y=train$Survived,ntree=500)

class(train$Survived)
test$Survived<-rep(c(0,1))
test$Survived<-as.factor(test$Survived)
str(train)

str(test)




y_pred=predict(classifier,newdata=test)
## From the above summary details, we will remove the columns which are least co related to Survival

## so new formula is 



test$Survived<-y_pred
## Prediction is 





submit<-data.frame(test$PassengerId,test$Survived)

names(submit)<-c("PassengerId","Survived")



write.csv(submit,file='survivalRF.csv',row.names=FALSE)