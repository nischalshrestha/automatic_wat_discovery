# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
library(tidyverse)

library(rpart)

library(rpart.plot) 

library(caret)

library(ggplot2)

library(Hmisc)



#import dataset

train<-read_csv("../input/train.csv") 

test<-read_csv("../input/test.csv") 
#How dose Embarked impact on the survival or pessengers

ggplot(train, aes(x = Embarked, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  labs(x = 'Embarked')
#How different the Pclass impact on survial of male & female

ggplot(train,aes(x=Sex,fill=factor(Survived)))+

  geom_bar(position='dodge')+

  facet_grid(.~Pclass)+

  labs(title = "How Different Pclass impact the survival of male&female passengers",x = "Pclass",y = "Count")
#Test how dose the family size impact the survival of pessengers

train$FamilySize<-train$SibSp+train$Parch

ggplot(train, aes(x = FamilySize, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  scale_x_continuous(breaks=c(1:11)) +

  labs(x = 'Family Size')
train$Child[train$Age < 16] <- 'Child'

train$Child[train$Age >= 16] <- 'Adult'



table(train$Child,train$Survived)
#Deal with the missing values

ggplot(train, aes(x=Embarked,y=Fare))+geom_boxplot(aes(fill=factor(Pclass)))

train$Embarked[is.na(train$Embarked)]<-'C'

test[is.na(test$Fare),]



test1<-test[c(test$Embarked=='S'),] 

test2<-test1[c(test1$Pclass==3),]

test3<-test2[complete.cases(test2$Fare),]

test$Fare[is.na(test$Fare)]<-mean(test3$Fare)
#feature engineering

# create title from passenger names

full<-bind_rows(train,test)

full$Child[full$Age < 16] <- 'Child'

full$Child[full$Age >= 16] <- 'Adult'

full$FamilySize<-full$SibSp+full$Parch

full$FsizeD[full$FamilySize == 0] <- 'singleton'

full$FsizeD[full$FamilySize< 4 & full$FamilySize > 0] <- 'small'

full$FsizeD[full$FamilySize >=4 ] <- 'large'



full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

full$Title[full$Title == 'Mlle']        <- 'Miss' 

full$Title[full$Title == 'Ms']          <- 'Miss'

full$Title[full$Title == 'Mme']         <- 'Mrs' 

full$Title[full$Title %in% rare_title]  <- 'Rare Title'

table(full$Sex, full$Title)
#Different combinations of feature

full$Sex <- as.factor(full$Sex)

full$Pclass <- as.factor(full$Pclass)

full$Title<-as.factor(full$Title)

full$Embarked<-as.factor(full$Embarked)

full$FsizeD<-as.factor(full$FsizeD)



train <- full[1:891,]

test <- full[892:1309,]



#Bulid our Modeling

fol <- formula(Survived ~Title+ Fare+ Pclass+Age)

model <- rpart(fol, method="class", data=train)
#Identify the change of the tree

rpart.plot(model,branch=0,branch.type=2,type=1,extra=102,shadow.col="pink",box.col="gray",split.col="magenta",

           main="Decision tree for model")

rpred <- predict(model, newdata=test, type="class")