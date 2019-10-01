#Arthor: Chuanhui Liu

#Date:   2017-03-20

library(tidyverse)

library(rpart)

library(rpart.plot) 

library(caret)

library(ggplot2)

library(Hmisc)



#import dataset

train<-read_csv("../input/train.csv") 

test<-read_csv("../input/test.csv") 



#basic information of dataset, finding missing values

#describe(train)

#describe(test)



#head(train)

# PassengerId Survived Pclass                                                Name    Sex   Age SibSp Parch           Ticket    Fare Cabin Embarked

# <int>    <int>  <int>                                               <chr>  <chr> <dbl> <int> <int>            <chr>   <dbl> <chr>    <chr>

#   1        0      3                             Braund, Mr. Owen Harris   male    22     1     0        A/5 21171  7.2500  <NA>        S

#   2        1      1 Cumings, Mrs. John Bradley (Florence Briggs Thayer) female    38     1     0         PC 17599 71.2833   C85        C

#   3        1      3                              Heikkinen, Miss. Laina female    26     0     0 STON/O2. 3101282  7.9250  <NA>        S

#   4        1      1        Futrelle, Mrs. Jacques Heath (Lily May Peel) female    35     1     0           113803 53.1000  C123        S

#   5        0      3                            Allen, Mr. William Henry   male    35     0     0           373450  8.0500  <NA>        S

#   6        0      3                                    Moran, Mr. James   male    NA     0     0           330877  8.4583  <NA>        Q
#

ggplot(train, aes(x = Embarked, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  labs(x = 'Embarked')



ggplot(train,aes(x=Sex,fill=factor(Survived)))+

  geom_bar(position='dodge')+

  facet_grid(.~Pclass)+

  labs(title = "How Different Pclass impact the survival of male&female passengers",x = "Pclass",y = "Count")
#Fsize<1 Fsize>=5 have penalty for survival chances

train$FamilySize<-train$SibSp+train$Parch

ggplot(train, aes(x = FamilySize, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  scale_x_continuous(breaks=c(1:11)) +

  labs(x = 'Family Size')





train$Child[train$Age < 16] <- 'Child'

train$Child[train$Age >= 16] <- 'Adult'



table(train$Child,train$Survived)







#missing value(Embarked/Fare/age)



ggplot(train, aes(x=Embarked,y=Fare))+geom_boxplot(aes(fill=factor(Pclass)))

#which means missing values in Embarked, most probably, is C

train$Embarked[is.na(train$Embarked)]<-'C'



#test[is.na(test$Fare),]

#  PassengerId Pclass               Name   Sex   Age SibSp Parch Ticket  Fare Cabin Embarked

# <int>  <int>              <chr> <chr> <dbl> <int> <int>  <chr> <dbl> <chr>    <chr>

#   1        1044      3 Storey, Mr. Thomas  male  60.5     0     0   3701    NA  <NA>    S

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



#factorize variables for modelling

full$Sex <- as.factor(full$Sex)

full$Pclass <- as.factor(full$Pclass)

full$Title<-as.factor(full$Title)

full$Embarked<-as.factor(full$Embarked)

full$FsizeD<-as.factor(full$FsizeD)



train <- full[1:891,]

test <- full[892:1309,]

#Modeling

fol <- formula(Survived ~Title+ Fare+ Pclass+Age)

model <- rpart(fol, method="class", data=train)
rpart.plot(model,branch=0,branch.type=2,type=1,extra=102,shadow.col="pink",box.col="gray",split.col="magenta",

           main="Decision tree for model")

rpred <- predict(model, newdata=test, type="class")
#write into solutions

Survived<-as.numeric(levels(rpred)[rpred])

PassengerId<-test$PassengerId

solution<-cbind(PassengerId,Survived)

write.csv(data.frame(solution),file = 'my solution.csv',row.names= F)