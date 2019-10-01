# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
## This is my first stab at a kaggle script. I have choosen to work with the Titanic Dataset afetr spending some time 

## poking around on the site and looking at other scripts made by other kagglers for inspiration. i will also focus 

## on doinfg some illustrative data visualizations along the way. I will then use Random Forest to create a model predicting

## Survival on the Titanic 

##There are three parts to my scripts as follows:

##  1. Feature Engineering 

 ## 2. Missing Value imoutation 

## 3. Prediction

## Load and check the Data

library(ggplot2)# visualization

library(ggthemes)# visualization

library(scales)# visualization

library(dplyr)# data manipulation

library(mice)# imputation

library(randomForest)# classification algorith
train <- read.csv('../input/train.csv',stringsAsFactors = FALSE)

test <- read.csv('../input/test.csv',stringsAsFactors = FALSE)

DataBind <- bind_rows(train,test)
## Check Data

str(DataBind)

summary(DataBind)

## WE have got a sense of our variables, their class type, and the first few observations of each. We know we are working with 1309 observation of 12 variables to make things a bit more explicit since a couple of the variable names aren't 100% illuminating here is what we have got to deal with.

## Variable Name                           Description

## Survived                               Survived(1) or died(0)

##Pclass                                  Passenger's class

##Name                                    Passenger's Name

## Sex                                    Passenger's Sex

##Age                                     Passenger's Age

##SibSp                                   Number of siblings/spouses abroad

## Parch                                  Number of Parents/ children abroad

##Tickets                                 Tickets Number

##Fare                                    Fare

## Cabin                                  Cabin

## Embarked                               Port of embarkation
## Feature Engineering

## Grab Title from Passenger Names

DataBind$Title <- gsub('(.*, )|(\\..*)','',DataBind$Name)
## Show Title Count By Sex

table(DataBind$Sex,DataBind$Title)
##Title with very low cell counts to be combined to "rare" level

rare_title <- c('Dona','Lady','The Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer')
## Also reassign mlle ,ms and mme accordingly

DataBind$Title[DataBind$Title == 'Mlle']  <-'Miss'

DataBind$Title[DataBind$Title == 'Ms']  <-'Miss'

DataBind$Title[DataBind$Title == 'Mme']  <-'Mrs'

DataBind$Title[DataBind$Title %in%  rare_title]  <-'Rare Title'
##Show title counts by sex again

table(DataBind$Sex,DataBind$Title)