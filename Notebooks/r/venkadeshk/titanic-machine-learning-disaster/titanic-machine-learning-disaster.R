# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
#read the data

#na.strings=c("") is used to each missing values coded as NAs.

raw.data<-read.csv("train.csv",header = T, na.strings = c(""))

#str() function is used to know the structure of our data

str(raw.data)

#summary() function is used to know the summary of each variables

summary(raw.data)

#sapply used to find any missing and unique values in data

sapply(raw.data,function(x) sum(is.na(x)))

sapply(raw.data,function(x) length(unique(x)))

#Amelia package is used to know how the missing values mixed in data

install.packages("Amelia",dependencies = T)

library(Amelia)

missmap(raw.data,main = "Missing Values VS Observed")

#the variable cabin and passengerId is neglegible.

#subset() function usedto select the relevant columns only

data<-subset(raw.data,select = c(2,3,5,6,7,8,10,12))

#data$Age missing values are replaced by the mean of data$Age

data$Age[is.na(data$Age)]<-mean(data$Age,na.rm = T)

#treating missing values in Embarked variable

data<-data[!is.na(data$Embarked),]

rownames(data)<-NULL

#MODEL FITTING

#split the data into two sets

train<-data[1:800,]

test<-data[801:889,]

#fitting the model

model<-glm(Survived~.,family =binomial(link=logit),data = train)

summary(model)

#Prediction of model

fitted.results<-predict(model,newdata = subset(test,select = c(2,3,4,5,6,7,8)),type = 'response')

fitted.results<-ifelse(fitted.results>0.5,1,0)



 #getting the ROC curve 

library(ROCR)

p<-predict(model,newdata = subset(test),select=c(2,3,4,5,6,7,8),type = 'response')

pr<-prediction(p,test$Survived)

prf<-performance(pr,measure = "tpr",x.measure = "fpr")

plot(prf)