#include libraries

library(ggplot2) #library for producing plots



system("ls ../input")

#load the training data:

df=read.csv("../input/train.csv",stringsAsFactors=FALSE) 



#Display a summary of all the variables and their type

str(df)
#Change the survived variable to make summary tables prettier:

df$Survived=factor(df$Survived, 

                   levels=c(0,1),

                   labels =c("died","lived"))
df$Sex=factor(df$Sex) #change the gender variable to a factor

table(df$Survived,df$Sex) #See a summary mortality by gender
options(repr.plot.width=5, repr.plot.height=3)#Plot size Options



#Determine age distribution

age_range=cut(df$Age, seq(0,100,10)) #Sub-divide the ange range into 10 year sections

qplot(age_range, xlab="Age Range", main="Age distribution on the Titanic") #plot age distributon



#Determine survival percentage:

ggplot(df, aes(x=Age, fill=Survived))+

  geom_histogram(binwidth = 5,position="fill")+

  ggtitle("Survival percentage amongst the age groups")



#check percentage of unknown age passengers:

print("Survival rate of passengers who's age is unknown:")

table(df$Survived[is.na(df$Age)]) 



#Replace the missing age entries with the average age

df$Age[is.na(df$Age)]=mean(df$Age, na.rm=TRUE)
#Explore embark location

df$Embarked[df$Embarked==""]="S" #replace missing values with majority (S), highest chance of being right

df$Embarked=factor(df$Embarked, levels=c("S","C","Q")) #Set as factor in order of S->C->Q

table(df$Survived,df$Embarked) #show summary table of survival chances
print("Survival of people who have parents/children aboard")

table(df$Survived,df$Parch) #parent children



print("Survival of people who have siblings/spouses aboard")

table(df$Survived,df$SibSp) #siblings/spouse
print("Survival rate against class")

table(df$Survived,df$Pclass) #Summary of passenger vs. class



#Show the histogram of the log-fare

hist(log(df$Fare)) #histogram, which looks more normal than the skewed Fare distribution



#Some values have Fare=0, this is not good for the log-fare, so we change these values with

#the mean of the log-fare

df$logfare=log(df$Fare)

df$logfare[df$Fare==0] = mean(log( df$Fare[df$Fare>0])  )



#Show the survival as a function of log Fare

ggplot(df, aes(x=log(Fare), fill=Survived))+

  geom_histogram(binwidth = 0.5,position="fill")+

  ggtitle("Survival likelyhood vs. log-fare")
library(caret) #

set.seed(3456) #set a seed for reproducible results



trainIndex <- createDataPartition(df$Survived, p = .8,list=FALSE)

df_train=df[trainIndex,]

df_test=df[-trainIndex,]
library(C50) #Import the C5.0 library



mc5=C5.0(Survived~Sex+Age+Embarked+logfare+Pclass+SibSp+Parch,

        data=df_train) #Train model



newval=predict(mc5, newdata=df_test) #Predict new values

confusionMatrix(newval, df_test$Survived) #Evaluate the perfromance
error_cost=matrix(c(0, 5, 5, 0), nrow = 2)

mc5=C5.0(Survived~Sex+Age+Embarked+logfare+Pclass+SibSp+Parch,

         data=df_train,

         costs = error_cost)



newval=predict(mc5, newdata=df_test) #Predict new values

confusionMatrix(newval, df_test$Survived) #Evaluate the perfromance
mc5=C5.0(Survived~Sex+Age+Embarked+logfare+Pclass+SibSp+Parch,

         data=df_train,

         trials=5) #Number of boosting iterations



newval=predict(mc5, newdata=df_test) #Predict new values

confusionMatrix(newval, df_test$Survived) #Evaluate the perfromance
#Importing testing data

dft=read.csv("../input/test.csv", stringsAsFactors = FALSE)



#Mandatory Data Manipulation prior to running the model:

#Age:

dft$Age[is.na(dft$Age)]=mean(dft$Age, na.rm=TRUE) 



#Embarked:

dft$Embarked[dft$Embarked==""]="S"

dft$Embarked=factor(dft$Embarked, levels=c("S","C","Q"))



#Missing Fare

dft$logfare=log(dft$Fare)

dft$logfare[is.na(dft$Fare)]= mean(log( dft$Fare[dft$Fare>0]), na.rm=TRUE  )







newval=predict(mc5, newdata=dft) #Predict the test data

dft$Survived=newval #add the predicted survival rates to dft

levels(dft$Survived)= c(0,1) #change the "survived" variable from died/lived to 0/1 as requested



write.csv(dft[c("PassengerId","Survived")], #select column names 

          file="submission.csv", #output file name

          row.names=FALSE, #do not print row names

          quote=FALSE) #do not encapsulate data by quotation marks