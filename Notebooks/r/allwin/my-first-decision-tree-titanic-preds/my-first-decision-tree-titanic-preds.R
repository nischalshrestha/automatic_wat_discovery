setwd("C:/Users/aall/Desktop/R Prog/Titanic") ####Setting Work Directory

prod = read.csv("train.csv",stringsAsFactors = F) ### Importing Files

test = read.csv("test.csv",stringsAsFactors = F)  ### Importing Files



######Binding rows #########

library(gtools)

indata = smartbind(prod,test)

str(indata)

head(indata)

tail(indata)



####Passenger Name Title Split######

library('dplyr')

indata$title = gsub('(.*, )|(\\..*)',(""),indata$Name)

str(indata$title)

table(indata$Sex,indata$title)

indatacommon = c('Miss', 'Mrs', 'Mr','Master')

indatanoble = c('Don', 'Dona','Sir','the Countess', 'Lady', 'Jonkheer')

##########Family Size #############

library(ggplot2)

indata$size = indata$SibSp + indata$Parch + 1

ggplot(indata[1:891, ],aes(x = size, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  scale_x_continuous(breaks=c(1:11)) +

  labs(x = 'Family Size')



#####Spliting the production data ######

install.packages("caToolS")

library(caTools)

library(rpart)

library(rpart.plot)

set.seed(560)

splitvar = sample.split(prod$Survived, SplitRatio = .7)

traindsn = subset(prod, splitvar == TRUE)

testdsn  = subset(prod, splitvar == FALSE)

dtmod = rpart(Survived ~ Age + Sex + SibSp + Parch + Pclass + Fare,data = traindsn, method= 'class',cp = 0)

prp(dtmod)

dtmod

predict(dtmod,type = 'class')

preds = predict(dtmod,type = 'class')

tt = table(traindsn$Survived,preds)

sum(diag(tt))/sum(tt)

dtmodtest = rpart(Survived ~ Age + Sex + SibSp + Parch + Pclass + Fare,data = testdsn, method= 'class',cp = 0)

prp(dtmodtest)

dtmodtest

predict(dtmodtest,type = 'class')

predstest = predict(dtmodtest,type = 'class')

tt2 = table(testdsn$Survived,predstest)

sum(diag(tt2))/sum(tt2)

sum(diag(tt))/sum(tt)



###### Output Prediction ######

prediction <- predict(dtmod, test)

output <- data.frame(PassengerID = test$PassengerId, Survived = prediction)

write.csv(output, file = 'output_DTmod.csv',row.names=FALSE)