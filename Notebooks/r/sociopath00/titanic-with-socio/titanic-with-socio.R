# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(randomForest) 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")

rfNews()

# Any results you write to the current directory are saved as output.
train <- read.table("../input/train.csv", header = T, sep = ",", stringsAsFactors = F)

test <- read.table("../input/test.csv", header = T, sep = ",", stringsAsFactors = F)
train$isTrain <- TRUE

test$isTrain <- FALSE



test$Survived <- NA
full <- rbind(train,test)

str(full)
full$Pclass<-as.ordered(full$Pclass)

full$Sex <- as.factor(full$Sex)

full$Embarked <- as.factor(full$Embarked)

str(full)
ageTrain <- full[is.na(full$Age),]

ageV <- full[is.na(full$Age)==FALSE,]



age.Formula <- "Age ~ Pclass + Sex + SibSp + Parch+ Fare"

age.Formula <- as.formula(age.Formula)

age.Model <- lm(age.Formula,ageV)



age.Predict<-predict(age.Model,ageTrain)

age.Predict<-round(age.Predict)

ageTrain$Age<-age.Predict

tail(ageTrain)
full<-rbind(ageTrain, ageV)



#dim(full)

full<- full[order(full$PassengerId),]

tail(full)
boxplot(train$Fare)

outBound <- boxplot.stats(train$Fare)$stats[5]

Fare.t<-full[full$Fare<=outBound,]

fare.formula<-"Fare ~ Pclass + Age + Sex + SibSp + Parch "

fare.formula<- as.formula(fare.formula)

fare.Model<-lm(fare.formula, Fare.t)



fare.Pred<-predict(fare.Model, full[1044,])

full[1044,]$Fare<- fare.Pred

full[1040:1045,]
train <- full[full$isTrain== TRUE,]

test <- full[full$isTrain== FALSE,]



train$Survived<- as.factor(train$Survived)
fml<- "Survived ~ Pclass + Sex + Age + SibSp + Parch + Embarked + Fare"

fml<-as.formula(fml)

titanic.model<-randomForest(fml,train, ntree=500, mtry=3, nodesize=0.01*nrow(train) )

Survived<-predict(titanic.model,test)
PassengerId<- test$PassengerId

op<-as.data.frame(PassengerId)



op$Survived <- Survived

write.csv(op, file= "Titanic_socio.csv", row.names = F)