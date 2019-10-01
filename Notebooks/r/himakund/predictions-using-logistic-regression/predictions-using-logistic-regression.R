library(Amelia)

library(checkmate)



system("ls ../input")

train <- read.csv("../input/train.csv")

test  <- read.csv("../input/test.csv")

PassID<-test$PassengerId

MT1<-anyMissing(train)

MT2<-anyMissing(test)

if(MT1 == "TRUE"){print("The Train data set has missing values")}

if(MT2 == "TRUE"){print("The Test data set has missing values")}

split.screen(c(1,2))

screen(1)

missmap(train, main = "Missing values vs observed")

screen(2)

missmap(test, main = "Missing values vs observed")

train <- subset(train,select=c(2,3,5,6,7,8,12)) 

test<-subset(test,select=c(2,4,5,6,7,11))

train$Age[is.na(train$Age)] <- mean(train$Age,na.rm=T)

test$Age[is.na(test$Age)] <- mean(test$Age,na.rm=T)

fitmodel <- glm(Survived ~.,family=binomial(link='logit'),data=train)

summary(fitmodel)

fitted.results <- predict(fitmodel,newdata=test)

fitted.results <- ifelse(fitted.results > 0.5,1,0)

result<-fitted.results

Predictions<-data.frame(result)

Pred<-data.frame("PassengerId"=PassID,"Survived"=Predictions)

names(Pred)[2]<-"Survived"

write.csv(Pred,"output.csv",row.names=F)