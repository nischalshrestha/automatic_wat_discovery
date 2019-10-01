library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(mice) #filling missing values

library(ROCR) #ROC curves



system("ls ../input")



TRAIN <- read.csv("../input/train.csv")

TEST  <- read.csv("../input/test.csv")



features <- c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked")
N.train <- (dim(TRAIN)[1])

pct1 <- 0.9



set.seed(110)                           #for reproducability

idx1 <- sample(1:N.train, N.train*pct1) #training set

#idx1 <- 1:N.train*pct1                  #reproducable, non-random training set...

idx2 <- setdiff(1:N.train, idx1)        #cv set
#training

D <- TRAIN[,features]

L <- TRAIN$Survived



log.model <- glm(formula=L[idx1]~.,data=D[idx1,],family=binomial(link="logit"))

#cross-validation

L0.predict <- unname(predict(log.model,newdata=D[idx2,],type="response"))

cv.pred <- prediction(L0.predict,L[idx2])

cv.roc <- performance(cv.pred,measure="tpr",x.measure="fpr")

cv.auc <- performance(cv.pred,measure="auc")@y.values[1]



plot(cv.roc); abline(a=0, b= 1)

cv.auc
D1 <- TEST[,features]

L1.predict <- unname(predict(log.model,newdata=D1,type="response"))

summary(L1.predict)
summary(D1)



#complete data, because we have missing values for our logit model that will give NA's



D2.mice<- mice(D1,printFlag=F)

D2 <- complete(D2.mice)

summary(D2)



L2.predict <- unname(predict(log.model,newdata=D2,type="response"))

summary(L2.predict)

#output

write.csv(cbind(TEST$ID,L2.predict),file="output.csv",row.names=F)