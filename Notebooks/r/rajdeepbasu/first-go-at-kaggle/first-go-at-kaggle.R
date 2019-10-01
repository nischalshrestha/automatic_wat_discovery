library(dplyr)

library(ggplot2)

library(ROCR)
train = read.csv('../input/train.csv')

test = read.csv('../input/test.csv')



summary(train)
train$Age = ifelse(!is.na(train$Age),

                   train$Age,

                   ifelse(train$Sex == 'male', 

                          mean(train$Age[train$Sex =='male'],na.rm = TRUE), 

                          mean(train$Age[train$Sex == 'female'],na.rm = TRUE)))



model = glm(Survived ~ Pclass + Sex + Age + SibSp, data = train,

           family = binomial)

summary(model)
prediction(predict(model,train,type = 'response') ,train$Survived) %>% 

    performance('tpr','fpr') %>%

    plot(colorize = TRUE)
table(predict(model,train,type = 'response') > 0.41, train$Survived)
summary(test)
test$Age = ifelse(!is.na(test$Age),

                   test$Age,

                   ifelse(test$Sex == 'male', 

                          mean(test$Age[test$Sex =='male'],na.rm = TRUE), 

                          mean(test$Age[test$Sex == 'female'],na.rm = TRUE)))

summary(test)
test$survived<-predict(model,test, type = 'response')

# set cutoff, >.55 survived, <.55 dead (Credit - HaoleiFang)

cutoff<-.55

sur<-vector()

for(i in 1:nrow(test)){

  if(test$survived[i] < cutoff) {sur[i] = 0} else

  {sur[i] = 1}

}

test$survived<-sur

test$survived