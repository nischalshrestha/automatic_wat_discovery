library('dplyr') # data manipulation

library('ggplot2') # Data Visualization

library('ggthemes') # Data Visualization



options(warn = -1)

# load train.csv

train <- read.csv('../input/train.csv', stringsAsFactors = F)

# load test.csv

test  <- read.csv('../input/test.csv', stringsAsFactors = F)

# combine them as a whole

test$Survived <- NA

full <- rbind(train,test)
# show first several rows of the data

head(full)


# check the data

str(full)
# Process Age Column



    # create a new data set age

age <- full$Age

n = length(age)

    # replace missing value with a random sample from raw data

set.seed(123)

for(i in 1:n){

  if(is.na(age[i])){

    age[i] = sample(na.omit(full$Age),1)

  }

}

    # check effect

par(mfrow=c(1,2))

hist(full$Age, freq=F, main='Before Replacement', 

  col='lightblue', ylim=c(0,0.04),xlab = "age")

hist(age, freq=F, main='After Replacement', 

  col='darkblue', ylim=c(0,0.04))
# Process Cabin Column to show number of cabins passenger has

cabin <- full$Cabin

n = length(cabin)

for(i in 1:n){

  if(nchar(cabin[i]) == 0){

    cabin[i] = 0

  } else{

    s = strsplit(cabin[i]," ")

    cabin[i] = length(s[[1]])

  }

} 

table(cabin)
# process fare column



# check missing

full$PassengerId[is.na(full$Fare)]
full[1044,]
ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], 

  aes(x = Fare)) +

  geom_density(fill = '#99d6ff', alpha=0.4) + 

  geom_vline(aes(xintercept=median(Fare, na.rm=T)),

    colour='red', linetype='dashed', lwd=1)


# we can see that fare is clustered around mode. we just repace the missing value with 

# median fare of according Pclass and Embarked



full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)

# process embarked column

embarked <- full$Embarked

n = length(embarked)

for(i in 1:n){

  if(embarked[i] != "S" && embarked[i] != "C" && embarked[i] != "Q"){

    embarked[i] = "S"

  }

}

table(embarked)
# number of survivals and nonsurvivals across different age

d <- data.frame(Age = age[1:891], Survived = train$Survived)

ggplot(d, aes(Age,fill = factor(Survived))) +

    geom_histogram()
# create bar chart to show relationship between survival rate and age intervals

cuts <- cut(d$Age,hist(d$Age,10,plot = F)$breaks)

rate <- tapply(d$Survived,cuts,mean)

d2 <- data.frame(age = names(rate),rate)

barplot(d2$rate, xlab = "age",ylab = "survival rate")
# create histgram to show effect of Sex on survival

ggplot(train, aes(Sex,fill = factor(Survived))) +

    geom_histogram(stat = "count")
# calculate survival rate

tapply(train$Survived,train$Sex,mean)
# extract title from Name 

# (here I process full data set but only plot title vs survival in train 

#    data set because there is no survival value for test data set)

n = length(full$Survived)

title = rep(NA,n)

for (i in 1:n){

  lastname = strsplit(full$Name[i],", ")[[1]][2]

  title[i] = strsplit(lastname,". ")[[1]][1]

}



# make a histogram of title v.s survival

d <- data.frame(title = title[1:891],Survived = train$Survived)

ggplot(d, aes(title,fill = factor(Survived))) +

    geom_histogram(stat = "count")
# count of title

table(title)
# survival rate

tapply(d$Survived,d$title,mean)
# replace rare titles to 'Rare'

title[title != 'Mr' & title != 'Miss' & title != 'Mrs' & title != 'Master'] <- 'Rare'

table(title)
# make a histogram

ggplot(train, aes(Pclass,fill = factor(Survived))) +

    geom_histogram(stat = "count")
# calculate survival rate

tapply(train$Survived,train$Pclass,mean)
# histogram of Parch

ggplot(train, aes(Parch,fill = factor(Survived))) +

    geom_histogram(stat = "count")
# histogram of SibSp

ggplot(train, aes(SibSp,fill = factor(Survived))) +

    geom_histogram(stat = "count")
# combine SibSp and Parch 

family <- full$SibSp + full$Parch

d <- data.frame(family = family[1:891],Survived = train$Survived)

ggplot(d, aes(family,fill = factor(Survived))) +

    geom_histogram(stat = "count")
tapply(d$Survived,d$family,mean)
# create histogram

d <- data.frame(Cabin = cabin[1:891],Survived = train$Survived)

ggplot(d, aes(Cabin,fill = factor(Survived))) +

    geom_histogram(stat = "count")
# calculate survival rate

tapply(d$Survived,d$Cabin,mean)
# make a histogram

ggplot(train, aes(Fare,fill = factor(Survived))) +

    geom_histogram()
# calculate

cuts <- cut(train$Fare,hist(train$Fare,10,plot = F)$breaks)

rate <- tapply(train$Survived,cuts,mean)

d <- data.frame(fare = names(rate),rate)

barplot(d$rate, xlab = "fare",ylab = "survival rate")
# make histogram

d <- data.frame(Embarked = embarked[1:891], Survived = train$Survived)

ggplot(d, aes(Embarked,fill = factor(Survived))) +

    geom_histogram(stat = "count")
# make table

tapply(train$Survived,train$Embarked,mean)
# response variable

f.survived = train$Survived
# feature

# 1. age

f.age = age[1:891]    # for training

t.age = age[892:1309]  # for testing
# 2. fare

f.fare = full$Fare[1:891]

t.fare = full$Fare[892:1309]
# 3. cabin

f.cabin = cabin[1:891]

t.cabin = cabin[892:1309]



# 4. title

f.title = title[1:891]

t.title = title[892:1309]



# 5. family

family <- full$SibSp + full$Parch

f.family = family[1:891]

t.family = family[892:1309]



# 6. plcass

f.pclass = train$Pclass

t.pclass = test$Pclass



# 7. sex

f.sex = train$Sex

t.sex = test$Sex



# 8. embarked

f.embarked = embarked[1:891]

t.embarked = embarked[892:1309]
# construct training data frame

new_train = data.frame(survived = f.survived, age = f.age, fare = f.fare , sex = f.sex, 

       embarked = f.embarked ,family = f.family ,title = f.title ,cabin =  f.cabin, pclass= f.pclass)
# logistic regression

fit_logit <- glm(factor(survived) ~ age + fare + sex + embarked + family 

                 + title + cabin + pclass,data = new_train,family = binomial)

    # predicted result of regression

ans_logit = rep(NA,891)

for(i in 1:891){

  ans_logit[i] = round(fit_logit$fitted.values[[i]],0)

}

    # check result

mean(ans_logit == train$Survived)

table(ans_logit)
# random forest

library('randomForest')



set.seed(123)

fit_rf <- randomForest(factor(survived) ~ age + fare + sex + embarked + family 

                 + title + cabin + pclass,data = new_train)



    # predicted result of regression

rf.fitted = predict(fit_rf)

ans_rf = rep(NA,891)

for(i in 1:891){

  ans_rf[i] = as.integer(rf.fitted[[i]]) - 1

}

    # check result

mean(ans_rf == train$Survived)

table(ans_rf)

# decision tree

library(rpart)



fit_dt <- rpart(factor(survived) ~ age + fare + sex + embarked + family 

                 + title + cabin + pclass,data = new_train)



    # predicted result of regression

dt.fitted = predict(fit_dt)

ans_dt = rep(NA,891)

for(i in 1:891){

  if(dt.fitted[i,1] >= dt.fitted[i,2] ){

    ans_dt[i] = 0

  } else{

    ans_dt[i] = 1

  }

}

    # check result

mean(ans_dt == train$Survived)

table(ans_dt)

# svm

library(e1071)



fit_svm <- svm(factor(survived) ~ age + fare + sex + embarked + family 

                 + title + cabin + pclass,data = new_train)



    # predicted result of regression

svm.fitted = predict(fit_svm)

ans_svm = rep(NA,891)

for(i in 1:891){

  ans_svm[i] = as.integer(svm.fitted[[i]]) - 1

}

    # check result

mean(ans_svm == train$Survived)

table(ans_svm)

# logistic

a = sum(ans_logit ==1 & f.survived == 1)

b = sum(ans_logit ==1 & f.survived == 0)

c = sum(ans_logit ==0 & f.survived == 1)

d = sum(ans_logit ==0 & f.survived == 0)

data.frame(a,b,c,d)
# Random Forest

a = sum(ans_rf ==1 & f.survived == 1)

b = sum(ans_rf ==1 & f.survived == 0)

c = sum(ans_rf ==0 & f.survived == 1)

d = sum(ans_rf ==0 & f.survived == 0)

data.frame(a,b,c,d)
# Decision Tree

a = sum(ans_dt ==1 & f.survived == 1)

b = sum(ans_dt ==1 & f.survived == 0)

c = sum(ans_dt ==0 & f.survived == 1)

d = sum(ans_dt ==0 & f.survived == 0)

data.frame(a,b,c,d)
# SVM

a = sum(ans_svm ==1 & f.survived == 1)

b = sum(ans_svm ==1 & f.survived == 0)

c = sum(ans_svm ==0 & f.survived == 1)

d = sum(ans_svm ==0 & f.survived == 0)

data.frame(a,b,c,d)
# construct testing data frame

test_data_set <- data.frame(age = t.age, fare = t.fare, sex = t.sex, embarked = t.embarked, 

                            family = t.family, title = t.title,cabin =  t.cabin, pclass = t.pclass)

# make prediction

svm_predict = predict(fit_svm,newdata = test_data_set )

ans_svm_predict = rep(NA,418)

for(i in 1:418){

  ans_svm_predict[i] = as.integer(svm_predict[[i]]) - 1

}

table(ans_svm_predict)

# create a csv file for submittion

d<-data.frame(PassengerId = test$PassengerId, Survived = ans_svm_predict)

write.csv(d,file = "TitanicResult.csv",row.names = F)