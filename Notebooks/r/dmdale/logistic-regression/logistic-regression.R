# This R script will run on our backend. You can write arbitrary code here!



#Import data

train_data<-read.csv("train.csv", header=T, na.strings=c("","NA"))

test_data<-read.csv("test.csv", header=T, na.strings=c("","NA"))



#Get mean age of data, exclude NAs.

mean(train_data$Age, na.rm=TRUE)



#Find missing data so we know what needs to get replaced

missing_train<-t(data.frame(sapply(train_data, function(x) sum(is.na(x)))))

missing_test<-t(data.frame(sapply(test_data, function(x) sum(is.na(x)))))



missing_train_plot<-missing_train[apply(missing_train, 1, function(row) {any(row > 10)}),]



barplot(missing_train_plot)







#find mean age of female and male. There is almost certainly a better way of replacing these NAs but I chose this simple method.

train_female<-subset(train_data,train_data$Sex=="female")

mean_female<-mean(train_female$Age, na.rm=TRUE)

train_male<-subset(train_data,train_data$Sex=="male")

mean_male<-mean(train_male$Age, na.rm=TRUE)



table(balanced_train$Sex)



#Replace age NAs with mean of ages.

train_data$Age[is.na(train_data$Age)&train_data$Sex=="female"]<-mean_female

train_data$Age[is.na(train_data$Age)&train_data$Sex=="male"]<-mean_male



test_data$Age[is.na(test_data$Age)&test_data$Sex=="female"]<-mean_female

test_data$Age[is.na(test_data$Age)&test_data$Sex=="male"]<-mean_male



#I found that the split between male and female was uneven. I decided to create a more evenly distrubted model by randomly subsetting the male portion.

set.seed(123)

split <- sample.split(train_male$Survived, SplitRatio = 0.54)

subm_train <- subset(train_male, split == T)



balanced_train<-rbind(train_female,subm_train)



#I randomly chose these variables. I should find a more substantial way of picking predictor variables. Use a decision tree. I did not scale any variables.

train_model<- glm(balanced_train$Survived~Sex+Pclass+SibSp,data = balanced_train, family = "binomial")

summary(balanced_train)



test_model<- predict(train_model, newdata = test_data, type = "response")



submission<-cbind(test_data$PassengerId,test_model)

submission<-data.frame(submission)

submission$test_model[submission$test_model<.5]<- 0

submission$test_model[submission$test_model>.5]<- 1





nms <- c("PassengerID", "Survived")

setnames(submission, nms)



write.csv(submission,"submission.csv")