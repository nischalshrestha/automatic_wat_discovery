#Importing the Training Dataset

train=read.csv("../input/train.csv")
print(head(train))
summary(train)

nrow(train)

ncol(train)

colnames(train)
#Since Cabin  has more than 70% of the data unknown , Using it will be pointless. Moreover with such less  data , replacing with mean and medium won't beuseful

#Survival of a person doesn't depends on the location the person embarked his/her journey. Hence omitting this too.



train_refined=train[,-c(11,12)]

colnames(train_refined)

#Replaceing NA's in Age with mean or median



boxplot(train_refined$Age)

hist(train_refined$Age)







#Removing Outliers

age_mean=mean(train_refined$Age,na.rm=TRUE)

train_refined$Age[is.na(train_refined$Age)]=age_mean

summary(train_refined)
#Prediction 

library(randomForest)

colnames(train_refined)

train_refined=train_refined[,c(1,2,3,5,6,7,8,10)]
nrow(train_refined)

train_data=train_refined[1:625,]

test_data=train_refined[626:891,]

target=test_data[,2]

as.factor(target)->target

test_data=test_data[,-c(2)]

test_data

class(train_data$Survived)

as.factor(train_data$Survived)->train_data$Survived
#'PassengerId' 'Pclass'  'Sex' 'Age' 'SibSp' 'Parch' 'Fare'



model_randomForest=randomForest(Survived ~ PassengerId + Pclass + Sex + Age +  SibSp + Parch + Fare, data = train_data)

model_randomForest

pred = predict(model_randomForest,test_data)

print(pred)

data.frame(target,pred)->accuracy_tab
accuracy_tab$target==accuracy_tab$pred->accuracy_vec

count=0

for(i in 1:nrow(accuracy_tab))

{

  if(accuracy_vec[i]==FALSE)

{

  count=count+1

}

  }



Error = (count/nrow(accuracy_tab))*100

Accuracy = 100 -Error

paste("Accuracy : ",Accuracy)
test <- read.csv("../input/test.csv")

summary(test)
#Fare has one NA . Hence omitting it

test=test[!is.na(test$Fare),]
test=test[,-c(3,8,10,11)]

colnames(test)
test_age_mean=mean(test$Age,na.rm=TRUE)

test$Age[is.na(test$Age)]=test_age_mean

summary(test)
Survived = predict(model_randomForest,test)

test$PassengerId->PassengerId

data.frame(PassengerId,Survived) -> Predicted_Results
write.csv(Predicted_Results,file='survival_prediction.csv',row.names=F)
