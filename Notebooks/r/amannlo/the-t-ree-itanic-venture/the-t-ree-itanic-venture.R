# activate librarys 

#tree making libraries:

library(rpart) 

library(party) 

library(randomForest)

#SVM:

library(e1071)

#Graphics:

library(ggplot2)

#data manging

#library(dplyr)

# get files

train_full = read.csv('../input/train.csv')

test  = read.csv('../input/test.csv')
str(train_full)

summary(train_full)

head(train_full)
#1)

train_full$Pclass=factor(train_full$Pclass,labels = c("first","second","third"))



#2)

#train_full$Cabin

train_full$Cabin_deck=substring(train_full$Cabin, 1, 1)

#this solution works but assumes that if a cell has more than one Cabin_num, the first is the correct one

train_full$Cabin_deck[train_full$Cabin_deck=='']="MiSS"

train_full$Cabin_deck=factor(train_full$Cabin_deck)



#i have tried:

#train_full$Cabin_num=substring(train_full$Cabin,2,4) but the data is dirty. so i need to tern to REGEX

train_full$Cabin_num=substring(train_full$Cabin, regexpr("(\\d)+", train_full$Cabin),regexpr("(\\d)+", train_full$Cabin)+2)

#this solution works but assumes that if a cell has more than one Cabin_num, the first is the correct one

train_full$Cabin_num[train_full$Cabin_num=='']="-1" #this changes all the '' cells to -1 char

train_full$Cabin_num=as.numeric(train_full$Cabin_num)  #the coaertion would eliminate any deck letters without numbers turning them to NA

train_full$Cabin_num[!is.finite(train_full$Cabin_num)]=-1 #finaly all NA are turned to -1





#3)

train_full$Name=as.character(train_full$Name)

pattern = "[A-Z][a-z]*\\." #see credit for the REGEX at the next frame

m = regexpr(pattern, train_full$Name)

train_full$title= regmatches(train_full$Name, m)

train_full$title= factor(train_full$title)



#4)

train_full$family_size=(train_full$SibSp+train_full$Parch)





str(train_full)

head(train_full)
DEV_ind=sample(nrow(train_full), floor(nrow(train_full)/2)) # gets indices of half the training data

DEV=train_full[DEV_ind,]

train=train_full[-DEV_ind,]
#form=formula(response ~ var1 + var2 + var3)

form=formula(Survived ~ Pclass + Sex + Age  + Fare+ Embarked+Cabin_deck+Cabin_num+family_size+title)





#tree

model = rpart(form, method="class", data=train)

print(model)

plot(model)

text(model)

post(model)



#library(party)

#party_model=as.party(model)

#plot(party_model)
model = ctree(form, data=train)

plot(model, main="Survival of people on the Titanic")

print(model)
model = ctree(form, data=train)

library("partykit")

spmodel<-as.simpleparty(model)

print(spmodel)

plot(spmodel)
#Look at: pred=predict(model,DEV) #this line generates a comfidance matrix, each col reperent a cathegorical

tree_pred=predict(model, DEV, type = "response") #this line would generate a single variable of the probability P

chose_DEV=(DEV$Survived==round(tree_pred))           #checks fore each cell, if chosen right 

true_accuracy=sum(chose_DEV==TRUE)/length(chose_DEV)          #calculates the accuracy

sprintf("the true accuracy of the randomly sampled tree on DEV set is: %s", true_accuracy)



tree_pred=predict(model, train, type = "response") 

chose_SELF=(train$Survived==round(tree_pred))  

optimized_accuracy=sum(chose_SELF==TRUE)/length(chose_SELF)

sprintf("the optimized accuracy of the randomly sampled tree on trainig set is: %s", optimized_accuracy)

# forest_model=randomForest(form, data=train ,importance=TRUE)
forest_model=randomForest(form, data=train ,importance=TRUE,na.action = na.exclude)
forest_pred=predict(forest_model, DEV, type = "response")

chose=(DEV$Survived==round(forest_pred))

#chose

accuracy=sum(chose==TRUE)/length(chose)

accuracy

sum(is.na(train_full$Survived))

sum(is.na(train_full$Pclass))

sum(is.na(train_full$Sex))

sum(is.na(train_full$Age))

sum(is.na(train_full$Fare))

sum(is.na(train_full$Embarked))

sum(is.na(train_full$Cabin_deck))

sum(is.na(train_full$Cabin_num))

sum(is.na(train_full$family_size))

sum(is.na(train_full$title))
DEV$Age[is.na(DEV$Age)]=121

train$Age[is.na(train$Age)]=121

# this is somthing that need to be added for later work

train_full$Age[is.na(train_full$Age)]=121



sum(is.na(train_full$Survived))

sum(is.na(train_full$Pclass))

sum(is.na(train_full$Sex))

sum(is.na(train_full$Age))

sum(is.na(train_full$Fare))

sum(is.na(train_full$Embarked))

sum(is.na(train_full$Cabin_deck))

sum(is.na(train_full$Cabin_num))

sum(is.na(train_full$family_size))

sum(is.na(train_full$title))





forest_model=randomForest(form, data=train ,importance=TRUE) #no NA to remove haha



forest_pred=predict(forest_model, DEV, type = "response") #this line would generate a single variable of the probability P

chose_DEV=(DEV$Survived==round(forest_pred))           #checks fore each cell, if chosen right 

true_accuracy=sum(chose_DEV==TRUE)/length(chose_DEV)          #calculates the accuracy

sprintf("the true accuracy of the random forest on DEV set is: %s", true_accuracy)



forest_pred=predict(forest_model, train, type = "response") 

chose_SELF=(train$Survived==round(forest_pred))  

optimized_accuracy=sum(chose_SELF==TRUE)/length(chose_SELF)

sprintf("the optimized accuracy of the random forest on training set is: %s", optimized_accuracy)


imp=importance(forest_model,type=1)

imp
form=formula(Survived ~ Pclass + Sex + Age  + Fare+ Embarked+Cabin_deck+Cabin_num+family_size+title)





best_true=true_accuracy

best_model=forest_model

iter_n=10



for (i in 1:iter_n){

    #resplit the training data

    DEV_ind=sample(nrow(train_full), floor(nrow(train_full)/2)) # gets indices of half the training data

    DEV=train_full[DEV_ind,]

    train=train_full[-DEV_ind,]

    

    forest_model=randomForest(form, data=train ,importance=FALSE) #no NA to remove haha



    forest_pred=predict(forest_model, DEV, type = "response") #this line would generate a single variable of the probability P

    chose_DEV=(DEV$Survived==round(forest_pred))           #checks fore each cell, if chosen right 

    true_accuracy=sum(chose_DEV==TRUE)/length(chose_DEV)          #calculates the accuracy

    

    forest_pred=predict(forest_model, train, type = "response") 

    chose_SELF=(train$Survived==round(forest_pred))  

    train_accuracy=sum(chose_SELF==TRUE)/length(chose_SELF)

       

    if (train_accuracy>=0.9){

        if (true_accuracy>best_true){

           best_model=forest_model

           best_true=true_accuracy

           best_train=train_accuracy 



        }

    }

    else{

        next

    }

    

}

sprintf("found a better model! with training accuracy: %s", best_train)

sprintf("and true accuracy: %s", best_true)







#1. rearrange the test file so to include all the variables I have included and computed



#1)

test$Pclass=factor(test$Pclass,labels = c("first","second","third"))



#2)

test$Cabin_deck=substring(test$Cabin, 1, 1)

test$Cabin_deck[test$Cabin_deck=='']="MiSS"

test$Cabin_deck=factor(test$Cabin_deck)



test$Cabin_num=substring(test$Cabin, regexpr("(\\d)+", test$Cabin),regexpr("(\\d)+", test$Cabin)+2)

test$Cabin_num[test$Cabin_num=='']="-1" #this changes all the '' cells to -1 char

test$Cabin_num=as.numeric(test$Cabin_num)  #the coaertion would eliminate any deck letters without numbers turning them to NA

test$Cabin_num[!is.finite(test$Cabin_num)]=-1 #finaly all NA are turned to -1





#3)

test$Name=as.character(test$Name)

pattern = "[A-Z][a-z]*\\." #see credit for the REGEX at the next frame

m = regexpr(pattern, test$Name)

test$title= regmatches(test$Name, m)

test$title= factor(test$title)



#4)

test$family_size=(test$SibSp+test$Parch)



#5)

test$Age[is.na(test$Age)]=121







head(test,10)
#forest_pred=predict(forest_model, test, type = "response")
#crudly check what are the levels of both variables. got any nicer way to do it?

levels(test$Pclass)

levels(train$Pclass)

levels(test$Sex)

levels(train$Sex)

levels(test$Embarked)

levels(train$Embarked)

levels(test$Cabin_deck)

levels(train$Cabin_deck)

levels(test$title)

levels(train$title)
levels(test$Embarked) = levels(train$Embarked)

levels(test$Cabin_deck) = levels(train$Cabin_deck)

levels(test$title) = levels(train$title)
#form=formula(Survived ~ Pclass + Sex + Age  + Fare+ Embarked+Cabin_deck+Cabin_num+family_size+title)

#forest_model=randomForest(form, data=train_full ,importance=TRUE)



#2. make a prediction besed on the optimized tree I found earlier

forest_pred=predict(best_model, test, type = "response")



#3. compute a new Survived variable,based on the prediction, and bind it. 

Survived=round(forest_pred)

test=cbind(test,Survived)



submission=cbind(test$PassengerId,Survived)

head(submission)

#4. submit.

write.csv(submission, file ='sub.csv', row.names = F)
