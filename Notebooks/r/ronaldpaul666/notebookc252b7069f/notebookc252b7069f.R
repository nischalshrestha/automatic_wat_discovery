# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.

library(rpart) 

library(party) 

library(randomForest)

library(rpart.plot)  

#SVM:





#Graphics:

library(ggplot2)

#data manging

library(plyr);      # load plyr prior to dplyr to avoid warnings

library(caret);    

library(dplyr); 

train_full = read.csv('../input/train.csv')

test  = read.csv('../input/test.csv')

str(train_full)

summary(train_full)

head(train_full)

train_full$Pclass=factor(train_full$Pclass,labels = c("1st","2nd","3rd"))



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

form=formula(Survived ~ Pclass + Sex + Age  + Fare+ Embarked+Cabin_deck+Cabin_num+family_size+title)

model = rpart(form, method="class", data=train)

print(model)

plot(model)

text(model)

post(model)

model = ctree(form, data=train)

plot(model, main="Survival of people on the Titanic")

print(model)

tree_pred=predict(model, DEV, type = "response") #this line would generate a single variable of the probability P

chose_DEV=(DEV$Survived==round(tree_pred))           #checks fore each cell, if chosen right 

true_accuracy=sum(chose_DEV==TRUE)/length(chose_DEV)          #calculates the accuracy

sprintf("the true accuracy of the randomly sampled tree on DEV set is: %s", true_accuracy)



tree_pred=predict(model, train, type = "response") 

chose_SELF=(train$Survived==round(tree_pred))  

optimized_accuracy=sum(chose_SELF==TRUE)/length(chose_SELF)

sprintf("the optimized accuracy of the randomly sampled tree on trainig set is: %s", optimized_accuracy)

forest_model=randomForest(form, data=train ,importance=TRUE,na.action = na.exclude)

forest_pred=predict(forest_model, DEV, type = "response")

chose=(DEV$Survived==round(forest_pred))

#chose

accuracy=sum(chose==TRUE)/length(chose)

accuracy

orest_model=randomForest(form, data=train ,importance=TRUE) #no NA to remove haha



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

form=formula(Survived ~ Pclass + Sex + Embarked+family_size+title)

# my model was over fitting, so I read in some kernel it might be a good idea to reduce variables 



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

test$Pclass=factor(test$Pclass,labels = c("1st","2nd","3rd"))



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

levels(test$Embarked) = levels(train$Embarked)

levels(test$Cabin_deck) = levels(train$Cabin_deck)

levels(test$title) = levels(train$title)

forest_pred=predict(best_model, test, type = "response")



#3. compute a new Survived variable,based on the prediction, and bind it. 

Survived=round(forest_pred)

test=cbind(test,Survived)



submission=cbind(test$PassengerId,test$Survived)

colnames(submission)=c("PassengerId","Survived")

submission=as.data.frame(submission)

submission$Survived[is.na(submission$Survived)]=0

#4. submit.

write.csv(submission, file ="sub.csv", row.names = FALSE)



submission