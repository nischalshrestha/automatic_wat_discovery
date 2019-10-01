# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



require(caret) 

require(caretEnsemble)

require(lattice)

require(rpart)

require(randomForest)

require(ada)

require(plyr)

require(adabag)

library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

setwd('D:/Narendra/!1Data Science/!5 Data Science Project/!1 Classification/Titanic Machine Learning from Disaster (Kaggle)/')

training=read.csv(file='!Original Data/train.csv',header=T,sep=",",na.strings=c("NA"," ",""))

dim(training) ### Check the dimension

str(training) ### Check the class type of variables

summary(training) #### Check the Summary on data



test=read.csv(file='!Original Data/test.csv',header=T,sep=",",na.strings=c("NA"," ",""))

dim(test)

str(test)

test$Survived=0



#### Combind the data using Rbind

Titanic_data=rbind(training,test)

summary(Titanic_data)

dim(Titanic_data)



########## Update the missing values using the "BagImpute" (Preprocessing) #############################



Titanic_data_preprocess=preProcess(Titanic_data[,c("PassengerId","Age","SibSp","Parch","Fare")],method=c("bagImpute"))

Titanic_data_preprocess

summary(Titanic_data_preprocess)

Titanic_data_preprocess$method

Titanic_data_preprocess_pred=predict(Titanic_data_preprocess,Titanic_data[,c("PassengerId","Age","SibSp","Parch","Ticket","Fare")])

summary(Titanic_data_preprocess_pred)

Titanic_data$Age=Titanic_data_preprocess_pred$Age

Titanic_data$Fare=Titanic_data_preprocess_pred$Fare

summary(Titanic_data)



Titanic_data$Embarked[is.na(Titanic_data$Embarked=='')]="S"

Titanic_data$Name=as.character(Titanic_data$Name)



######### Extract the Titles in name filed like ###############



require(stringr)

extract_Title=function(x){

  

  Title=str_trim(strsplit(x,split='[,.]')[[1]][2])

  return(Title)

}



Titanic_data$Title=sapply(Titanic_data$Name,FUN=extract_Title)



Titanic_data$Title=as.factor(Titanic_data$Title)

summary(Titanic_data)

levels(Titanic_data$Title)



### Renaming the titles 

Titanic_data$Title[Titanic_data$Title=='Capt']='Mr'

Titanic_data$Title[Titanic_data$Title=='Col']='Mr'

Titanic_data$Title[Titanic_data$Title=='Dona']='Mr'

Titanic_data$Title[Titanic_data$Title=='Don']='Mr'

Titanic_data$Title[Titanic_data$Title=='Dr']='Sir'

Titanic_data$Title[Titanic_data$Title=='Jonkheer']='Mr'

Titanic_data$Title[Titanic_data$Title=='Lady']='Miss'

Titanic_data$Title[Titanic_data$Title=='Major']='Mr'

Titanic_data$Title[Titanic_data$Title=='Mlle']='Miss'

Titanic_data$Title[Titanic_data$Title=='Ms']='Miss'

Titanic_data$Title[Titanic_data$Title=='Sir']='Mr'

Titanic_data$Title[Titanic_data$Title=='Mme']='Miss'

Titanic_data$Title[Titanic_data$Title=='Rev']='Mr'

Titanic_data$Title[Titanic_data$Title=='the Countess']='Miss'



##### Save the files after PreProcess and Feature engineering

write.csv(training_data,file="train.csv",row.names=F)

write.csv(test_data,file="test.csv",row.names=F)

#### Load Files

train=read.csv('train.csv',header=T,sep=",")

summary(train)

dim(train)

test=read.csv('test.csv')

dim(test)

#### Combind the Data

datamerging=rbind(train,test)

summary(datamerging)

dim(datamerging)

#### Change the levels

datamerging$Survived=as.factor(datamerging$Survived)

datamerging$Pclass=as.factor(datamerging$Pclass)

datamerging$Ticket=as.factor(datamerging$Ticket)



summary(datamerging)

train1=datamerging[1:891,]

test1=datamerging[892:1309,]

### Explore the data

dim(train1)

class(train1)

str(train1)

dim(test)

summary(train1)

levels(train1$Survived)

levels(train1$Pclass)

levels(train1$Ticket)



####### Ensamble the Process ###############

set.seed(10000)

control1=trainControl(method="repeatedcv",repeats=3,savePredictions=TRUE,classProbs=TRUE)



al_list=c('gbm','rf','rpart','C5.0')





##### Changing the Levels on target variable 0,1 to X0 and X1

?make.names 

levels(train1$Survived)=make.names(levels(factor(train1$Survived)))



###### Build the model using Caretlist $ caretStack ########



set.seed(10000)

ens_model=caretList(Survived~.,data=train1,trControl=control1,methodList=al_list)

ens_model

ens_model_result=resamples(ens_model)

ens_model_result

summary(ens_model_result)

x11()

dotplot(ens_model_result)

splom(ens_model_result)

final_ens_model=caretStack(ens_model,trControl=control1,method="rf",tuneLength=8)

final_ens_model

final_ens_model$models

final_ens_model$ens_model

final_ens_model$error

x11()

summary(final_ens_model)





##### Predict the train result in TEST ########



test1$Survived=predict(final_ens_model,test1)

summary(test1$Survived)

write.csv(test1[,c("PassengerId","Survived")],file="RForest_Model3aa_pred.csv",row.names=F)


