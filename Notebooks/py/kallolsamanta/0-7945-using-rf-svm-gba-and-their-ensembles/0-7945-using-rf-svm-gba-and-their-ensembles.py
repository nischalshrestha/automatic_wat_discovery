#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


#read the files
titanicTrain=pd.read_csv("../input/train.csv")
titanicTest=pd.read_csv("../input/test.csv")


# In[ ]:


#display first five row of train data
titanicTrain.head()


# In[ ]:


#display first five row of the test data
titanicTest.head()


# In[ ]:


#find total no. of records in train data
titanicTrain.PassengerId.count()


# In[ ]:


#removing unwanted columns from the dataframe
#name, passanger id, cabin doesn't make any sense for prediction
#Same goes for embarked
titanicTrain=titanicTrain.drop('Ticket',1)
titanicTrain=titanicTrain.drop('Fare',1)
titanicTrain=titanicTrain.drop('Cabin',1)
titanicTrain=titanicTrain.drop('PassengerId',1)
titanicTrain=titanicTrain.drop('Embarked',1)
titanicTrain=titanicTrain.drop('Name',1)


# In[ ]:


#Same thing will be done for test data as well
titanicTest=titanicTest.drop('Ticket',1)
titanicTest=titanicTest.drop('Fare',1)
titanicTest=titanicTest.drop('Cabin',1)
titanicTest=titanicTest.drop('PassengerId',1)
titanicTest=titanicTest.drop('Embarked',1)
titanicTest=titanicTest.drop('Name',1)


# In[ ]:


#now check for train data
titanicTrain.head()


# In[ ]:


#test data
titanicTest.head()


# In[ ]:


#now we found that ticket is also not required
#let's remove tickets as well
#titanicTrain=titanicTrain.drop('Ticket',1)
#titanicTest=titanicTest.drop('Ticket',1)


# In[ ]:


#now we found that there are two categories in sex(e.g. male and female) We can create another category as child based on age.
#so first let's check if any null value present in age and sex column
print (sum(titanicTrain.Sex.isnull()))
print (sum(titanicTrain.Age.isnull()))


# In[ ]:


#we found 177 null values for age column should we replace those with some value or can we go for removing those columns
#lets check if the data is imbalanced
from collections import Counter
Counter(titanicTrain.Survived)


# In[ ]:


#lets check the same for those 177 rows
nullRaw=pd.read_csv("../input/train.csv")
nullData=nullRaw[(nullRaw.Age.isnull())]


# In[ ]:


#Lets check
Counter(nullData.Survived)


# In[ ]:


#check for null data head
nullData.head()


# In[ ]:


#create another dataframe where age is not null
notNullData=nullRaw[(nullRaw.Age.notnull())]


# In[ ]:


#check the head for notNullData
notNullData.head()


# In[ ]:


#now I will apply random forest methode to replace missing age values
#so we need to remove unwanted columns
notNullData=notNullData.drop('Survived',1)
notNullData=notNullData.drop('PassengerId',1)
notNullData=notNullData.drop('Name',1)
notNullData=notNullData.drop('Ticket',1)
notNullData=notNullData.drop('Cabin',1)


# In[ ]:


#calling sklearn library to apply random forest
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


#importing necessary files
from sklearn.model_selection import train_test_split


# In[ ]:


modelRF=RandomForestRegressor()


# In[ ]:


notNullData.head()


# In[ ]:


notNullData.iloc[:,[0,1,3,4,5,6]].head()


# In[ ]:


#converting into dummies for sex and Embarked
notNullData.Sex=pd.get_dummies(notNullData.Sex)
notNullData.Embarked=pd.get_dummies(notNullData.Embarked)


# In[ ]:


#passing parameters in random forest classifier for creating the model
modelRF.fit(np.array(notNullData.iloc[:,[0,1,3,4,5,6]]),notNullData.iloc[:,2])


# In[ ]:


nullData=nullData.drop('Survived',1)
nullData=nullData.drop('PassengerId',1)
nullData=nullData.drop('Name',1)
nullData=nullData.drop('Ticket',1)
nullData=nullData.drop('Cabin',1)


# In[ ]:


nullData=nullData.drop('Age',1)


# In[ ]:


nullData.head()


# In[ ]:


#converting into dummies for sex and embarked
nullData.Embarked=pd.get_dummies(nullData.Embarked)
nullData.Sex=pd.get_dummies(nullData.Sex)


# In[ ]:


#check for nullData
nullData.head()


# In[ ]:


#remove Survived and Age for prediction
#nullData=nullData.drop('Survived',1)
#nullData=nullData.drop('Age',1)


# In[ ]:


#check again
#nullData.head()


# In[ ]:


#storing the predicted value
p=modelRF.predict(nullData)


# In[ ]:


#assigning the predicted value in place of NA
nullData['Age']=p


# In[ ]:


#checking again
nullData.head()


# In[ ]:


#check for not null data head
notNullData.head()


# In[ ]:


#check the titanic train
titanicTrain.head()


# In[ ]:


#titanicTrain.iloc[1,3]==nullData
#pd.isnull(titanicTrain.iloc[1,3])==False
p[0]


# In[ ]:


x=0
for i in range(0,890):
    if(pd.isnull(titanicTrain.iloc[i,3])==True):
        titanicTrain.iloc[i,3]=p[x]
        x=x+1


# In[ ]:


#check for titanicData if it got replaced
sum(titanicTrain.Age.isnull())


# In[ ]:


#so it NA got replaced
#lets see titanic data again for further analysis
titanicTrain.head()


# In[ ]:


#adding a new column
titanicTrain['Child']=np.nan


# In[ ]:


#right now we have two categories of sex, either male or female but in original titanic movie, child had a big impact.
#so we will create another class child in sex who had age less than 12 years
for i in range(0,891):
    if(titanicTrain.iloc[i,3]<=12):
        titanicTrain['Child']='child'
    else:
        titanicTrain['Child']='adult'


# In[ ]:


#convert Sex and Child into dummies
titanicTrain.Child=pd.get_dummies(titanicTrain.Child)
titanicTrain.Sex=pd.get_dummies(titanicTrain.Sex)


# In[ ]:


titanicTrain.head()


# In[ ]:


#now we can find the family size by adding SibSp and Parch
titanicTrain['FamilySize']=np.nan
for i in range(0,891):
    titanicTrain.iloc[i,7]=titanicTrain.iloc[i,4]+titanicTrain.iloc[i,5]


# In[ ]:


#now delete SibSp and Parch columns
#titanicTrain=titanicTrain.drop('SibSp',1)
#titanicTrain=titanicTrain.drop('Parch',1)


# In[ ]:


#Final EDA
titanicTrain.head()


# In[ ]:


#adding dummy variables for sex
#titanicTrain.Sex=pd.get_dummies(titanicTrain.Sex)


# In[ ]:


#adding libraries for cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


# In[ ]:


#applying train test split to measure the accuracy
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,4,5,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#xtrain.head()


# In[ ]:


#use random forest first for prediction:
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[ ]:


#cross validation scores using all colums got after EDA
#rf_cv_score=cross_val_score(estimator=rf,X=xtrain,y=xtest,cv=5)
#rf_cv_score.mean()


# In[ ]:


#applying train test split to measure the accuracy
#we are removing SibSp column from EDA
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,5,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#cross validation scores after removing sibsp colums got after EDA
#rf_cv_score=cross_val_score(estimator=rf,X=xtrain,y=xtest,cv=5)
#rf_cv_score.mean()


# In[ ]:


#applying train test split to measure the accuracy
#we are removing ParCh column from EDA
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,4,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#cross validation scores After removing Parch colums got after EDA
#rf_cv_score=cross_val_score(estimator=rf,X=xtrain,y=xtest,cv=5)
#rf_cv_score.mean()


# In[ ]:


#applying train test split to measure the accuracy
#we are removing ParCh and SibSp column from EDA
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#cross validation scores after removing parch and sibsp colums got after EDA
#rf_cv_score=cross_val_score(estimator=rf,X=xtrain,y=xtest,cv=5)
#rf_cv_score.mean()


# In[ ]:


#we found if we remove SIBSP colums from the final EDA we got the better accuray as we dont know which columns to remove.
#So the final rf model will be build after removing sibSp column from the final EDA


# In[ ]:


#we need the same EDA for test data as well
#lets check the titanic test data at first
nullRaw=pd.read_csv("../input/test.csv")

nullData=nullRaw[(nullRaw.Age.isnull())]
notNullData=nullRaw[(nullRaw.Age.notnull())]

notNullData=notNullData.drop('PassengerId',1)
notNullData=notNullData.drop('Name',1)
notNullData=notNullData.drop('Ticket',1)
notNullData=notNullData.drop('Cabin',1)

nullData=nullData.drop('PassengerId',1)
nullData=nullData.drop('Name',1)
nullData=nullData.drop('Ticket',1)
nullData=nullData.drop('Cabin',1)
nullData=nullData.drop('Age',1)


# In[ ]:


nullData.head()


# In[ ]:


notNullData.head()


# In[ ]:


#check if there is any null value in age
print (sum(titanicTest.Age.isnull()))


# In[ ]:


#we have one null value in fare and 86 values in Age.
#we will apply mean value for one null fare
#we will use random forest for null age values like train data
#check how many rows are there in test data
titanicTest.Pclass.count()


# In[ ]:


notNullData.head()
sum(notNullData.Fare.isnull())


# In[ ]:


for i in range(0,notNullData.Pclass.count()):
    if(pd.isnull(notNullData.iloc[i,5])==True):
        notNullData.iloc[i,5]=notNullData.Fare.mean()


# In[ ]:


#creating test data where age is null
#nullTest=titanicTest[titanicTest.Age.isnull()]


# In[ ]:


#creating not null test data
#notNullTest=titanicTest[titanicTest.Age.notnull()]


# In[ ]:


#lets check for not null test data
nullData.head()


# In[ ]:


notNullData.head()


# In[ ]:


#converting sex,embarked into dummy variables
nullData.Sex=pd.get_dummies(nullData.Sex)
nullData.Embarked=pd.get_dummies(nullData.Embarked)
notNullData.Sex=pd.get_dummies(notNullData.Sex)
notNullData.Embarked=pd.get_dummies(notNullData.Embarked)


# In[ ]:


#applying random forest for test
testRF=RandomForestRegressor()


# In[ ]:


#fitting data
modelTestRF=testRF.fit(np.array(notNullData.iloc[:,[0,1,3,4,5,6]]),notNullData.iloc[:,2])


# In[ ]:


#assigning the predicted value into a variable
z=modelTestRF.predict(nullData)


# In[ ]:


#check for the values in z
z[1]


# In[ ]:


titanicTest.head()


# In[ ]:


#assigning these falues into titanic test data set
x=0
for i in range(0,418):
    if(pd.isnull(titanicTest.iloc[i,2])==True):
        titanicTest.iloc[i,2]=z[x]
        x=x+1


# In[ ]:


#check for titanic test data
titanicTest.head()


# In[ ]:


#adding a new column
titanicTest['Child']=np.nan
#assigning the values
for i in range(0,418):
    if(titanicTest.iloc[i,2]<=12):
        titanicTest['Child']='child'
    else:
        titanicTest['Child']='adult'


# In[ ]:


#check if the column child is created
titanicTest.head()


# In[ ]:


#creating dummy variable for Sex and child category
titanicTest.Sex=pd.get_dummies(titanicTest.Sex)
titanicTest.Child=pd.get_dummies(titanicTest.Child)


# In[ ]:


#creating family size for test data
titanicTest['FamilySize']=np.nan
for i in range(0,418):
    titanicTest.iloc[i,6]=titanicTest.iloc[i,3]+titanicTest.iloc[i,4]


# In[ ]:


titanicTest.head()


# In[ ]:


#applying train test split to measure the accuracy
#we are removing SibSp column from EDA
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,5,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#titanicTest.head()


# In[ ]:


titanicTrain.head()


# In[ ]:


Counter(titanicTrain.Survived)


# In[ ]:


#we found that this is an imbalanced dataset, so we are balancing it by undersampling
titanicTrain1=titanicTrain[titanicTrain.Survived==1]


# In[ ]:


#fetching those rows having target class as zero
titanicTrain2=titanicTrain[titanicTrain.Survived==0]


# In[ ]:


#taking a sample of 342
titanicTrain0=titanicTrain2.sample(n=342)


# In[ ]:


#joining two different datasets
titanicTrain=titanicTrain1.append(titanicTrain2)


# In[ ]:


titanicTrain=titanicTrain.sample(frac=1).reset_index(drop=True)


# In[ ]:


#creating test set and train set for cross validation
xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,4,6]],titanicTrain['Survived'],test_size=0.25,random_state=123)


# In[ ]:


xtrain.head()


# In[ ]:


#now we are going to apply random forest but with some parameter tuning let's check for the accuracy 
#cross validation scores using all colums got after EDA
m=1
for i in range (0,10):
        rf=RandomForestClassifier(max_depth=m)
        rf_cv_score=cross_val_score(estimator=rf,X=xtrain,y=xtest,cv=5)
        print (rf_cv_score.mean())
        m=m+1


# In[ ]:


#now we are going to apply random forest but with some parameter tuning let's check for the accuracy 
#cross validation scores using all colums got after EDA
m=65
for i in range (0,20):
        rf=RandomForestClassifier(max_depth=4,n_estimators=m)
        rf_cv_score=cross_val_score(estimator=rf,X=xtrain,y=xtest,cv=5)
        print (rf_cv_score.mean())
        m=m+1


# In[ ]:


#so we found the parameters
#let's predict for the test data
rfFinalModel=RandomForestClassifier(max_depth=6,n_estimators=74)


# In[ ]:


#lets check the data before fitting
titanicTrain.head()


# In[ ]:


xtrain.head()


# In[ ]:


#we found parch and family are not required for final model, we are not putting it
rfFinalModel.fit(titanicTrain.iloc[:,[1,2,3,4,6]],titanicTrain['Survived'])


# In[ ]:


ytrain.head()


# In[ ]:


#x=sclf.predict(ytrain)


# In[ ]:


from sklearn import metrics


# In[ ]:


#metrics.accuracy_score(x,ytest)


# In[ ]:


ytrain.head()


# In[ ]:


#model got build, now check for test data before predicting
titanicTest.head()


# In[ ]:


#Lets predict and store it
rfModelOutput=rfFinalModel.predict(titanicTest.iloc[:,[0,1,2,3,5]])


# In[ ]:


Counter(rfModelOutput)
#titanicTest.Age.count()


# In[ ]:


#now we will use SVM for another model we will do ensambling later.
from sklearn import svm


# In[ ]:


modelSvm=svm.SVC()


# In[ ]:


#applying train test split to measure the accuracy
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,4,5,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#cross validation scores using all colums got after EDA
#rf_cv_score=cross_val_score(estimator=modelSvm,X=xtrain,y=xtest,cv=5)
#rf_cv_score.mean()


# In[ ]:


#applying train test split to measure the accuracy
#we are removing SibSp column from EDA
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,5,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#cross validation scores after removing sibsp colums got after EDA
#rf_cv_score=cross_val_score(estimator=modelSvm,X=xtrain,y=xtest,cv=5)
#rf_cv_score.mean()


# In[ ]:


#applying train test split to measure the accuracy
#we are removing ParCh column from EDA
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,4,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#cross validation scores After removing Parch colums got after EDA
#rf_cv_score=cross_val_score(estimator=modelSvm,X=xtrain,y=xtest,cv=5)
#rf_cv_score.mean()


# In[ ]:


#applying train test split to measure the accuracy
#we are removing ParCh and SibSp column from EDA
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#cross validation scores after removing parch and sibsp colums got after EDA
#rf_cv_score=cross_val_score(estimator=modelSvm,X=xtrain,y=xtest,cv=5)
#rf_cv_score.mean()


# In[ ]:


#for final one
xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,4,6]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#here the the train set with all the variables are giving better results
#lets tune the parameters befor creating final model 8202
m=4.5
for i in range(0,10):
    modelSvm=svm.SVC(kernel='rbf',C=m)
    svm_cv_score=cross_val_score(estimator=modelSvm,X=xtrain,y=xtest,cv=5)
    print (svm_cv_score.mean())
    m=m+0.1


# In[ ]:


#so we found if cost is 0.04 the accuracy is maximum, lets find the gamma function lets predict the output
#but lets check the train and test data first
titanicTrain.head()


# In[ ]:


titanicTest.head()


# In[ ]:


#fit the model 0.4
modelSvm=svm.SVC(kernel='rbf',C=5)
modelSvm.fit(titanicTrain.iloc[:,[1,2,3,4,6]],titanicTrain['Survived'])


# In[ ]:


#lets predict the output
svmModelOutput=modelSvm.predict(titanicTest.iloc[:,[0,1,2,3,5]])


# In[ ]:


Counter(svmModelOutput)


# In[ ]:


#gradiant boosting
from sklearn.ensemble import GradientBoostingClassifier
modelGradBoost=GradientBoostingClassifier(max_depth=4,min_samples_leaf=4)
modelGradBoost.fit(titanicTrain.iloc[:,[1,2,3,4,6]],titanicTrain['Survived'])


# In[ ]:


m=2
for i in range(0,10):
    modelGradBoost=GradientBoostingClassifier(max_depth=4,min_samples_split=m)
    svm_cv_score=cross_val_score(estimator=modelGradBoost,X=xtrain,y=xtest,cv=5)
    print (svm_cv_score.mean())
    m=m+1


# In[ ]:


modelGradBoost=GradientBoostingClassifier(max_depth=4,min_samples_split=5)


# In[ ]:


#cross validation scores using all colums got after EDA
rf_cv_score=cross_val_score(estimator=modelGradBoost,X=xtrain,y=xtest,cv=5)
rf_cv_score.mean()


# In[ ]:


#fitting the data
modelGradBoost.fit(titanicTrain.iloc[:,[1,2,3,4,6]],titanicTrain['Survived'])


# In[ ]:


#storing the output of gradient boosting
gradBoostOutput=modelGradBoost.predict(titanicTest.iloc[:,[0,1,2,3,5]])


# In[ ]:


#lets try logistic regression model
from sklearn import linear_model


# In[ ]:


modelLogit=linear_model.LogisticRegression(penalty='l1')


# In[ ]:


titanicTrain.head()


# In[ ]:


#applying train test split to measure the accuracy
xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,4,6]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#cross validation scores using all colums got after EDA
rf_cv_score=cross_val_score(estimator=modelLogit,X=xtrain,y=xtest,cv=5)
rf_cv_score.mean()


# In[ ]:


#applying train test split to measure the accuracy
#we are removing SibSp column from EDA
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,5,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#m=1
#for i in range(0,10):
    #modelLogit=linear_model.LogisticRegression(penalty='l1')
    #svm_cv_score=cross_val_score(estimator=modelSvm,X=xtrain,y=xtest,cv=5)
    #print svm_cv_score.mean()
    #m=m+1


# In[ ]:


#cross validation scores after removing sibsp colums got after EDA
#rf_cv_score=cross_val_score(estimator=modelLogit,X=xtrain,y=xtest,cv=5)
#rf_cv_score.mean()


# In[ ]:


#applying train test split to measure the accuracy
#we are removing ParCh column from EDA
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,4,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#cross validation scores After removing Parch colums got after EDA
#rf_cv_score=cross_val_score(estimator=modelLogit,X=xtrain,y=xtest,cv=5)
#rf_cv_score.mean()


# In[ ]:


#applying train test split to measure the accuracy
#we are removing ParCh and SibSp column from EDA
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#cross validation scores after removing parch and sibsp colums got after EDA
#rf_cv_score=cross_val_score(estimator=modelLogit,X=xtrain,y=xtest,cv=5)
#rf_cv_score.mean()


# In[ ]:


titanicTrain.head()


# In[ ]:


#we need to include all columns to create logistic regression model
#fit the model
modelLogit=linear_model.LogisticRegression(penalty='l1')
modelLogit.fit(titanicTrain.iloc[:,[1,2,3,4,6]],titanicTrain['Survived'])


# In[ ]:


titanicTest.head()


# In[ ]:


#predict the output
logitFinalOutput=modelLogit.predict(titanicTest.iloc[:,[0,1,2,3,5]])


# In[ ]:


Counter(logitFinalOutput)


# In[ ]:


#lets try knn method
from sklearn import neighbors


# In[ ]:


modelKnn=neighbors.KNeighborsClassifier()


# In[ ]:


titanicTrain.head()


# In[ ]:


#applying train test split to measure the accuracy
xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,4,6]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#cross validation scores using all colums got after EDA
rf_cv_score=cross_val_score(estimator=modelKnn,X=xtrain,y=xtest,cv=5)
rf_cv_score.mean()


# In[ ]:


#applying train test split to measure the accuracy
#we are removing SibSp column from EDA
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,5,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#cross validation scores after removing sibsp colums got after EDA
#rf_cv_score=cross_val_score(estimator=modelKnn,X=xtrain,y=xtest,cv=5)
#rf_cv_score.mean()


# In[ ]:


#applying train test split to measure the accuracy
#we are removing ParCh column from EDA
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,4,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#cross validation scores After removing Parch colums got after EDA
#rf_cv_score=cross_val_score(estimator=modelKnn,X=xtrain,y=xtest,cv=5)
#rf_cv_score.mean()


# In[ ]:


#applying train test split to measure the accuracy
#we are removing ParCh and SibSp column from EDA
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#cross validation scores after removing parch and sibsp colums got after EDA
#rf_cv_score=cross_val_score(estimator=modelKnn,X=xtrain,y=xtest,cv=5)
#rf_cv_score.mean()


# In[ ]:


#the final one
#applying train test split to measure the accuracy
#xtrain,ytrain,xtest,ytest=train_test_split(titanicTrain.iloc[:,[1,2,3,4,5,6,7]],titanicTrain['Survived'],test_size=0.3,random_state=123)


# In[ ]:


#first one is giving better accuracy hence, we will be going for this one but need to tune parameters
m=1
for i in range(0,10):
    modelKnn=neighbors.KNeighborsClassifier(n_neighbors=m)
    knn_cv_score=cross_val_score(estimator=modelKnn,X=xtrain,y=xtest,cv=5)
    print (knn_cv_score.mean())
    m=m+1


# In[ ]:


#when n=16 it gives best results
#lets create the model with n=3
#fit the model
modelKnn=neighbors.KNeighborsClassifier(n_neighbors=3)
modelKnn.fit(titanicTrain.iloc[:,[1,2,3,6]],titanicTrain['Survived'])


# In[ ]:


#lets predict
knnOutput=modelKnn.predict(titanicTest.iloc[:,[0,1,2,5]])


# In[ ]:


#lets try with nural net
from sklearn.neural_network import MLPClassifier


# In[ ]:


modelAnn = MLPClassifier()


# In[ ]:


#fitdata into model
modelAnn.fit(titanicTrain.iloc[:,[1,2,3,4,6]],titanicTrain['Survived'])


# In[ ]:


annOutput=modelAnn.predict(titanicTest.iloc[:,[0,1,2,3,5]])


# In[ ]:


#lets do voting ensemble
z=[]
for i in range(0,418):
    p=(rfModelOutput[i]+svmModelOutput[i]+gradBoostOutput[i])
    #p=(rfModelOutput[i]+svmModelOutput[i]+logitFinalOutput[i])
    if(p>2):
        z.append(1)
    elif(p<2):
        z.append(0)
    elif(p==2):
        z.append(rfModelOutput[i])
    #if(rfModelOutput[i]==0 and svmModelOutput[i]==0 and logitFinalOutput[i]==0 and knnOutput[i]==0  and annOutput[i]==1):
        #z.append(1)
    #else:
        #z.append(rfModelOutput[i])


# In[ ]:


Counter(z)


# In[ ]:


#data preperation according to the format
Test=pd.read_csv("../input/test.csv")


# In[ ]:


Test=Test.drop('Pclass',axis=1)
Test=Test.drop('Name',axis=1)
Test=Test.drop('Sex',axis=1)
Test=Test.drop('Age',axis=1)
Test=Test.drop('SibSp',axis=1)
Test=Test.drop('Parch',axis=1)
Test=Test.drop('Ticket',axis=1)
Test=Test.drop('Fare',axis=1)
Test=Test.drop('Cabin',axis=1)
Test=Test.drop('Embarked',axis=1)


# In[ ]:


Test['Survived']=z


# In[ ]:


#check for the submission file
Test.head()


# In[ ]:


#Lets convert this to excel file
Test.to_csv('titanicOutput.csv',index=False)

