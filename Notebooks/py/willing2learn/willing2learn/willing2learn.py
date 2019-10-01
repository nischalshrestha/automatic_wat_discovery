#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
#path="E:\surya\work\kaggle titanic dataset"
#os.chdir(path)
# Any results you write to the current directory are saved as output.


# In[ ]:


print (os.listdir(os.getcwd()))


# In[ ]:


#Step 1: Dataset Exploration


# In[ ]:


print ("\ntotal number of datapoints : 891")
print ("\nnumber of useful features available : 9")
print ("\nname of the passenger is not used as a feature.")
print ("\ncabin number has many missing values")


# In[ ]:


#reading training dataset
feature_list=['PassengerId','Pclass','Name','Sex', 'Age','SibSp','Parch',
                                                   'Ticket','Fare','Cabin','Embarked']
df_train_features=pd.read_csv("../input/train.csv",usecols=feature_list)
df_train_labels=pd.read_csv("../input/train.csv",usecols=['Survived'])


# In[ ]:


df_train_features.head()


# In[ ]:


df_train_labels.head()


# In[ ]:


#DATA PRE-PROCESSING

#replacing 'male' with 1 and 'female' with 0 in the 'sex' column
df_train_features=df_train_features.replace('male',1)
df_train_features=df_train_features.replace('female',0)

#extracting the numerical part of the ticket number
c=5
for s in df_train_features.iloc[:,7]:
    if isinstance(s,str):
        value=[int(s) for s in s.split(' ') if s.isdigit()]
        if (len(value)!=0):
            tktnum=value[0]
        else:
            tktnum=-1
        if (c>0):
            c-=1
        df_train_features=df_train_features.replace(s,tktnum)

#In 'embarked' column, replacing 'S' by 1,'C' by 2 and 'Q' by 3
df_train_features=df_train_features.replace({"S":1,"C":2,"Q":3})

#Extracting only the surnames
for s in df_train_features.iloc[:,2]:
    if (len(s)!=0):
        value=[s for s in s.split(',')]
        surname=value[0]
    df_train_features=df_train_features.replace(s,surname)

#finding the list of unique surnames present and assigning them a numerical value
ls=df_train_features.Name.unique()
df_train_features=df_train_features.replace(ls,range(len(ls)))

#For cases where a passenger has more than one cabin number, extra features will be added. 
#If a person has two cabins, then 4 features will be added. 2 for alpha. part and 2 for numerical part.    
#splitting cabin number in two parts: cabin1 : contains the alphabetical part and cabin2 : contains the numerical part

#first let us find the maximum number of cabins a passenger has.
Max=0
for s in df_train_features.iloc[:,9]:
    if isinstance(s,str):
        value=[s for s in s.split(' ')]
        if (Max<len(value)):
            Max=len(value)
print ('maximum number of cabins a passenger has : ',Max)

#now let us add the required number of features with default values for each row. Later on the value of a row will be changed as 
#'needed'
x=range(Max)
for i in x:
    df_train_features.loc[:,'ap'+str(i)]=-1
    df_train_features.loc[:,'np'+str(i)]=-1
    feature_list.append('ap'+str(i))
    feature_list.append('np'+str(i))
#now let us fill in the apprpriate values in these new columns
ap=11
np=12
rowin=0

for s in df_train_features.iloc[:,9]:
    if isinstance(s,str):
        #print (s)
        #print (type(s))
        value=[s for s in s.split(' ')]
        for cn in value:
            #print (cn[0])
            #print (cn[1:])
            #print (ap)
            df_train_features.iloc[rowin,ap]=ord(cn[0])
            if (cn[1:]!=''):
                df_train_features.iloc[rowin,np]=int(cn[1:])
            else:
                df_train_features.iloc[rowin,np]=-1
            ap+=2
            np+=2
    ap=11
    np=12
    rowin+=1
    
            
#finally removing the original 'cabin' column
df_train_features=df_train_features.drop(columns=['Cabin'])
#removing from features list as well
del feature_list[feature_list.index('Cabin')]

#replacing all the missing values in age column by mean age
mean_age=df_train_features['Age'].mean()
df_train_features['Age']=df_train_features['Age'].fillna(mean_age)

#there are two nan values present in 'Embarked' column. we are replacing it with median value
median=df_train_features['Embarked'].median()
df_train_features['Embarked']=df_train_features['Embarked'].fillna(median)





            
            
            



# In[ ]:


df_train_features


# In[ ]:


#Converting dataframe to numpy arrays for further use
X=df_train_features.values
y=df_train_labels.values
print (X.shape)
print (y.shape)


# In[ ]:


#Step 2: OPTIMIZE FEATURE SELECTION/ENGINEERING


# In[ ]:


#First, let us do feature scalling so that no feature gets more importance simply based on it's numerical value
#feature scalling
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X=scaler.fit_transform(X)


# In[ ]:


X[0:5]


# In[ ]:


print (y[:5])


# In[ ]:


len(y)


# In[ ]:


new=[]
for i in y:
    for j in i:
        new.append(j)
print (new[:5])
y=new


# In[ ]:


#now let us find the importance of all features using selectkpercentile
from sklearn.feature_selection import SelectPercentile, f_classif
selector = SelectPercentile(f_classif, percentile=40)#highest accuracy .80 (approx.) from decision tree classifier
#                                                                                                   at this percentile
selector.fit(X,y)
X_new=selector.transform(X)
print ('shape of X_new ',X_new.shape)
try:
    X_points = range(X.shape[1])
except IndexError:
    X_points = 1


# In[ ]:


#checking out the scores of the features
score=selector.scores_.tolist()
names=list(df_train_features)
new=zip(names,score)
for i in new:
    print (i[0]," score = {:8.2f}".format(i[1]))


# In[ ]:


plt.bar(X_points , selector.scores_, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
        edgecolor='black')


# In[ ]:


#STEP 3:Trying out a variety of classifiers and tuning them as well 


# In[ ]:


#Splitting data into training and testing set
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(X_new, y, test_size=0.30, random_state=42)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[ ]:


#Trial 1: Decision Tree Classifier 

from sklearn.tree import DecisionTreeClassifier

parameters={'criterion':('gini','entropy'),'splitter':('best','random')}
dtc = DecisionTreeClassifier()
clf=GridSearchCV(dtc,parameters)

clf.fit(features_train, labels_train)

print("Best estimator found by grid search:")
print(clf.best_estimator_)
pred=clf.predict(features_test)
print ("\naccuracy_score : ",accuracy_score(labels_test,pred))
print ('\nprecision : \n',precision_score(labels_test,pred))
print ('\nrecall : \n',recall_score(labels_test,pred))


# In[ ]:


#Trial 2: SVC
from sklearn.svm import SVC

parameters={'C':(0.1,1,10,100),'kernel':('linear','rbf','poly')}
svc=SVC()
clf=GridSearchCV(svc,parameters)

clf.fit(features_train, labels_train)

print("Best estimator found by grid search:")
print(clf.best_estimator_)
pred=clf.predict(features_test)
print ("\naccuracy_score : ",accuracy_score(labels_test,pred))
print ('\nprecision : \n',precision_score(labels_test,pred))
print ('\nrecall : \n',recall_score(labels_test,pred))


# In[ ]:


#Trial 3: Naive Bayes
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(features_train, labels_train)

pred=clf.predict(features_test)
print ("\naccuracy_score : ",accuracy_score(labels_test,pred))
print ('\nprecision : \n',precision_score(labels_test,pred))
print ('\nrecall : \n',recall_score(labels_test,pred))


# In[ ]:


#Now that we've got our final trained algorithm DTC, we we'll retrain it again.
#This time with the entire training dataset.
svc = SVC(C=100,kernel='linear')
svc.fit(X_new, y)


# In[ ]:


#now let us import data from test file and predict the survival
feature_list=['PassengerId','Pclass','Name','Sex', 'Age','SibSp','Parch',
                                                   'Ticket','Fare','Cabin','Embarked']
df_test_features=pd.read_csv("../input/test.csv",usecols=feature_list)

#data-preprocessing
#DATA PRE-PROCESSING

#replacing 'male' with 1 and 'female' with 0 in the 'sex' column
df_test_features=df_test_features.replace('male',1)
df_test_features=df_test_features.replace('female',0)

#extracting the numerical part of the ticket number
c=5
for s in df_test_features.iloc[:,7]:
    if isinstance(s,str):
        value=[int(s) for s in s.split(' ') if s.isdigit()]
        if (len(value)!=0):
            tktnum=value[0]
        else:
            tktnum=-1
        if (c>0):
            c-=1
        df_test_features=df_test_features.replace(s,tktnum)

#In 'embarked' column, replacing 'S' by 1,'C' by 2 and 'Q' by 3
df_test_features=df_test_features.replace({"S":1,"C":2,"Q":3})

#Extracting only the surnames
for s in df_test_features.iloc[:,2]:
    if (len(s)!=0):
        value=[s for s in s.split(',')]
        surname=value[0]
    df_test_features=df_test_features.replace(s,surname)

#finding the list of unique surnames present and assigning them a numerical value
ls=df_test_features.Name.unique()
df_test_features=df_test_features.replace(ls,range(len(ls)))

#For cases where a passenger has more than one cabin number, extra features will be added. 
#If a person has two cabins, then 4 features will be added. 2 for alpha. part and 2 for numerical part.    
#splitting cabin number in two parts: cabin1 : contains the alphabetical part and cabin2 : contains the numerical part

#first let us find the maximum number of cabins a passenger has.
Max=0
for s in df_test_features.iloc[:,9]:
    if isinstance(s,str):
        value=[s for s in s.split(' ')]
        if (Max<len(value)):
            Max=len(value)
print ('maximum number of cabins a passenger has : ',Max)

#now let us add the required number of features with default values for each row. Later on the value of a row will be changed as 
#'needed'
x=range(Max)
for i in x:
    df_test_features.loc[:,'ap'+str(i)]=-1
    df_test_features.loc[:,'np'+str(i)]=-1
    feature_list.append('ap'+str(i))
    feature_list.append('np'+str(i))
#now let us fill in the apprpriate values in these new columns
ap=11
np=12
rowin=0

for s in df_test_features.iloc[:,9]:
    if isinstance(s,str):
        #print (s)
        #print (type(s))
        value=[s for s in s.split(' ')]
        for cn in value:
            #print (cn[0])
            #print (cn[1:])
            #print (ap)
            df_test_features.iloc[rowin,ap]=ord(cn[0])
            if (cn[1:]!=''):
                df_test_features.iloc[rowin,np]=int(cn[1:])
            else:
                df_test_features.iloc[rowin,np]=-1
            ap+=2
            np+=2
    ap=11
    np=12
    rowin+=1
    
            
#finally removing the original 'cabin' column
df_test_features=df_test_features.drop(columns=['Cabin'])
#removing from features list as well
del feature_list[feature_list.index('Cabin')]

#replacing all the missing values in age column by mean age
mean_age=df_test_features['Age'].mean()
df_test_features['Age']=df_test_features['Age'].fillna(mean_age)

#there are two nan values present in 'Embarked' column. we are replacing it with median value
median=df_test_features['Embarked'].median()
df_test_features['Embarked']=df_test_features['Embarked'].fillna(median)


# In[ ]:


#checking for any NAN values left
l=[]
for i in feature_list:
    x=df_test_features[i].isnull().sum().sum()
    if x>0:
        print (x)
        l.append(i)
for i in l:
    print (i)


# In[ ]:


avg_fare=df_test_features['Fare'].mean()
df_test_features['Fare']=df_test_features['Fare'].fillna(avg_fare)


# In[ ]:


X=df_test_features.values
print (X.shape)


# In[ ]:


#Converting dataframe to numpy arrays for further use


#First, let us do feature scalling so that no feature gets more importance simply based on it's numerical value
#feature scalling
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X=scaler.fit_transform(X)

#using previously selected features
X_new=selector.transform(X)
print ('shape of X_new ',X_new.shape)
try:
    X_points = range(X.shape[1])
except IndexError:
    X_points = 1




# In[ ]:


# Decision Tree Classifier 

pred=svc.predict(X_new)


# In[ ]:


print (pred.shape)


# In[ ]:


x =range(892,1310)
#creating the submission file
submission=pd.DataFrame({'PassengerId':x,'Survived':pred})
submission.to_csv(path_or_buf='submission.csv',index=False)


# In[ ]:




