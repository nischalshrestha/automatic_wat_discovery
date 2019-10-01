#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
sns.set()

#Sklearn OneHot Encoder to Encode categorical integer features
from sklearn.preprocessing import OneHotEncoder
#Sklearn train_test_split to split a set on train and test 
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split      # for old sklearn version use this to split a dataset 
# SVM Classifier from sklearn
from sklearn import svm


# In[4]:


#Import the training data set
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data.head()


# In[3]:


data.isnull().sum()


# In[4]:


#Construct an X matrix
x_train = data[['Pclass','Sex','Age','Parch','SibSp','Embarked']].copy()
x_test = test[['Pclass','Sex','Age','Parch','SibSp','Embarked']].copy()
x_train.shape, x_test.shape


# In[5]:


PassengerID = np.array(test['PassengerId'])


# In[6]:


#Create Y array
y = np.array(data[['Survived']])
print(y.shape)


# In[7]:


data[data.Survived==1].SibSp.plot.hist(alpha=0.5,color='blue',bins=50)
data[data.Survived==0].SibSp.plot.hist(alpha=0.5,color='red',bins=50)
plt.legend(['Survived','Died'])
plt.show()


# In[8]:


data[data.Survived==1].Parch.plot.hist(alpha=0.5,color='blue',bins=50)
data[data.Survived==0].Parch.plot.hist(alpha=0.5,color='red',bins=50)
plt.legend(['Survived','Died'])
plt.show()


# In[9]:


data[data.Survived==1].Age.plot.hist(alpha=0.5,color='blue',bins=50)
data[data.Survived==0].Age.plot.hist(alpha=0.5,color='red',bins=50)
plt.legend(['Survived','Died'])
plt.show()


# In[10]:


ax1 = sns.countplot(x="Pclass", data=data[data['Age'].isnull()], hue='Sex')


# In[11]:


ax2 = sns.countplot(x="SibSp", data=data, hue='Pclass')


# In[12]:


g = sns.factorplot(x="Pclass", hue="Sex", col="SibSp",
                   data=data, kind="count",
                   size=4, aspect=.7);


# In[13]:


sns.pointplot(x="Sex", y="Survived", hue="Pclass", data=data)


# In[14]:


# Work with NaN values
def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5,16,100]        
label_names = ['Missed',"Infant","Adult","Senior"]

x_train = process_age(x_train,cut_points,label_names)
x_test = process_age(x_test,cut_points,label_names)


# In[15]:


# Create a dummies
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

for column in ["Pclass","Sex", 'Embarked', 'Age_categories']:
    x_train = create_dummies(x_train,column)
    x_test = create_dummies(x_test,column)


# In[16]:


# Drop the pre-dummies columns 
x_train=x_train.drop(['Pclass','Sex','Age_categories', 'Age', 'SibSp', 'Parch', 'Embarked'], axis=1)
x_test=x_test.drop(['Pclass','Sex','Age_categories', 'Age', 'SibSp', 'Parch', 'Embarked'],axis=1)
x_train.head()


# In[17]:


#Construct X matrix
x_train = np.array(x_train)
x_test = np.array(x_test)
x_train.shape, x_test.shape


# In[18]:


# The data and task at all has a high value of uncertainty, so lets set test_size=0.5
xn_train, xn_test, yn_train, yn_test = train_test_split(x_train, y, test_size=0.5, random_state=40)
xn_train.shape, xn_test.shape, yn_train.shape, yn_test.shape


# In[19]:


# Support Vector Machine Algorithm will be used
# The best C and gamma parameters determines as follows

C=np.array([0.3,7,10,12,15,20,70,100])
g=np.array([0.4,0.45,0.5,1,3,5,10,30,100])
CC, gg = np.meshgrid(C,g)
scores = np.zeros(CC.shape)
for i in range (len(scores[:,0])):
    for j in range (len(scores[0,:])):
        svc = svm.SVC(C=CC[i,j], gamma=gg[i,j], probability=True, kernel='rbf')  
        svc.fit(xn_train, yn_train) 
        scores[i,j] = svc.score(xn_test,yn_test)


# In[20]:


ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
print('max Score = ',scores[ind],'\noptimal gamma = ',gg[ind],'\noptimal C = ', CC[ind])


# In[21]:


svc = svm.SVC(C=0.3, gamma=0.4, probability=True, kernel='rbf')  
svc.fit(x_train, y) 
prediciton = svc.predict(x_test)


# In[22]:


# Submit the result

submission_df = {"PassengerId": PassengerID,
                 "Survived": prediciton}
submission = pd.DataFrame(submission_df)


# In[23]:


submission.to_csv("submission.csv",index=False)


# In[ ]:




