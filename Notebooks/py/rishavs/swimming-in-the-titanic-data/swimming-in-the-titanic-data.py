#!/usr/bin/env python
# coding: utf-8

# ----------
# 
# 
# #Please note that this isn't a solution. 
# The following notebook might be helpful to **beginners**, such as myself.
# It provides a **general work flow** you might follow to solve Machine learning problems such as the  **Titanic challenge** and many more.
# 

# ----------
# 
# 
# 

# **General work flow:**
# 
#  - Get the csv data
#  - Structure the data such that it could be passed to the Machine learning model
#  - Do machine learning
#  - Validate your model
#  - Go have a pizza
# 

# ----------
# 
# 
# ----------
# 
# 
# #First things first: Import

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv("../input/train.csv",index_col = 'PassengerId') #I'll only use Training data for demonstration
train.describe()


# ----------
# 
# 
# ----------
# 
# 
# #Structuring the Data first

# In[ ]:


train['Age'].fillna(train['Age'].median(), inplace=True)


# Above, i've filled the missing values in **Age** column with the median age.

# **What is in the name?** Nothing and everything.
# 

# In[ ]:


train['Mr'] = 0
train['Mrs'] = 0
train['Miss'] = 0
train['royalty'] = 0
train['officer'] = 0


# In[ ]:


#depending on the name, categorizing individuals
for index,row in train.iterrows():
    name = row['Name']
    if 'Mr.' in name:
        train.set_value(index,'Mr',1)
    elif 'Mrs.' in name:
        train.set_value(index,'Mrs',1)
    elif 'Miss.' in name:
        train.set_value(index,'Miss',1)
    elif 'Lady' or 'Don' or 'Dona' or 'sir' or 'master' in name:
        train.set_value(index,'royalty',1)
    elif 'rev' in name:
        train.set_value(index,'officer',1)
        
train.head()


# In[ ]:


train.drop('Name',inplace=True, axis=1)
train.head() #Dropped the names column


# Port of Embarkation:
#                    (**C** = Cherbourg; **Q** = Queenstown; **S** = Southampton)
# 

# In[ ]:



train['Embarked_S'] = 0
train['Embarked_C'] = 0
train['Embarked_Q'] = 0
train['Embarked_unknown'] = 0

for index,row in train.iterrows():
    embarkment = row['Embarked']
    if embarkment == 'S':
        train.set_value(index,'Embarked_S',1)
    elif embarkment == 'C':
        train.set_value(index,'Embarked_C',1)
    elif embarkment == 'Q':
        train.set_value(index,'Embarked_Q',1)
    else:
        train.set_value(index,'Embarked_unknown',1)
   

train.head()


# In[ ]:


train.drop('Embarked', inplace = True, axis = 1) #Dropped column 'Embarked'


# Sex:
# 
#  - 1 = Male
#  - 0 = Female

# In[ ]:



for index,row in train.iterrows():
    if row['Sex'] == 'male':
        train.set_value(index, 'Sex', 1)
    else:
        train.set_value(index,'Sex',0)
train.head()


# In[ ]:


#wont be using the feature "Ticket", so drop it

train.drop('Ticket', inplace= True, axis = 1)
train.head()


# In[ ]:


#lets categorize the fares as: cheap, average, and costly

train['Fare_cheap']=0
train['Fare_average']=0
train['Fare_costly']=0

for index,row in train.iterrows():
    if row['Fare'] <= 30.0 :
        train.set_value(index, 'Fare_cheap', 1)
    elif row['Fare'] >30 and  row['Fare'] <= 70.0:
        train.set_value(index,'Fare_average',1)
    else:
        train.set_value(index, 'Fare_costly',1)
        
train.head()


# In[ ]:


train.drop('Fare',inplace = True, axis =1) #now we don't need the fare column
train.head()


# In[ ]:


#we wont be considering the feature 'Cabin' 
#So,dropping that column as well
train.drop('Cabin',inplace = True, axis = 1)
train.head()


# In[ ]:


train.describe() #Checking for any missing values due to manipulation


# In[ ]:


X = train[['Pclass','Sex','Age','SibSp','Parch','Mr','Mrs','Miss','royalty','officer','Embarked_S','Embarked_C','Embarked_Q','Embarked_unknown','Fare_cheap','Fare_average','Fare_costly']]
y = train.Survived #Works if there aren't any spaces in the column name

#17 features
X.shape


# In[ ]:


y.shape


# **Till now, we were just trying to make the dataset *suitable so that we can pass it to a machine learning model*** 
# 
# **So finally, we have our feature matrix and a response matrix, X,y (which we will use for the machine learning models)**
# 
# 
# ----------
# 
# 
# ----------
# 
# 
# #Machine Learning

# **Models we will use : Support Vector Machine & Knn**
# 
# 

# **SVM :**

# In[ ]:


from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score #k fold cross validation

svm_model = SVC() 
svm_model.kernel= 'linear'
score_svm = cross_val_score(svm_model,X,y,cv=10, scoring= 'accuracy')
print(score_svm.mean())


# Thus, our (linear) SVM model produces an average accuracy of about **81.25%**
# #Trying this on the second model i.e, Knn
# 
# **KNN :**

# In[ ]:


get_ipython().magic(u'matplotlib inline')
from sklearn.neighbors import KNeighborsClassifier
k_range= range(1,31)
score_knn_list=[]
#how many neighbours should we consider?
for n in k_range:
    knn_model = KNeighborsClassifier(n_neighbors = n)
    score_knn = cross_val_score(knn_model,X,y,cv=10,scoring ='accuracy')
    score_knn_list.append(score_knn.mean())

plt.plot(k_range,score_knn_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross Validated Accuracy')


# As we can see, for (k_neighbors) **k= 5 - 10**, we will get better accuracy( So we choose k=5)
# 

# In[ ]:


knn_model_2 = KNeighborsClassifier(n_neighbors = 5)
score_knn = cross_val_score(knn_model_2,X,y,cv=10,scoring ='accuracy')
print(score_knn.mean())


# Thus, our KNN model produces an average accuracy of about **79.8%**
# 
# 
# #Conclusion: 
# #SVM model gave us a better accuracy than the KNN model, which is about **81.25%** !
# :)
# 
# 
# (**Please note that this isn't a solution and is only aimed at providing some help to the beginners**)
# 
# **Let me know if it helped!**
# #Thank You.
