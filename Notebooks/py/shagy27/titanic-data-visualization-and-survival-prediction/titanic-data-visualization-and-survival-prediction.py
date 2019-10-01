#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization, graph plotting
from sklearn import linear_model,preprocessing,tree,model_selection # for prediction models

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# First step is to load the training data from train.csv file so that we can do data analysis and visualize the training data with the help of graphs in order to find the features for  our prediction model.'Features'  mean the fields of our training data using which we will predict the target value, i.e whether a passenger on titanic will survive or not .
# 
# To load data from file we use read_csv() method of pandas package.
# 

# In[ ]:


train = pd.read_csv("../input/train.csv")
train.head() #print first 5 rows of train dataframe from top
train.tail() #print last 5 rows of train dataframe from top
train


# To find the correlation between features in a  dataset use corr() method of pandas Dataframe
# negative value means inverse realation(target value will decrease if feature value increases) and positive means direct      relation(target value increases with increase in feature value) 

# In[ ]:


# to find the correlation between features of the dataset
# negative value means inverse realation(target value will decrease if feature value 
# increases) and positive means direct relation(target value increases with increase in 
# feature value) 
train.corr()


# In order to find whether all the columns in the dataset have equal no. of values or any values are missing, use count() method of 
# panda.DataFrame
# Columns 'Embarked' and 'Sex' are missing from above correlation table because they are non-numerical features.
# We will convert them into numerical values later using the user defined function clean_data()

# In[ ]:


train.count() 


# Because missing values are not good for the analysis and target prediction and they would not be of any help so we need to drop column 'Age' and 'Cabin' but since 'Age' is a numerical value and majority of data is available for it so we can fill missing values with the mean age of all passengers but we can't do the same for 'Cabin' since majority values are missing.
# 
# there are are lot of values as NaN which means not applicable i.e data are not avalaible for that column entry, so we need to replace those missing values with the mean/median value if the majority of values are available for that column.

# In[ ]:


def clean_data(data):
    data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median()).astype(int)
    data['Age'] = data['Age'].fillna(data['Age'].dropna().median()).astype(int)
    
    data['Embarked'] = data['Embarked'].fillna('S')
    data.loc[data['Embarked'] == 'S','Embarked'] = 0
    data.loc[data['Embarked'] == 'C','Embarked'] = 1
    data.loc[data['Embarked'] == 'Q','Embarked'] = 2
    
    data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
    
    
    return(data)


# In[ ]:


train_df = clean_data(train)


# In[ ]:


train_df
train_df.info()


# In[ ]:


train_df.corr()


# In[ ]:


fig = plt.figure(figsize = (30,20))
alpha = alphascatterplot = 0.2
alpha_bar = 0.5

ax1 =plt.subplot2grid((4,4),(0,0))
train_df.Survived.value_counts().sort_index().plot(kind = "bar",alpha = alpha_bar)

ax1.set_xlim(-1,2)

plt.title("Distirbution of passsengers survived")
plt.grid(b = True , which = "major")

ax2 =plt.subplot2grid((4,4),(0,1))
plt.scatter(train_df.Survived, train_df.Age, alpha = alphascatterplot)
plt.xlabel("Survived")
plt.ylabel("Age")
plt.title("Survival by Age, Surivived = 1")
plt.grid(b = True , which = "major")

ax3 =plt.subplot2grid((4,4),(0,2))
train_df.Pclass.value_counts().sort_index().plot(kind = "bar",alpha = alpha_bar) 
plt.xlabel("Passenger Class")
ax3.set_xlim(-1,len(train.Pclass.value_counts()))
plt.title("Distirbution of passsengers class")
plt.grid(b = True , which = "major")

ax4 =plt.subplot2grid((4,4),(1,0))
train_df.Sex.value_counts().sort_index().plot(kind = "bar",alpha = alpha_bar) 
plt.xlabel("Sex(0-male 1-female)")
ax4.set_xlim(-1,len(train_df.Sex.value_counts()))
plt.grid(b = True , which = "major")
plt.title("Sex Distribution")

ax5 =plt.subplot2grid((4,4),(1,1),colspan = 2)
train_df.Embarked.value_counts().sort_index().plot(kind = "bar",alpha = alpha_bar) 
plt.xlabel("Boarding Station S(0),C(1),Q(2)")
plt.ylabel("No. of Passenger")
ax5.set_xlim(-1,len(train_df.Embarked.value_counts()))

plt.grid(b = True , which = "major")
plt.title("Passengers per boarding station")






ax6 =plt.subplot2grid((4,4),(2,0), colspan = 2)
train_df.Age[train_df.Pclass == 1].plot(kind = "kde")
train_df.Age[train_df.Pclass == 2].plot(kind = "kde")
train_df.Age[train_df.Pclass == 3].plot(kind = "kde")
plt.xlabel("Age")
plt.title("Age Density in each Passenger class")
plt.legend(('1st Class','2nd Class','3rd Class'),loc = 'best')
plt.title("Age Density in each Passenger class")
plt.grid(b = True , which = "major")



ax7 =plt.subplot2grid((4,4),(3,0), colspan = 2)
train_df.Age[train_df.Embarked == 0].plot(kind = "kde")
train_df.Age[train_df.Embarked == 1].plot(kind = "kde")
train_df.Age[train_df.Embarked == 2].plot(kind = "kde")
plt.xlabel("Age")
plt.title("Age Density by Boarding station")
plt.legend(('1st Class','2nd Class','3rd Class'),loc = 'best')
plt.grid(b = True , which = "major")


# In[ ]:


train_df.loc[ train_df['Age'] <= 16, 'Age'] = 0
train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 32), 'Age'] = 1
train_df.loc[(train_df['Age'] > 32) & (train_df['Age'] <= 48), 'Age'] = 2
train_df.loc[(train_df['Age'] > 48) & (train_df['Age'] <= 64), 'Age'] = 3
train_df.loc[ train_df['Age'] > 64, 'Age']
train_df


# In[ ]:




feature_names = ['Pclass','Age','Sex','Embarked','Parch','SibSp']
features = train_df[feature_names].values
target = train_df['Survived'].values
classifier = linear_model.LogisticRegression()
c = classifier.fit(features,target)
print (c.score(features,target))


# In[ ]:


ypred = c.predict(features)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(target,ypred))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(features,target)
ypred = knn.predict(features)
print(metrics.accuracy_score(target,ypred))


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(features,target)
ypred = knn.predict(features)
print(metrics.accuracy_score(target,ypred))


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size = 0.4,random_state= 4)


# In[ ]:


logreg = linear_model.LogisticRegression()


# In[ ]:


logreg.fit(X_train,y_train)


# In[ ]:


y_pred = logreg.predict(X_test)


# In[ ]:


print(metrics.accuracy_score(y_test,y_pred))


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 16)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))


# In[ ]:


k_range = range(1,25)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))


# In[ ]:


plt.plot(k_range,scores)


# In[ ]:


test = pd.read_csv('../input/test.csv')
test = clean_data(test)
test.loc[ test['Age'] <= 16, 'Age'] = 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3
test.loc[ test['Age'] > 64, 'Age']
test


# In[ ]:



x_test=test[feature_names].values

knn = KNeighborsClassifier(n_neighbors = 16)
knn.fit(features,target)
y_pred=knn.predict(x_test)
submission =pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })
submission


# In[ ]:


submission.to_csv("../output/submission.csv",index = False)


# In[ ]:




