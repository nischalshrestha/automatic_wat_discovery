#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize']=10,10
sns.set(style='white')

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train.corr()


# In[ ]:


sns.pairplot(train.drop("PassengerId", axis=1).dropna(), size=2,kind='reg',hue='Survived')


# In[ ]:


print('Min Age: ' + str(train.Age.min()) + ' Max Age: ' + str(train.Age.max()))


# In[ ]:


plt.subplot(311)
sns.distplot(train[train.Survived==1]['Age'].dropna().values, bins=80, kde=True,color='green')
sns.distplot(train[train.Survived==0]['Age'].dropna().values, bins=80, kde=True,color='red')
plt.subplot(312)
plt.hist([train[train.Survived == label].Age.dropna().values for label in [0,1]],          stacked=True,bins=80,color=['red','green'], label=['red','green'])
plt.show()


# In[ ]:


plt.subplot(311)
sns.countplot(x='Pclass', data=train[train.Survived==0], palette=sns.light_palette("purple"),hue="Sex")
plt.subplot(312)
sns.countplot(x='Pclass', data=train[train.Survived==1], palette=sns.light_palette("green"),hue="Sex")


# In[ ]:


sns.lmplot(data=train,x='Fare',y='Survived',fit_reg=False)


# In[ ]:


cleanedtrain = train[train.Fare < 350].copy()
plt.scatter(x=np.log10(cleanedtrain.Fare.values),y=cleanedtrain.Survived.values)


# In[ ]:


g = sns.boxplot(x="Pclass", y="Fare", hue="Survived", data=cleanedtrain, palette="Paired")


# In[ ]:


from sklearn import preprocessing


le1 = preprocessing.LabelEncoder()
le1.fit(train.Sex.unique())
le2 = preprocessing.LabelEncoder()
le2.fit(train.Pclass.unique())

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(cleanedtrain.as_matrix(columns=['Fare']))

feature1=min_max_scaler.transform(cleanedtrain.as_matrix(columns=['Fare']))
feature2=le1.transform(cleanedtrain.as_matrix(columns=['Sex']))
feature3=le2.transform(cleanedtrain.as_matrix(columns=['Pclass']))

temp = np.column_stack((feature2,feature3))
    
enc = preprocessing.OneHotEncoder()
enc.fit(temp)
trainfeatures = enc.transform(temp)
trainfeatures = np.append(trainfeatures.toarray(), feature1, 1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

Y = cleanedtrain.as_matrix(columns=['Survived'])
    
clf = RandomForestClassifier(n_estimators=5)
labels = clf.fit(trainfeatures,Y)


# In[ ]:


feature1=min_max_scaler.transform(train.as_matrix(columns=['Fare']))
feature2=le1.transform(train.as_matrix(columns=['Sex']))
feature3=le2.transform(train.as_matrix(columns=['Pclass']))

temp = np.column_stack((feature2,feature3))
trainfeatures = enc.transform(temp)
trainfeatures = np.append(trainfeatures.toarray(), feature1, 1)
                       
Y = train.as_matrix(columns=['Survived'])

clf.score(trainfeatures,Y)


# In[ ]:


test[test['Fare'].isnull()]


# In[ ]:


test['Fare'].iat[152] = test.Fare.mean()


# In[ ]:


#min_max_scaler = preprocessing.MinMaxScaler()
#min_max_scaler.fit(test.as_matrix(columns=['Fare']))

feature1=min_max_scaler.transform(test.as_matrix(columns=['Fare']))
feature2=le1.transform(test.as_matrix(columns=['Sex']))
feature3=le2.transform(test.as_matrix(columns=['Pclass']))

temp = np.column_stack((feature2,feature3))
testfeatures = enc.transform(temp)
testfeatures = np.append(testfeatures.toarray(), feature1, 1)

Y_pred = clf.predict(testfeatures)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"].values,
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:




