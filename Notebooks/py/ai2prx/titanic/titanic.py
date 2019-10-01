#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
gender_submission = pd.read_csv('../input/gender_submission.csv')


# In[ ]:


gender_submission.head()


# In[ ]:


combine = pd.concat([train,test], axis=0, ignore_index=True)
combine


# In[ ]:


combine.info()


# **Lets look at Cabin Feature and it's corolation with survived**

# In[ ]:


c = combine[combine.Cabin.notnull()]


# In[ ]:


c.Cabin = c.Cabin.str.replace('\d+', '')


# In[ ]:


c.head()


# In[ ]:


for i in c.Cabin:
    if len(i) != 1:
        d = c[(c.Cabin == i)].index
        c = c.drop(d)
   


# In[ ]:


plt.subplots(figsize=(9,4))
sns.countplot('Cabin',hue="Survived", data=c,edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()


# **Drop Cabin Feature because it has numbers of missing values**

# In[ ]:


combine.drop('Cabin', axis=1, inplace=True)


# ***Lets look at Age Feature***

# In[ ]:


a = combine[combine.Age.notnull()]


# In[ ]:


a_s = a[(a.Survived == 1)]
a_d = a[(a.Survived == 0)]


# In[ ]:


a_d


# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot('Age',hue="Survived", data=a_d,edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of dead based on their age')
plt.show()


# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot('Age',hue="Survived", data=a_s,edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of survived based on their age')
plt.show()


# ***Fill missing values***

# In[ ]:


combine.Age = combine.Age.fillna(combine.Age.mean())


# In[ ]:


combine.Embarked = combine.Embarked.fillna(combine.Embarked.value_counts().index[0])


# ***Lets analys Name feature***

# In[ ]:


import re


# In[ ]:


train.Name


# In[ ]:


review = []
for i in range(0, len(combine.Name)):
    sentence = re.sub('[^a-zA-Z]', ' ', combine['Name'][i])
    sentence = sentence.lower()
    sentence = sentence.split()
    review.append(sentence)


# In[ ]:


for i in range (0, len(review)):
    for j in range(0, len(review[i])):
        if review[i][j] == 'mr' or review[i][j] == 'miss' or review[i][j] == 'mrs':
            review[i] = review[i][j]
            break;


# In[ ]:


combine.Name = review


# In[ ]:


combine.head()


# In[ ]:


combine.drop(['Ticket', 'PassengerId'], axis=1, inplace=True)


# In[ ]:


combine.info()


# In[ ]:


train = combine.iloc[0: 891, :]
test = combine.iloc[891:, :]


# In[ ]:


l = [test, train]
drop_rows = ''
indexs = ''
for i in l:
    drop_rows = i[(i.Name != 'miss') & (i.Name != 'mrs') & (i.Sex == 'female')].index.tolist()
    i.drop(drop_rows, inplace=True)
    
    indexs = i[(i.Name != 'mr') & (i.Sex == 'male')].index.tolist()
    i.loc[indexs, 'Name'] = 'mr'


# In[ ]:


X = train.iloc[:, :-1]
y = train.iloc[:, 8]


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()


# In[ ]:


for i in ['Embarked', 'Pclass', 'Sex', 'Name']:
    X[i] = labelencoder_X.fit_transform(X[i])


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


from sklearn.preprocessing import StandardScaler

X_sc = StandardScaler()
X_train = X_sc.fit_transform(X_train)
X_test = X_sc.transform(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]


# In[ ]:


for clf in classifiers:
    clf.fit(X_train, y_train)
    
    print("="*30)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}", acc)
    
    
print("="*30)


# In[ ]:




