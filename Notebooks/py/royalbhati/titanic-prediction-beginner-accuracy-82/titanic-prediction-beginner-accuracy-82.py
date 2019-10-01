#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting graphs
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set(color_codes=True)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Any results you write to the current directory are saved as output.


# In[ ]:


test2 = pd.read_csv('../input/test.csv')


# In[ ]:


train.head() #reading the data 


# In[ ]:


#By first look we can see that Cabin Paremeter has alpha numeric values and it does not seem to contribute


# In[ ]:


train.shape


# In[ ]:


train['Cabin'].isna().sum() # since cabin has lot of NaN Values and theres no way of filling those NaN so we gonna drop it


# In[ ]:


train.drop(['PassengerId','Cabin',],axis=1,inplace=True) #passenger id does not contribute in model building


# In[ ]:


test.drop(['PassengerId','Cabin'],axis=1,inplace=True)


# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# In[ ]:


#We have lot of missing values in Age column in both test and train data 
#Age can be a important factor in prediction since children adn senior citizens are first to be saved 


# # Handling Missing Values in Age column
# I have seen lot of approaches of handling mising values in this data set but what I am gonna do is 
# I am going to make a list of all the initials of the Name column and use them to fill missing values
# Since Name has no missing values we will get initial of all the people
# Master and Miss initial would be=0-10
# 'Mr.' or 'Mrs.'=20-60

# In[ ]:


import re
train_ini=[] # list for storing initials
for i in train['Name']:
    
    train_ini.append(re.findall(r"\w+\.",i))


# In[ ]:


import re
test_ini=[] 
for i in test['Name']:
    
    test_ini.append(re.findall(r"\w+\.",i))


# In[ ]:


np.where(train['Age'].isnull())#checking the indexes where age is null


# In[ ]:


train_null_age_index=[  5,  17,  19,  26,  28,  29,  31,  32,  36,  42,  45,  46,  47,
         48,  55,  64,  65,  76,  77,  82,  87,  95, 101, 107, 109, 121,
        126, 128, 140, 154, 158, 159, 166, 168, 176, 180, 181, 185, 186,
        196, 198, 201, 214, 223, 229, 235, 240, 241, 250, 256, 260, 264,
        270, 274, 277, 284, 295, 298, 300, 301, 303, 304, 306, 324, 330,
        334, 335, 347, 351, 354, 358, 359, 364, 367, 368, 375, 384, 388,
        409, 410, 411, 413, 415, 420, 425, 428, 431, 444, 451, 454, 457,
        459, 464, 466, 468, 470, 475, 481, 485, 490, 495, 497, 502, 507,
        511, 517, 522, 524, 527, 531, 533, 538, 547, 552, 557, 560, 563,
        564, 568, 573, 578, 584, 589, 593, 596, 598, 601, 602, 611, 612,
        613, 629, 633, 639, 643, 648, 650, 653, 656, 667, 669, 674, 680,
        692, 697, 709, 711, 718, 727, 732, 738, 739, 740, 760, 766, 768,
        773, 776, 778, 783, 790, 792, 793, 815, 825, 826, 828, 832, 837,
        839, 846, 849, 859, 863, 868, 878, 888]


# In[ ]:


np.where(test['Age'].isnull())#same for test data


# In[ ]:


test_null_age_index=[ 10,  22,  29,  33,  36,  39,  41,  47,  54,  58,  65,  76,  83,
         84,  85,  88,  91,  93, 102, 107, 108, 111, 116, 121, 124, 127,
        132, 133, 146, 148, 151, 160, 163, 168, 170, 173, 183, 188, 191,
        199, 200, 205, 211, 216, 219, 225, 227, 233, 243, 244, 249, 255,
        256, 265, 266, 267, 268, 271, 273, 274, 282, 286, 288, 289, 290,
        292, 297, 301, 304, 312, 332, 339, 342, 344, 357, 358, 365, 366,
        380, 382, 384, 408, 410, 413, 416, 417]


# In[ ]:


import random
from random import randint

train_newages=[]
for i in range(len(train_null_age_index)):
    if 'Mr.' or 'Mrs.' in train_ini[train_null_age_index[i]]:
        train_newages.append(random.randint(20,35))
                             
    elif 'Master.' or 'Miss.' in train_ini[train_null_age_index[i]]:
        train_newages.append(random.randint(10,18))


# In[ ]:


import random
from random import randint

testages=[]
for i in range(len(test_null_age_index)):
#     print(i[0])
    if 'Mr.' or 'Mrs.' in le[test_null_age_index[i]]:
        testages.append(random.randint(20,35))
                             
    elif 'Master.' or 'Miss.' in le[test_null_age_index[i]]:
        testages.append(random.randint(10,18))


# In[ ]:


for i in range(len(train_newages)):#assigning values to missing indexes
    train['Age'][train_null_age_index.pop()]=train_newages[i]


# In[ ]:


for i in range(len(test_null_age_index)):
    test['Age'][test_null_age_index.pop()]=testages[i]



# In[ ]:


train.drop(['Name','Ticket'],axis=1,inplace=True)#we don't need name and Ticket


# In[ ]:


test.drop(['Name','Ticket'],axis=1,inplace=True)


# **Now Lets begin analysing the Features **

# In[ ]:


train[['Parch','Survived']].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train[['SibSp','Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Fare', bins=50)


# **Frome above visualizations we are sure that these are all important features**

# In[ ]:


train.isnull().any()


# In[ ]:


#We need to fill values in Embarked


# In[ ]:


train['Embarked'].mode()


# In[ ]:


test['Embarked'].mode()


# In[ ]:


train['Embarked'].fillna('S',inplace=True)


# In[ ]:


test['Embarked'].fillna('S',inplace=True)


# In[ ]:


train.drop(['Fare'],axis=1,inplace=True)#dropping Fare because it does not contribute and has missing values


# In[ ]:


test.drop(['Fare'],axis=1,inplace=True)#dropping Fare because it does not contribute and has missing values


# In[ ]:


x=np.array(train.drop(['Survived','Embarked'],axis=1))# we are just dropping it for time being and we will add it later


# In[ ]:


test_x=np.array(test.drop(['Embarked'],axis=1))


# In[ ]:


test.isna().any()


# In[ ]:


train.isnull().any()


# In[ ]:


x


# In[ ]:


test_x


# In[ ]:


# we  have lot alphabetical values and we need to convert them into numerical categorical values


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:,1] = labelencoder_X.fit_transform(x[:,1])#selecting the column 
onehotencoder = OneHotEncoder(categorical_features= [1])#onehot encoder
x= onehotencoder.fit_transform(x).toarray()


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#Encodes the Categorical values into numerical values
test_x[:,1] = labelencoder_X.fit_transform(test_x[:,1])#selecting the column 
onehotencoder = OneHotEncoder(categorical_features= [1])#onehot encoder
test_x= onehotencoder.fit_transform(test_x).toarray()


# In[ ]:


x.shape


# In[ ]:


test_x.shape


# In[ ]:


emb=np.array(train.iloc[:,6]) # assigning embarked column from train data to emb variable


# In[ ]:


emb_x=np.array(test.iloc[:,5])# assigning embarked column from test data to emb variable


# In[ ]:


emb=emb.reshape(-1,1)


# In[ ]:


emb_x=emb_x.reshape(-1,1)


# In[ ]:


x=np.append(arr=x,values=emb,axis=1)#joining embarked column to x 


# In[ ]:


test_x=np.append(arr=test_x,values=emb_x,axis=1)#joining embarked column to test_x


# In[ ]:


test_x.shape,x.shape


# In[ ]:


labelencoder_emb = LabelEncoder()
#Encodes the Categorical values into numerical values
x[:,6] = labelencoder_X.fit_transform(x[:,6])#selecting the column 
onehotencoder = OneHotEncoder(categorical_features= [6])#onehot encoder
x= onehotencoder.fit_transform(x).toarray()


# In[ ]:


labelencoder_emb = LabelEncoder()
#Encodes the Categorical values into numerical values
test_x[:,6] = labelencoder_X.fit_transform(test_x[:,6])#selecting the column 
onehotencoder = OneHotEncoder(categorical_features= [6])#onehot encoder
test_x= onehotencoder.fit_transform(test_x).toarray()


# In[ ]:




g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Embarked')



# In[ ]:


x=x[:,:-1]


# In[ ]:


test_x=test_x[:,:-1]


# In[ ]:


y = np.array(train["Survived"])
test_x.shape,x.shape


# In[ ]:


from sklearn.model_selection import train_test_split

x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(x_tr, y_tr)
# random_forest.score(x_tr, y_tr)
etc=ExtraTreesClassifier(n_estimators=400)
etc.fit(x,y)
ypred=etc.predict(test_x)


# In[ ]:


# ypred=random_forest.predict(test_x)


# In[ ]:


passenger=test2['PassengerId']


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": passenger,
        "Survived": ypred
    })


# In[ ]:




