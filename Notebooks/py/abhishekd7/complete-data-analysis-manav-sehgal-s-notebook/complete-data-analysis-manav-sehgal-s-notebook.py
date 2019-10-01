#!/usr/bin/env python
# coding: utf-8

# *Steps involved in the prediction(binary Classification) Process:-*
# ===================================================================
# 
# 1. Importing the Dataset (Training and Testing) into Dataframe.
# 2. Visualise the Dataframe

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


# **Step-1 Import Dataset into Dataframes and Visualizing the Dataset**
# ---------------------------------------------------------------------

# In[ ]:


#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Load Data into Dataframes
Data_train = pd.read_csv("../input/train.csv")
Data_test = pd.read_csv("../input/test.csv")

Data_full = [Data_train,Data_test]

Data_train.shape


# In[ ]:


#Preview the Data
Data_train.head()


# In[ ]:


Data_train.describe()


# In[ ]:


#Understanding the Data

Data_train.info()
print("************")
print("************")
Data_test.info()


# From the above Information we can understand that:-
# 
# *Training Data* -
# 
#  1.  Features "age" and "cabin" & "Embarked" have missing values (Total =891)[Age = 714 , Cabin = 204 , Embarked = 889] 
# 
# *Testing Data* -
# 
#  1.  Features "age" and "cabin" have missing values (Total =418)[Age = 332 , Cabin = 91] 
# 
# Therefore, we need to take care of the missing values before proceeding for further analysis.

# In[ ]:



#For "Embarked" feature only 2 missing values so we will fill it with the value occuring thr most which is S
Data_full["Embarked"] = Data_full["Embarked"].fillna("S")

print (Data_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())


# In[ ]:


#Fare also has some missing value in Test Data and we will replace it with the median. then we categorize it into 4 ranges.
Data_test['Fare'] = Data_test['Fare'].fillna(Data_test['Fare'].median())

Data_train['CategoricalFare'] = pd.qcut(Data_test['Fare'], 4)
print (Data_train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())


# 
#  Age
# ------------
# 
# We have plenty of missing values in this feature. # generate random numbers between (mean - std) and (mean + std). then we categorize age into 5 range.

# In[ ]:


for dataset in Data_full:
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
Data_train['CategoricalAge'] = pd.cut(Data_train['Age'], 5)

print (Data_train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())


# In[ ]:


#Pearson Correlation Heatmap
#let us generate some correlation plots of the features to see how related one feature is to the next. To do so, we will utilise the Seaborn plotting package which allows us to plot heatmaps very conveniently as follows

colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(Data_train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# *Machine Learning*
# ================

# In[ ]:


# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:




