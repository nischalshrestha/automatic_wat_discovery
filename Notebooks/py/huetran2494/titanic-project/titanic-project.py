#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
get_ipython().magic(u'matplotlib inline')
import os
print(os.listdir("../input"))

#open the files, formatted .csv
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

# Any results you write to the current directory are saved as output.


# In[ ]:


train1.pivot(index='Pclass',
            columns='Sex',
            values='Survived')


# In[ ]:


train1=train[['Pclass', 'Survived','Sex']].iloc[0:3,:]
train1.rename(columns={'Pclass':'pClass'}).set_index('Sex').rename_axis(None).T
train1


# **List of all FUNCTIONS**

# In[ ]:


# def male_female_child( passenger):
#     # Take the Age and Sex
#     age,sex = passenger # no need any more
#     # Compare the age, otherwise leave the sex
#     if passenger['Age']<16:
#         return 'child'
#     else:
#         return passenger['Sex']
# train[['Age','Sex']].apply(male_female_child, axis=1).head()
# train['Passenger']=train[['Age','Sex']].apply(male_female_child, axis=1)
# print(train.head())


# Function is used for plotting. bar_char function is used for investigating each features of datasets
# This step is processing
# 1. Observation data
# 2. Processing data (Check NA, Fill NaN (with mean or other methods), normalize)

# In[ ]:


alpha_color=0.5
# bins=[0,10,20,30,40,50,60,70,80]
# train['AgeBin']=pd.cut(train['Age'],bins)

#bar_char function
def bar_chart(feature):
    survived=train[train['Survived']==1][feature].value_counts(normalize=True).sort_index()
    dead=train[train['Survived']==0][feature].value_counts(normalize=True).sort_index()
    data=pd.DataFrame([survived, dead])
    data.index=['Survived', 'Dead']
    data.plot(kind='bar',alpha=alpha_color)
    
#facet_grid function
def facet_grid(feature,a, b):
    facet=sns.FacetGrid(train, hue='Survived', aspect=4)
    facet.map(sns.kdeplot, feature, shade=True)
    facet.set(xlim=(0, train[feature].max()))
    facet.add_legend()
    plt.xlim(a,b)
    plt.show()
    
#fillna one feature based on mean of other feature
def fill_na(groupbyfeature, fillnafeature):
    train[fillnafeature].fillna(train.groupby(groupbyfeature)[fillnafeature].transform('mean'), inplace=True)
    test[fillnafeature].fillna(train.groupby(groupbyfeature)[meanfeature].transform('mean'), inplace=True)
    
# drop column --> Works along each row --> axis =1
#work along each row: axis=1, work along each column axis=0
def drop_feature(dataframe,feature):
    dataframe=dataframe.drop(feature,axis=1)
    return dataframe


# Feature Engineering is the process of using domain knowledge of data to create features ( featers vectors) that make machine learning algorithms work
# Each column is a feature,  
# Change text into value, such as Male, Female into 0 or 1

# In[ ]:


#MAIN STEPS FOR PRE-PROCESSING DATA
# Firstly, Handle missing value, and fill in them
#Secondly, Normalize data: categories,...(formatted as number)
train_test_data=[train, test]
for dataset in train_test_data:
    dataset['Title']=dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    
#change title into group
title_mapping={"Mr":0, "Miss":1, "Mrs":2,
               "Master":3, "Dr":3, "Rev":3,
               "Col":3, "Major":3, "Mile":3,
               "Countess":3, "Ms":3, "Lady":3,
               "Jonkheer":3, "Don":3, "Dona":3, "Mme":3, "Capt":3, "Sir":3}
for dataset in train_test_data:
    dataset['Title']=dataset['Title'].map(title_mapping)
    
#Sex feature: change male and female into value 0 and 1
sex_mapping={'male':0, 'female':1}
for dataset in train_test_data:
    dataset['Sex']=dataset['Sex'].map(sex_mapping)
    
#Age: some age is missing --> use Title's meadian age for missing Age
train['Age'].fillna(train.groupby('Title')['Age'].transform("median"), inplace=True)
test['Age'].fillna(test.groupby('Title')['Age'].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset.loc[dataset['Age']<=16, 'Age']=0,
    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=26), 'Age']=1,
    dataset.loc[(dataset['Age']>26) & (dataset['Age']<=36), 'Age']=2,
    dataset.loc[(dataset['Age']>36) & (dataset['Age']<=62), 'Age']=3,
    dataset.loc[dataset['Age']>62, 'Age']=4
    
# Embarked feature
Pclass1=train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2=train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3=train[train['Pclass']==3]['Embarked'].value_counts()
df=pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index=('1st class','2nd class', '3rd class')
df.plot(kind='bar', stacked=True)
#The above code to visualize data -->S is the most --> Filling S for missing values
for dataset in train_test_data:
    dataset['Embarked']=dataset['Embarked'].fillna('S')
    
embarked_mapping={'S':0, 'C':1, 'Q':2}
for dataset in train_test_data:
    dataset['Embarked']=dataset['Embarked'].map(embarked_mapping)
    
#Fare feature: Filling missing fares with mean of fare, grouped by Pclass
train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'), inplace=True)
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'), inplace=True)
for dataset in train_test_data:
    dataset.loc[dataset['Fare']<=17, 'Fare']=0,
    dataset.loc[(dataset['Fare']>17) & (dataset['Fare']<=30), 'Fare']=1,
    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=100), 'Fare']=2,
    dataset.loc[dataset['Fare']>100, 'Fare']=3
    
#Cabin feature
# fill_na('Pclass', 'Cabin')
#Take the first letter in Cabin feature
for dataset in train_test_data:
    dataset['Cabin']=dataset['Cabin'].str[:1]
#mapping/categorize the name of cabin with numbers
cabin_mapping={'A':0, 'B':0.4, 'C':0.8, 'D':1.2, 'E':1.6, 'F': 2, 'G': 2.4, 'T':2.8}
for dataset in train_test_data:
    dataset['Cabin']=dataset['Cabin'].map(cabin_mapping)
    
train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'), inplace=True)

#Family size
train['FamilySize']=train['SibSp']+train['Parch']+1
test['FamilySize']=test['SibSp']+test['Parch']+1
family_mapping={1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2.0, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4}
for dataset in train_test_data:
    dataset['FamilySize']=dataset['FamilySize'].map(family_mapping)


# In[ ]:


train=drop_feature(train, ['PassengerId','Name','SibSp','Parch','Ticket'])
test= drop_feature(test, ['PassengerId','Name','SibSp','Parch','Ticket'])


# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(16,16))
plt.title('Correlation between Features', y=1.05, size = 20)
sns.heatmap(train.corr(),
            linewidths=0.1, 
            vmax=1.0, 
            square=True, 
            cmap=colormap, 
            linecolor='white', 
            annot=True)


# 5. Modelling 

# In[ ]:


#importing classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import numpy as np


# 6. Cross validation (K-fold)

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
k_fold=KFold(n_splits=10, shuffle=True, random_state=0)


# 6.2.1 kNN (k-nearest neighbors)

# In[ ]:


y_train=train[['Survived']]
x_train=train.drop(['Survived'], axis=1)

x_test=test
x_train.head()


# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree


# In[ ]:


train[640:645]


# In[ ]:


# x=train.iloc[:,1:9]
# y=train.iloc[:,0:1]
# x
train.mask(np.isinf(train))
train=train.fillna(0)


# In[ ]:


svc=SVC()
svc.fit(x_train, y_train)
svc.predict(x_test)
round(svc.score(x_train, y_train)*100,2)


# In[ ]:


#Feature scaling
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


# In[ ]:


classifier=KNeighborsClassifier(n_neighbors=28, p=2, metric='euclidean')
classifier.fit(x_train, y_train)
y_pred=classifier.predict(x_test)
acc_KNN=round(classifier.score(x_train, y_train)*100,2)
acc_KNN


# # classifier.fit(x_train, y_train)

# In[ ]:


lr=LinearRegression()
lr.fit(x_train, y_train)
y_pred=lr.predict(x_test)
lr.score(x_test, y_pred)


# In[ ]:


y_pred=classifier.predict(x_test)
y_pred[0:10]


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


n=train['Sex'].value_counts().index
app_mapping={}
i=1
for it in n:
    app_mapping[it]='Frequency_app_'+str(i)
    i=i+1
app_mapping


# In[ ]:


df.rename(index=str, columns=app_mapping)


# In[ ]:


df=train[['Pclass','Sex','Survived']].head(3).pivot(index='Pclass',
    columns='Sex',
    values='Survived')


# In[ ]:


df01=train[['Pclass','Sex','Survived']].head(3)


# In[ ]:


n=df01['Sex'].nunique() # how many distinct app this data have
value=df01['Sex'].value_counts().index #list all names of app
app_mapping={}
i=1
for it in value:
    app_mapping[it]='Frequency_app_'+str(i)
    i=i+1

df02=df01.pivot(index='Pclass',
    columns='Sex',
    values='Survived')

df02=df02.rename(index=str, columns=app_mapping)

print(df02)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




