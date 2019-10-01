#!/usr/bin/env python
# coding: utf-8

# ## import libraries and read files

# In[273]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

from collections import Counter

#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import linear_model
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


# In[274]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# ### outliers in train

# In[275]:


def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])

train.loc[Outliers_to_drop]


# In[276]:


train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True) #delete Outliers in train
train.shape


# ### concat datasets

# In[277]:


ntrain = train.shape[0] #set for prediction model
ntest = test.shape[0]
Y_train = train.Survived.values


# In[278]:


all_data = pd.concat((train, test)).reset_index(drop=True) #combine datasets
all_data.drop(['Survived'], axis=1, inplace=True)
all_data.info()


# 
# ### missing values

# In[279]:


all_data = all_data.replace(np.inf, np.nan) # important
all_data_na = (all_data.isnull().sum() / len(all_data))
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# In[280]:


all_data["Cabin"] = all_data["Cabin"].fillna("None")
all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].mode()[0])
all_data["Age"] = all_data["Age"].fillna(all_data['Age'].median()) 
all_data["Fare"] = all_data["Fare"].fillna(all_data['Fare'].median())


# In[281]:


all_data = all_data.replace(np.inf, np.nan) # important
all_data_na = (all_data.isnull().sum() / len(all_data))
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# # features

# In[282]:


all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['IsAlone'] = 1 #initialize to yes/1 is alone
all_data['IsAlone'].loc[all_data['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1
all_data['Title'] = all_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
all_data['FareBin'] = pd.qcut(all_data['Fare'], 5)
all_data['AgeBin'] = pd.cut(all_data['Age'].astype(int), 5)

all_data["Title"] = all_data["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data["Title"] = all_data["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
all_data["Title"] = all_data["Title"].astype(int)


# In[283]:


all_data.head()


# In[284]:


all_data.drop(columns=['Name','Age','Fare','Parch','SibSp','Ticket'],inplace=True)


# In[285]:


symbol =[]
for each in all_data['Cabin']:
    if each != 'None':
        symbol.append(each[0])
    else:
        symbol.append('None')
all_data['Cabin']=symbol
all_data['Cabin'].unique()


# In[286]:


all_data.head()


# In[287]:


all_data = pd.get_dummies(all_data, columns = ['Embarked','Cabin','Pclass',"FareBin",'AgeBin','Title','FamilySize','Sex'])


# In[288]:


all_data.head()


# ## modeling

# In[289]:


X_train = all_data[:ntrain]
X_test = all_data[ntrain:]


# In[290]:


from sklearn.cross_validation import KFold, cross_val_score


# In[296]:


kfold = StratifiedKFold(n_splits=10)
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_


# In[297]:


RFC_best


# In[298]:


RFC_best.fit(X_train, Y_train)
y_test = RFC_best.predict(X_test) 


# In[299]:


cross_val_score(forest, X_test, y_test ,cv=10).mean()


# In[300]:


len(y_test)


# In[301]:


sub = pd.DataFrame()
sub['PassengerId'] = X_test["PassengerId"]
sub['Survived'] = y_test
sub.to_csv('submission.csv',index=False)


# In[ ]:




