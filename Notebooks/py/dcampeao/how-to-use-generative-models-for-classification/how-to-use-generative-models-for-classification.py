#!/usr/bin/env python
# coding: utf-8

# # Classification with the Generative Model

# The goal of this notebook is to use generative models to predict the Titanic survivors.
# 
# We will create a Multi-Gaussian distribution and use probabilities to predict the results. No black boxes!
# 
# The idea is to calculate the probabilty (pi) of each label (survived or not) and multiply it by the corresponding value of the probability density function of the other features (P(feature)). 
# 
# So, for each person we will have 2 pi*P(feature) values. The largest one will be the chosen label.

# ## Load the data 

# Load libraries and data.

# In[23]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from scipy.stats import norm


# In[2]:


# Load data set.
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')


# First, let's take a look at the dataset.

# In[3]:


data_train.head()


# In[4]:


data_train.describe()


# The data need some adjustments. The functions below will do the job.

# In[5]:


def simplify_ages(df):
    #Get titles
    unknow = df['Age'].isnull()
    master_title = df['Title'] == 'Master'
    mr_title = df['Title'] == 'Mr'
    mrs_title = df['Title'] == 'Mrs'
    miss_title = df['Title'] == 'Miss'
    dr_title = df['Title'] == 'Dr'
    
    #Fill the missing ages with the mean age of the corresponding title
    df.loc[master_title & unknow,'Age'] = df[master_title]['Age'].mean()
    df.loc[mr_title & unknow,'Age'] = df[mr_title]['Age'].mean()
    df.loc[mrs_title & unknow,'Age'] = df[mrs_title]['Age'].mean()
    df.loc[miss_title & unknow,'Age'] = df[miss_title]['Age'].mean()
    df.loc[dr_title & unknow,'Age'] = df[dr_title]['Age'].mean()

    df.Age = df.Age.fillna(-0.5)
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(',')[0])
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    return df    

def simplify_family(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df

def simplify_embarks(df):
    df.Embarked = df.Embarked.fillna('N')
    df.Embarked = df.Embarked.apply(lambda x: x[0])
    return df

def simplify_titles(df):
    #Groups together some titles
    df['Title'] = df['Title'].replace(['Col','Dr','Major'], 'MidNoble')
    df['Title'] = df['Title'].replace(['Mlle','Ms','Mme','Countess','Sir', 'Lady'], 'Noble')
    df['Title'] = df['Title'].replace(['Don','Jonkheer'], 'Mr')
    df['Title'] = df['Title'].replace(['Capt','Rev'], 'Worker')
    return df

def create_sex_class(df):
    #Combine gender and class to create a new feature
    male = df['Sex']=='male'
    female = df['Sex']=='female'
    Class1 = df['Pclass']==1
    Class2 = df['Pclass']==2
    Class3 = df['Pclass']==3

    df.loc[female & Class1,'Class'] = (0)
    df.loc[female & Class2,'Class'] = (1)
    df.loc[female & Class3,'Class'] = (2)
    df.loc[male & Class1,'Class'] = (3)
    df.loc[male & Class2,'Class'] = (4)
    df.loc[male & Class3,'Class'] = (5)
    
    return df

def drop_features(df):
    #Drop features that were combined or are not very usefull
    return df.drop(['Ticket', 'SibSp', 'Parch', 'Name', 'Lname', 'Pclass'], axis=1)

def transform_features(df):
    df = format_name(df)
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = simplify_family(df)
    df = simplify_embarks(df)
    df = simplify_titles(df)
    df = create_sex_class(df)
    df = drop_features(df)
    return df

def encode_features(df_train, df_test):
    #Features that are not numeric must be converted before we can make statistics on them.
    features = ['Cabin', 'Title', 'Embarked', 'Sex']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test


# In[6]:


data_train = transform_features(data_train)
data_test = transform_features(data_test)


# In[7]:


data_train, data_test = encode_features(data_train, data_test)


# Spliting the data to train our model

# In[8]:


X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']


# First, let's take a look at the normal distributions. The Age, for instance, seems to be well represented by its Gaussian distribution.

# In[10]:


sns.distplot(X_all.Age.values, label = "Age")
x = np.random.normal(X_all.Age.mean(), X_all.Age.std(), size=1000)
sns.distplot(x, label="Gaussian Age")
plt.legend()


# The Fares however, are not very well represented.

# In[11]:


sns.distplot(X_all.Fare.values, label = "Fare")
x = np.random.normal(X_all.Fare.mean(), X_all.Fare.std(), size=1000)
sns.distplot(x, label="Gaussian Fare")


# Function to extract the mean vector em covariance matrix:

# In[13]:


def MultivariateGaussian(x,y):
    k = 2  # labels 1,2,...,k
    d = (x.shape)[1]  # number of features
    mu = np.zeros((k,d))
    sigma = np.zeros((k,d,d))
    pi = np.zeros(k)
    for label in range(2):
        indices = (y == label)
        mu[label] = np.mean(x[indices,:], axis=0)
        sigma[label] = np.cov(x[indices,:], rowvar=0, bias=1)
        pi[label] = float(sum(indices))/float(len(y))
    return mu, sigma, pi


# In[14]:


mu, sigma, pi = MultivariateGaussian(X_all.values,y_all.values)


# ## Testing the model

# In[15]:


# Now test the performance of a predictor based on a subset of features
def test_model(mu, sigma, pi, features, tx, ty):   
    preds = []
    errors = 0
    for x,y in zip(tx,ty):
        piP_list = []
        for label in range(2):
            S = np.linalg.inv(sigma[label])
            dS = np.linalg.det(sigma[label])
            xTsigmax=0
            for i in features:
                for j in features:
                    xTsigmax += S[i][j]*(x[i]-mu[label][i])*(x[j]-mu[label][j])
            piP_list.append(pi[label]*np.exp(-0.5*xTsigmax)/np.sqrt(dS))
            
        predict = np.argmax(piP_list)
        
        preds.append(predict)
        if predict != y:
            errors+=1
        
    return errors, preds


# In[16]:


data_test.tail()


# Using this method, the accuracy in the training data we get is:

# In[17]:


errors, preds = test_model(mu, sigma, pi, [0,1,2,3,4,5,6,7], X_all.values, y_all.values)
print('Accuracy for training data:', (1-errors/len(y_all))*100, '%' )


# Which is a good estimate of the test accuracy of this notebook which is 0.78468.

# ## Calculate Predictions

# In[18]:


def write_predictions(mu, sigma, pi, features, tx):   
    preds = []
    for x in tx:
        piP_list = []
        for label in range(2):
            S = np.linalg.inv(sigma[label])
            dS = np.linalg.det(sigma[label])
            xTsigmax=0
            for i in features:
                for j in features:
                    xTsigmax += S[i][j]*(x[i]-mu[label][i])*(x[j]-mu[label][j])
            piP_list.append(pi[label]*np.exp(-0.5*xTsigmax)/np.sqrt(dS))
            
        predict = np.argmax(piP_list)
        
        preds.append(predict)
        
    return preds


# In[19]:


preds = write_predictions(mu, sigma, pi, [0,1,2,3,4,5,6,7], data_test.drop('PassengerId',axis=1).values)


# In[20]:


data_pred = pd.DataFrame()
data_pred['PassengerId']=data_test['PassengerId']
data_pred['Survived']=preds


# In[21]:


data_pred.head()


# ## Write results

# In[22]:


data_pred.to_csv('predictions_generative_multGauss.csv', index = False)

