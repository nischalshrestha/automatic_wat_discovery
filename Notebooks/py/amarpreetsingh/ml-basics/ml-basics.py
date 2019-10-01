#!/usr/bin/env python
# coding: utf-8

# **<h1>Machine Learning - The Art of Making Computers Learn</h1>**

# ![](https://media.licdn.com/mpr/mpr/AAEAAQAAAAAAAAz2AAAAJDMzNThmYjA5LTM2ZWYtNDUzZC1iNDQyLTMxNzZkMWYyNGExOQ.jpg)

# <h2>Objective</h2>
# * Understand the possibilities and limitations of Machine Learning.
# * Understand the main ideas behind the most widely used learning algorithms in the industry.
# * How to build predictive models from data and analyze their performance.
# 
# <h2>What this session is and is not ?</h2>
# * Session is for audience with no or very little experience in ML.
# * Session will focus on applied ML and will not cover implementation of Algorithms.

# <h1>1.1 Demystifying artificial intelligence & machine learning.</h1>
# <h2>Artificial Intelligence</h2>
# <p>Artificial intelligence (AI) is an area of computer science that emphasizes the creation of intelligent machines that work and react like humans</p>
# <h2>Machine Learning</h2>
# <p>Machine Learning is a subfield within Artificial Intelligence that builds algorithms that allow computers to learn to perform tasks from data instead of being explicitly programmed.</p>
# **Machine + Learn **
# 
# <h3>Traditional Programming</h3>
# <br>
# ![](https://i.imgur.com/31Z2hX7.jpg)
# <h3>Machine Learning</h3>
# <br>
# ![](https://i.imgur.com/BLpEzg2.jpg)
# 

# <h3>AI vs ML vs Deep Learning</h3>
# <br>
# ![](https://media-exp2.licdn.com/mpr/mpr/AAEAAQAAAAAAAA1gAAAAJDA5MzlmNGJlLTg5YWMtNDU5MC1hYWQ5LWQ3YjU1ZDBhY2I4Zg.png)

# <h3>Why Machine Learning is taking off ?</h3>
# * Vast amount of Data
# * Faster computation power of computers
# * Improvement of the learning algorithms.

# <h3>The most common Problems ML can solve</h3>
# * Classification
# * Regression
# * Clustering

# <h2>Applications and Limitations to Machine Learning</h2>
# <h3>Applications</h3>
# 1. Virtual Personal Assistants
# 2. Predictions while Commuting (Driverless cars, traffic prediction etc )
# 3. Videos Surveillance (Crime detection)
# 4. Social Media Services (People recomendation, Face recoginition, Similar Pins etc )
# 5. Email Spam and Malware Filtering
# 6. Online Customer Support
# 7. Product Recommendations
# 8. Banking and financial sector
# 9. Medicine and Healthcare
# 
# <h3>Limitations</h3>
# * Require large amounts of hand-crafted, structured training data
# * No known one-model-fits-all solution exists.
# * Computational and technological barriers can limit real time testing and deployment of ML solutions.
# * ML algorithms does not understand context.
# 
# 

# <h2>There are three types of Machine Learning Algorithms</h2>
# <h3>Supervised Learning</h3>
# <p>The majority of practical machine learning uses supervised learning. Supervised learning is where you have input variables (X) and an output variable (Y ) and you use an algorithm to learn the mapping function from the input to the output. For example: Classification, Regression.</p>
# * Linear Regression
# * Logistic Regression
# * Random Forest
# * SVM
# 
# <h3>Unsupervised Learning</h3>
# <p>Unsupervised learning is where you you only have input data (X) and no corresponding output
# variables. The goal for unsupervised learning is to model the underlying structure or distribution
# in the data in order to learn more about the data. For example: Clustering, Association.</p>
# * K-Means
# * Apriori Algorithm
# 
# <h3>Semi Supervised Learning</h3>
# Problems where you have a large amount of input data (X) and only some of the data is labeled (Y ) are called semi-supervised learning problems. These problems sit in between both supervised and unsupervised learning. A good example is a photo archive where only some of the images are labeled, (e.g. dog, cat, person) and the majority are unlabeled.

# <h2>1.2 How do machines really learn?</h2>
# 
# ![](https://cdn-images-1.medium.com/max/2000/1*KzmIUYPmxgEHhXX7SlbP4w.jpeg)

# 

# ![](http://oi67.tinypic.com/14dddug.jpg)

# The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew.

# <h2>Tools used</h2>
# * **The Jupyter Notebook** - The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text.
# * **Python** - Python is a powerful high-level, object-oriented programming language created by Guido van Rossum.
# * **Pandas** - Pandas is an open source library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
# * **Matplotlib/ Seaborn** - These are Python visualization libraries.
# * **Scikit** - Scikit-learn is a free software machine learning library for the Python programming language.
# * **NumPy** - NumPy is a Python library for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

# <h2>1.3 Data really powers everything that we do.</h2>
# **Exploratory Data Analysis(EDA): **
# 1. Analysis of the features.
# 2. Finding any relations or trends considering multiple features.
# 
# **Feature Engineering and Data Cleaning: **
# 1. Adding any few features.
# 2. Removing redundant features.
# 3. Converting features into suitable form for modeling.
# 

# <h3> Import required Libraries</h3>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset into pandas DataFrame (2D data structure ).

# In[ ]:


df = pd.read_csv('../input/train.csv',encoding = "ISO-8859-1",low_memory=False)


# In[ ]:


df.head()


# <h3>Analysing The Features</h3>

# In[ ]:


df.columns.values


# <h3>Types Of Features</h3>
# 
# **Qualitative (Categorical Features)**
# A categorical variable is one that has two or more categories and each value in that feature can be categorised by them.
# * Nominal Variables - No relation between values.( Sex,Embarked)
# * Ordinal Features: Relative ordering or sorting between the values. ( PClass)
# 
# **Quantitative (Continous Feature/ Discrete)**
#  If a variable can take on any value between its minimum value and its maximum value, it is called a continuous variable; otherwise, it is called a discrete variable.
# Example: Age

# <h3>Data Exploration</h3>

# In[ ]:


# Did the gender affected survival chances? 
sns.countplot('Sex',hue='Survived',data=df)
plt.show()


# In[ ]:


# How about the Pclass
pd.crosstab(df.Pclass,df.Survived,margins=True).style.background_gradient(cmap='summer_r')


# <h3>Correlation Between The Features</h3>
# POSITIVE CORRELATION: If an increase in feature A leads to increase in feature B, then they are positively correlated. A value 1 means perfect positive correlation.
# 
# NEGATIVE CORRELATION: If an increase in feature A leads to decrease in feature B, then they are negatively correlated. A value -1 means perfect negative correlation.

# In[ ]:


sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# <h3>Feature Engineering and Data Cleaning </h3>

# In[ ]:


# Cleaning data
df.Age.isnull().sum()


# In[ ]:


mean_age = df.Age.mean()
mean_age


# In[ ]:


df.loc[df.Age.isnull(),'Age'] = mean_age
df.Age.isnull().sum()


# In[ ]:


# Age as a Categorical feature
df['Age_band']=0
df.loc[df['Age']<=16,'Age_band']=0
df.loc[(df['Age']>16)&(df['Age']<=32),'Age_band']=1
df.loc[(df['Age']>32)&(df['Age']<=48),'Age_band']=2
df.loc[(df['Age']>48)&(df['Age']<=64),'Age_band']=3
df.loc[df['Age']>64,'Age_band']=4
df.head(2)


# In[ ]:


#checking the number of passenegers in each band
df['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')


# In[ ]:


# Converting String Values into Numeric
df['Sex'].replace(['male','female'],[0,1],inplace=True)

# dropna() can used to remove values


# In[ ]:


df_train = df[['Survived','Pclass', 'Age_band', 'Sex']]


# In[ ]:


from sklearn.model_selection import train_test_split #training and testing data split

train,test=train_test_split(df_train,test_size=0.3,random_state=0,stratify=df_train['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=df_train[df_train.columns[1:]]
Y=df_train['Survived']
train_Y = np.ravel(train_Y)
test_Y=np.ravel(test_Y)


# <h3>Logistic regression</h3>
# 
# Logistic regression technique borrowed by machine learning from the field of statistics. It is the go-to method for binary classification problems. Logistic regression models the probability of the default class (the first class). 
# 
# For example, if we are modeling person as survived or not from their age, gender, fare etc. then the first class could be 'survived' and the logistic regression model could be written as the probability of 'survived' given a person’s age, gender, fare or more formally:
# 
# P (survived = age|gender|fare)
# 
# If the probability is greater than 0.5 we can take the output as a prediction for the default class (class 0), otherwise the prediction is for the other class (class 1).
# 
# Learning method of Logistic Regression is: 
# 
# **Y = W.X + B**
# 
# W = Weight
# B = Bais
# X = Input
# Y = Output
# 
# W and B are constant values and also called as coefficients.
# 
# The goal is to find the best estimates for the coefficients to minimize the errors in predicting y from x inputs.
# 

# <h3>Model Optimization</h3>
# Optimization is a big part of machine learning. Almost every machine learning algorithm has an optimization algorithm at it’s core. 
# 
# Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost). The goal is to continue to try different values for the coefficients, evaluate their cost and select new coefficients that have a slightly better (lower) cost. Repeating this process enough times will lead to the bottom of the bowl and you will know the values of the coefficients that result in the minimum cost.
# 
# Mathematic formula to update coefficient
# coefficient = coefficient − (alpha × delta)
# 
# alpha = Learning Rate (Ideal learning rate should be between 0.01 to 0.03)
# 
# Two types of Gradient Descent:
# Batch Gradient Descent
# Stochastic Gradient Descent

# <h3>Logistic Regression with sklearn</h3>

# In[ ]:


# LogisticRegression
from sklearn.linear_model import LogisticRegression


# In[ ]:


model = LogisticRegression()


# In[ ]:


model.fit(train_X,train_Y)


# In[ ]:



model.score(test_X,test_Y)


#  <h3>Cross validation</h3> 
#   Cross validation is process where we split the whole data several consecutive times in different train set and test set, and then return the averaged value of the prediction scores obtained with the different sets

# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score(model, X, Y, cv=5)


# **Overfitting**
# Overfitting refers to a model that models the training data too well.
# 
# **Underfitting** 
# Underfitting refers to a model that can neither model the training data nor generalize to new data.

# <h3> Few other Algorithms</h3>

# In[ ]:


# support vector Machine
from sklearn import svm
model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
model.fit(train_X,train_Y)
model.score(train_X,train_Y)


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_Y)
model.score(train_X,train_Y)


# In[ ]:


#k nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier() 
model.fit(train_X,train_Y)
model.score(train_X,train_Y)


# ![](https://www.kdnuggets.com/images/cartoon-machine-learning-what-they-think.jpg)

# In[ ]:




