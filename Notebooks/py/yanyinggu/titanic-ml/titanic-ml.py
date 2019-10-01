#!/usr/bin/env python
# coding: utf-8

# # Title: Titanic
# 
# Chapter 1: Analysis and Feature engineering
# 
# * Section 1.1 Data Exploration
# 
# * Section 1.2 Visualization
# 
# * Section 1.3 Missing value
# 
# * Section 1.4 Feature Engineering
# 
# Chapter 2: Fit the model
# 
# * Section 2.1 Prepare Data for Logistic Regression
# 
# * Section 2.2 Fit the model
# 
# * Section 2.3 Evaluation
# 
# * Section 2.4 Cross validation
# 
# * Section 2.5 Predict

# ### Introduction
# 
# Logistic regression, limited analysis and feature engineering

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)


# In[ ]:


import statsmodels.api as sm
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


# # Chapter 1 Analysis and Feature engineering

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


test.head()


# In[ ]:


test.info()


# In[ ]:


#Check whether there is missing value
train.isnull().sum()


# ### Section 1.1 Data Exploration

# In[ ]:


train.groupby('Survived').mean()
# to further check Pclass and Fare 


# In[ ]:


train.groupby('Pclass').mean()
# Higher class and higher Fare has higher chance to survive
# Use columns Pclass and Fare further in the analysis


# In[ ]:


train.groupby('Sex').mean()
# Female has higher chance to survive
# Use columns Sex further in the analysis


# In[ ]:


train.groupby('Embarked').mean()
# Embarked C has higher chance to survive
# Use columns Embarked further in the analysis


# In[ ]:


#Other columns: Name, Age, SibSp, Parch, Ticket, Cabin,


# ### Section 1.2 Visualization

# In[ ]:


get_ipython().magic(u'matplotlib inline')
train.Pclass.hist()
plt.title('Histogram of Pclass')
plt.xlabel('Class Level')
plt.ylabel('Frequency')


# In[ ]:


pd.crosstab(train.Pclass, train.Survived.astype(bool)).plot(kind='bar')
plt.title('Passenger Class Distribution by Survived Status')
plt.xlabel('Passenger Class')
plt.ylabel('Frequency')


# In[ ]:


sur_class = pd.crosstab(train.Pclass, train.Survived.astype(bool))
sur_class.div(sur_class.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Passenger Class Distribution by Survived Status')
plt.xlabel('Passenger Class')
plt.ylabel('Percentage')


# In[ ]:


#Correlation
correlation=train.corr()
plt.figure(figsize=(10, 10))

sns.heatmap(correlation, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');


# ### Section 1.3 Missing value
# 
# The following filling missing value methods reduces the accuracy of forecast.

# In[ ]:


# Column Age: Replace missing values with the mean
train.Age.fillna(np.mean(train.Age), inplace = True)
test.Age.fillna(np.mean(test.Age), inplace = True)
train.describe()


# In[ ]:


#Column Embarked
train[train['Embarked'].isnull()]


# In[ ]:


sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=test)


# In[ ]:


# For the two missing values, Fare is 80, Pclass is 1, it high chance that Embarked is C
train['Embarked'] = train['Embarked'].fillna('C')


# In[ ]:


# Drop rows with missing values
#train.dropna(axis=0, inplace = True)
# too many raws are dropped


# In[ ]:


#interpolate
from scipy import interpolate
train.interpolate(inplace = True)
train
#test.interpolate()


# In[ ]:


# Column Fare (only for test data set): Replace missing values with the mean
test.Fare.fillna(np.mean(test.Fare), inplace = True)


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


# For Column Cabin, there are too many missing values.


# ### Section 1.4 Feature Engineering

# In[ ]:


# Add new feature, Family
train['Family'] = train['SibSp'] + train['Parch'] + 1
test['Family'] = test['SibSp'] + test['Parch'] + 1


# In[ ]:


sur_Family = pd.crosstab(train.Family, train.Survived.astype(bool))
sur_Family.div(sur_Family.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Passenger Family Distribution by Survived Status')
plt.xlabel('Passenger Family')
plt.ylabel('Percentage')


# In[ ]:


# Column Embarked
# Factorize the values 
train_labels,train_levels = pd.factorize(train.Embarked)
test_labels,test_levels = pd.factorize(test.Embarked)
# Save the encoded variables in `iris.Class`
train.Embarked = train_labels
test.Embarked = test_labels
# Print out the first rows
train.head()


# In[ ]:


train_sex = pd.get_dummies(train['Sex'])
train2 = pd.concat([train, train_sex], axis=1)
test_sex = pd.get_dummies(test['Sex'])
test2 = pd.concat([test, test_sex], axis=1)


# In[ ]:


# filter columns
train3 = train2[['Survived', 'Pclass', 'female', 'Age', 'Family', 'Embarked']]
test3 = test2[['Pclass', 'female', 'Age', 'Family', 'Embarked']]


# In[ ]:


# logistical regression, X input has a column 'Intercept'
test3['Intercept'] = 1


# In[ ]:


correlation=train3.corr()
plt.figure(figsize=(10, 10))

sns.heatmap(correlation, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');


# # Chapter 2 Logistic Regresssion

# ### Section 2.1 Prepare Data for Logistic Regression

# In[ ]:


#splitting into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


y, X = dmatrices('Survived ~ Pclass + female + Age + Family + Embarked',
                  train3, return_type="dataframe")


# In[ ]:


X.head()
# logistical regression, X input has a column 'Intercept'


# In[ ]:


# flatten y into a 1-D array
y = np.ravel(y)


# ### Section 2.2 Fit the model

# In[ ]:


# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)

# check the accuracy on the training set
model.score(X, y)


# In[ ]:


# examine the coefficients
Co = pd.DataFrame(model.coef_)
Co.columns = list(X.columns)
Co


# ### Section 2.3 Evaluation

# In[ ]:


predicted_train = model.predict(X)
probs_train = model.predict_proba(X)


# In[ ]:


#Accuracy
metrics.accuracy_score(y, predicted_train)


# In[ ]:


metrics.roc_auc_score(y, probs_train[:, 1])


# In[ ]:


#confusion matrix: binary classification, 
#the count of true negatives is C_{0,0}, 
#false negatives is C_{1,0}, 
#true positives is C_{1,1} 
#and false positives is C_{0,1}.
metrics.confusion_matrix(y, predicted_train)


# In[ ]:


print(metrics.classification_report(y, predicted_train))


# ### Section 2.4 Cross validation

# In[ ]:


# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print(scores)
print(scores.mean())


# ### Section 2.5 Use the model to predict

# In[ ]:


#splitting into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


# predict class labels for the test set
X_test = test3
y_test = model.predict(X_test)
y_test = y_test.astype(int)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_test
    })
submission.to_csv('titanic.csv', index=False)


# In[ ]:




