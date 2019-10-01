#!/usr/bin/env python
# coding: utf-8

# > Last update: 1/11/2017

# **1. Import datasets & packages**

# In[ ]:


import numpy as np
import pandas as pd
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# **2. Some EDA**

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.tail()


# Check data type

# In[ ]:


train_df.info()


# This only contains summary for numerical columns:

# In[ ]:


train_df.describe()


# This would have other columns which are "object":

# In[ ]:


train_df.describe(include=['O'])


# **3. Clean data**

# Check if there's missing value. Looks like there are missing values in "Age", "Cabin", and " Embarked". However, since I don't think "Embarked" and "Cabin" have a lot of impact on the chance of survival, I will only fill missing values for "Age", and remove "Embarked" and "Cabin" in the end.

# In[ ]:


train_df.isnull().any()


# For "Age" column, first plot histogram to see the skewness of the distribution. Looks like it's a little right skewed, so the best way to fill missing values might be using the median instead of mean.

# In[ ]:


train_df['Age'].plot(kind='hist')


# In[ ]:


train_df['Age'].fillna(train_df['Age'].median(),inplace=True)


# Drop the columns that will not be used for modeling.

# In[ ]:


train_df.drop(['PassengerId','Name','Ticket','Embarked','Cabin'], axis=1,inplace=True)


# Check if test_df has missing values, and fill missing values. Drop unused columns too.

# In[ ]:


test_df.isnull().any()


# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(),inplace=True)


# In[ ]:


test_df.drop(['Name','Ticket','Embarked','Cabin'], axis=1,inplace=True)


# **4. Check correlation between variables**

# Sex vs Survived. As we can see from the bar chart, the survival rate of female is much higher than male.

# In[ ]:


import seaborn as sns
sns.barplot(x='Sex', y='Survived', data=train_df)


# Pclass vs Survived. The higher the class, the higher the survival chance.

# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=train_df)


# Heatmap

# In[ ]:


import matplotlib.pyplot as plt
plt.subplots(figsize=(10,10))
ax = plt.axes()
ax.set_title("Correlation Heatmap")
corr = train_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,cmap='autumn')


# **5. Build prediction model**

# In[ ]:


import statsmodels.formula.api as sm


# Check linear model first. Clearly, linear model would not fit since we are only expecting 2 outcomes (0 or 1).

# In[ ]:


lm = sm.ols(formula='Survived~Pclass+Sex+Age+SibSp+Parch+Fare', data=train_df).fit()
lm.summary()


# Logistic regression model. To use logistic regression model, categorical columns should be changed to numerical value. So changing Sex to 0 or 1.

# In[ ]:


train_df['Sex'].replace(['male','female'],[0,1],inplace=True)
test_df['Sex'].replace(['male','female'],[0,1],inplace=True)


# In[ ]:


X_train = train_df.drop("Survived",axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred1 = logreg.predict(X_test)
logreg.score(X_train, Y_train)


# Decision tree model.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred2 = decision_tree.predict(X_test)
decision_tree.score(X_train, Y_train)


# K-NN model.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier() 
knn.fit(X_train,Y_train)
Y_pred3=knn.predict(X_test)
knn.score(X_train, Y_train)


# 

# Decision Tree > KNN > Logistic Regression. However, the scores I got for submission are Logistic Regression > KNN > Decision Tree. So I will stay using Logistic Regression for now.

# **6. Submission file**

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred1
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:




