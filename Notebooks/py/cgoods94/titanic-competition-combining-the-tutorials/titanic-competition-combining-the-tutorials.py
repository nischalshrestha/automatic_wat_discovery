#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

# preprocessing
from fancyimpute import KNN

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ## Loading the Data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')


# PassengerId won't have information about survival since it is abritrarily cast, and Cabin is too sparse to be reliable, so they will be discarded.

# In[ ]:


train.drop(['PassengerId', 'Cabin'], axis=1, inplace = True)
test = test_raw.drop(['Cabin'], axis = 1)

test.info()


# ## Single-Value Imputation
# 
# As is evident from the above info call, the Fare, Embarked, and Age variables have missing observations. I'm going to impute these using k-nearest neighbors. 
# 

# ### Embarked
# 
# Let's take a look at the distribution of Fare across the three ports from which the passengers Embarked.

# In[ ]:


full = pd.concat([train, test])
sns.boxplot(x = "Fare", y = "Embarked", data = full)


# Those outliers out of Cherbourg sure are annoying. Who paid over $500 for a ticket?
# 
# Anyway, let's now analyze the 2 passengers who are missing a port.

# In[ ]:


full[full.Embarked != full.Embarked]


# These two passengers paid \$80 for a first-class ticket, and it is in fact the same ticket number. Let's see which port is most likely to sell a first-class ticket for \$80.

# In[ ]:


sns.boxplot(x = "Fare", y = "Embarked", data = full[full.Pclass == 1])
plt.axvline(x = 80, color = 'r', linewidth = 3)


# It is most likely that these ladies sailed out of Cherbourg. Let's impute those values now.

# In[ ]:


train.loc[train.Embarked != train.Embarked, "Embarked"] = "C"
test.loc[test.Embarked != test.Embarked, "Embarked"] = "C"

full = pd.concat([train, test])
full.loc[(full.Fare == 80) & (full.Pclass == 1),:]


# ## Fare
# 
# Only one Fare value is missing. Let's check it out.

# In[ ]:


test[test.Fare != test.Fare]


# Ah, a passenger from the test set. This time, we'll just use the mean fare for a third-class passenger embarking from Southampton.

# In[ ]:


imp_fare = full.loc[(full.Embarked == 'S') & (full.Pclass == 3), "Fare"].mean() 

test.loc[test.Fare != test.Fare, "Fare"] = round(imp_fare, 2)
test.loc[(test.Name == "Storey, Mr. Thomas"),:]


# # One Hot Encoding
# 
# Before imputing the Age variable, it would be wise to one-hot encode the categorical Embarked and Sex variables. This will help with the age imputation by giving us more data for the k-nearest neighbors algorithm and it will be useful for the actual model training at the end.

# In[ ]:


full = pd.concat([train, test])
full.info()


# In[ ]:


# Embarked
embark_dummies_train  = pd.get_dummies(train['Embarked'])
embark_dummies_test = pd.get_dummies(test['Embarked'])

train = train.join(embark_dummies_train)
test = test.join(embark_dummies_test)

train.drop(['Embarked'], axis = 1, inplace = True)
test.drop(['Embarked'], axis = 1, inplace = True)

# Sex

sex_dummies_train = pd.get_dummies(train['Sex'])
sex_dummies_test = pd.get_dummies(test['Sex'])

train = train.join(sex_dummies_train)
test = test.join(sex_dummies_test)

train.drop(['Sex'], axis = 1, inplace = True)
test.drop(['Sex'], axis = 1, inplace = True)



# In[ ]:


test.info()


# ## Age
# 
# With 263 missing values, imputing this variable will require something more systematic than the other two. Specifically, I am going to employ k-nearest neighbors imputation. For k, I am going to use the square root of the sample size rounded to the nearest whole number.

# In[ ]:


k_train = int(np.sqrt(891))
k_test = int(np.sqrt(418))

train_features = train.drop(['Survived'], axis = 1).select_dtypes(include = [np.float, np.int])
test_features = test.select_dtypes(include = [np.float, np.int])

filled_ages_train = pd.DataFrame(KNN(k = k_train).complete(train_features)).loc[:,1]
filled_ages_test = pd.DataFrame(KNN(k = k_test).complete(test_features)).loc[:,1]

train.Age = round(filled_ages_train, 1)
test.Age = round(filled_ages_train, 1)

full = pd.concat([train, test])
full.info()


# ## Feature Engineering
# 
# ### Title
# 
# One feature I'd like to borrow from Megan Risdal's tutorial is the Title feature. This is extracted from the Name feature using regular expressions.
# 
# 

# In[ ]:


train_titles = train.Name.str.replace('(.*, )|(\\..*)', '').rename('Title')
train = train.join(train_titles)

test_titles = test.Name.str.replace('(.*, )|(\\..*)', '').rename('Title')
test = test.join(test_titles)


# In[ ]:


full = pd.concat([train, test])
full.groupby("Title").Title.count()


# Also going to lump the rare titles into one descriptor:

# In[ ]:


rare_title = ["Capt", "Col", "Don", "Dona", "Dr", "Jonkheer", 
              "Lady", "Major", "Rev", "Sir", "the Countess"]

train.loc[train.Title.isin(rare_title), "Title"] = "Rare"
test.loc[test.Title.isin(rare_title), "Title"] = "Rare"

full = pd.concat([train ,test])
full.groupby("Title").Title.count()


# And then we just have to fix the female title abbreviations to Mrs/Miss to have nice and tidy Title factors.

# In[ ]:


train.loc[train.Title.isin(["Mlle", "Ms"]), "Title"] = "Miss"
train.loc[train.Title == "Mme", "Title"] = "Mrs"

test.loc[test.Title.isin(["Mlle", "Ms"]), "Title"] = "Miss"
test.loc[test.Title == "Mme", "Title"] = "Mrs"

full = pd.concat([train ,test])
full.groupby("Title").Title.count()


# Let's one-hot encode these since there are only five of them.

# In[ ]:


title_dummies_train  = pd.get_dummies(train['Title'])
title_dummies_test = pd.get_dummies(test['Title'])

train = train.join(title_dummies_train)
test = test.join(title_dummies_test)

train.drop(['Title'], axis = 1, inplace = True)
test.drop(['Title'], axis = 1, inplace = True)

full = pd.concat([train ,test])
full.describe()


# ### Family Size
# 
# We can make a Family Size variable from Parch and Sibsp.
# 

# In[ ]:


train_fsize = train.Parch + train.SibSp
train = train.join(train_fsize.rename('Fsize'))

test_fsize = test.Parch + test.SibSp
test = test.join(test_fsize.rename('Fsize'))

full = pd.concat([train ,test])
full.describe()


# ### Child
# 
# Let's make a variable for whether or not the passenger was a child.

# In[ ]:


train_child = train.Age < 16
train = train.join(train_child.rename('Child'))

test_child = test.Age < 16
test = test.join(test_child.rename('Child'))

full = pd.concat([train ,test])
full.groupby("Child").Child.count()


# In[ ]:


test.info()


# ## Model Building
# 
# We'll go by Omar El Gabry's Python tutorial for this portion.

# In[ ]:


X_train = train.drop(["Survived", "Name", "Ticket"],axis=1)
Y_train = train["Survived"]
X_test  = test.drop(["PassengerId", "Name", "Ticket"], axis = 1)


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)


# In[ ]:


# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('pySubmission.csv', index=False)

