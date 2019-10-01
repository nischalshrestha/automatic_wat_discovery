#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
get_ipython().magic(u'matplotlib inline')


# In[ ]:


# read CSV file as DataFrame
train = pd.read_csv('../input/train.csv', index_col=0)
test = pd.read_csv('../input/test.csv', index_col=0)

# display the first 5 rows
train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()
print ('-------------------------------------')
test.info()


# ## Preprocessing data

# In[ ]:


all_data = pd.concat((train.loc[:, :],
                      test.loc[:, :]))

all_data.shape


# In[ ]:


# Create new feature FamllySize as a combination of SibSp and Parch
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
# Create new feature IsAlone from FamilySize
all_data['IsAlone'] = 0
all_data.loc[all_data['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS
all_data['Embarked'] = all_data['Embarked'].fillna('S')
all_data['Fare'] = all_data['Fare'].fillna(train['Fare'].median())   
# Mapping Sex, Embarked
all_data['Sex'] = all_data['Sex'].map( {'female':0, 'male':1} ).astype(int)
all_data['Embarked'] = all_data['Embarked'].map( {'S':0, 'C':1, 'Q':2} ).astype(int)
# Mapping Fare
    all_data.loc[ all_data['Fare'] <= 7.91, 'Fare'] = 0
    all_data.loc[(all_data['Fare'] > 7.91) & (all_data['Fare'] <= 14.454), 'Fare'] = 1
    all_data.loc[(all_data['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    all_data.loc[ all_data['Fare'] > 31, 'Fare'] 							        = 3
    all_data['Fare'] = all_data['Fare'].astype(int)


# In[ ]:


# drop unnecessary columns
drop_elements = ['Name', 'Ticket', 'Cabin']
all_data = all_data.drop(drop_elements, axis=1)

all_data.info()


# In[ ]:


# visualize the relationship between the features and the response using scatterplots
sns.pairplot(train, x_vars=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male'], 
             y_vars='Survived', kind='reg')


# In[ ]:


# define training and testing sets
X_train = all_data[:891]
Y_train = all_data[:891]['Survived']
X_test = all_data[891:]

print (X_train.shape, Y_train.shape, X_test.shape)


# In[ ]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
logreg.score(X_train, Y_train)


# In[ ]:


submission = pd.Dataframe({
        "PassengerId": test['PassengerId'],
        "Survived": Y_pred
})
submission.to_csv('titanic.csv', index=False)

