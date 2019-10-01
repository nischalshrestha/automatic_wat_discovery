#!/usr/bin/env python
# coding: utf-8

# # Random Forrest Pipeline

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# 
# ## Import Data and Overview

# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# ## EDA and Cleaning

# In[ ]:


# Replace null values with placeholders
df = df.fillna(value = {'Age' : 150,       # Unknown age may be relevant
                        'Cabin' : 'U',     # Unknown cabin may be relevant
                        'Embarked' : 'S'}, # Most common embark is 'S'
               inplace=True)


# In[ ]:


# Get Cabin Letter

df['Cabin_Letter'] = df['Cabin'].apply(lambda x: x[0])


# In[ ]:


# Change ages < 1 to 0
df['Age'] = np.where(df['Age'] < 1, 0, df['Age'])

# Estimated ages are XX.5
df['Age_Known'] = df['Age'].apply(lambda x: 1 if x % 1.0 == 0 else 0)


# In[ ]:


# View % survived by feature subclasses
def pct_survived(feature): 
    survived = df.groupby(feature).agg({'Survived' : 
                                   {'Passengers' : 'count',
                                   'Survived' : 'sum'}}).reset_index()
    survived['Pct_Survived'] = survived['Survived', 'Survived'] /                                survived['Survived', 'Passengers']
    sns.barplot(x=feature, y='Pct_Survived', data=survived)
    plt.plot


# In[ ]:


pct_survived('Pclass')


# In[ ]:


pct_survived('Sex')


# In[ ]:


pct_survived('SibSp')


# In[ ]:


pct_survived('Parch')


# In[ ]:


pct_survived('Age')


# In[ ]:


pct_survived('Cabin_Letter')


# In[ ]:


pct_survived('Embarked')


# In[ ]:


pct_survived('Fare')


# In[ ]:


pct_survived('Age_Known')


# In[ ]:


# Create dummy columns and remove unnecessary features
data = pd.get_dummies(df, columns=['Sex', 'Cabin_Letter', 'Embarked'], 
                      drop_first=True)
data.drop(['PassengerId', 'Name', 'Ticket', 'Fare',
           'Cabin', 'Parch', 'SibSp'], axis=1, inplace=True)


# In[ ]:


# Investigate Correlations between features and target
data.corr()['Survived'].sort_values()


# ## Modeling Pipeline

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


# Create train and test features and targets
X = data.drop(['Survived'], axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


# Create Pipeline
clf = RandomForestClassifier()

steps = [('random_forest', clf)] 

pipeline = Pipeline(steps)

parameters = dict(random_forest__n_estimators = [50, 100, 250],
                  random_forest__min_samples_leaf = [2, 5, 10])

cv = GridSearchCV(pipeline, param_grid=parameters)

cv.fit(X_train, y_train)

y_predictions = cv.predict(X_test)

cr = classification_report(y_test, y_predictions)
                  
cm = pd.DataFrame(confusion_matrix(y_test, y_predictions),
                  columns = ['Pred_Died', 'Pred_Surv'],
                  index = ['Died', 'Survived'])

best_params = cv.best_params_


# ## Results

# In[ ]:


print (cm)
print ()
print (cr)


# In[ ]:


best_params


# ## Create Output

# In[ ]:


df = pd.read_csv('../input/test.csv')

# Replace null values with placeholders
df_test = df.fillna(value = {'Age' : 150,       # Unknown age may be relevant
                        'Cabin' : 'U',     # Unknown cabin may be relevant
                        'Embarked' : 'S'}, # Most common embark is 'S'
               inplace=True)

# Get Cabin Letter

df['Cabin_Letter'] = df['Cabin'].apply(lambda x: x[0])

# Change ages < 1 to 0
df['Age'] = np.where(df['Age'] < 1, 0, df['Age'])

# Estimated ages are XX.5
df['Age_Known'] = df['Age'].apply(lambda x: 1 if x % 1.0 == 0 else 0)

# Create dummy columns and remove unnecessary features
data = pd.get_dummies(df, columns=['Sex', 'Cabin_Letter', 'Embarked'], 
                      drop_first=True)

data.drop(['PassengerId', 'Name', 'Ticket', 'Fare',
           'Cabin', 'Parch', 'SibSp'], axis=1, inplace=True)

# test data didn't have Cabin_Letter_T columns
data['Cabin_Letter_T'] = 0

# Output predictions
df['Survived'] = cv.predict(data)
output = df[['PassengerId', 'Survived']]
output.to_csv('../working/kernel_1_output.csv', index=False)


# In[ ]:




