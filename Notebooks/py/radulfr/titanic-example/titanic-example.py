#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


traindf = pd.read_csv("../input/train.csv")
testdf = pd.read_csv("../input/test.csv")
gsdf = pd.read_csv("../input/gender_submission.csv")


# In[ ]:


traindf.head()


# ### Variable explanation
# * PassengerId: simple ID for each passenger
# * Survived: {1, 0} survived or perish to the titanic sinking
# * Pclass: A proxy for socio-economic status (SES) 1st = Upper 2nd = Middle 3rd = Lower
# * Name: Name of the passenger
# * Sex: Sex of the passenger
# * Age: Age in years of the passenger
# * SibSp: The dataset defines family relations in this way...  Sibling = brother, sister, stepbrother, stepsister.  Spouse = husband, wife (mistresses and fiancés were ignored)
# * Parch: The dataset defines family relations in this way... Parent = mother, father. Child = daughter, son, stepdaughter, stepson. Some children travelled only with a nanny, therefore parch=0 for them.
# * Ticket: Ticket number
# * Fare: Passenger fare
# * Cabin: Cabin number
# * Embarked: Port of embarkation. C = Cherbourg, Q = Queenstown, S = Southampton
# 

# In[ ]:


traindf.shape


# ## **Exploratory Data Analysis (EDA)**

# In[ ]:


traindf['Fare'].describe().reset_index()


# In[ ]:


cols = ['Survived', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']
nr_rows = 2
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        
        i = r*nr_cols+c       
        ax = axs[r][c]
        sns.countplot(traindf[cols[i]], hue=traindf["Survived"], ax=ax)
        ax.set_title(cols[i])
        ax.legend() 
        
plt.tight_layout()   


# In[ ]:


bins = np.arange(0, 80, 5)
g = sns.FacetGrid(traindf, row='Sex', col='Pclass', hue='Survived', margin_titles=True, aspect=1.1)
g.map(sns.distplot, 'Age', kde=False, bins=bins, hist_kws=dict(alpha=0.6))
g.add_legend()  
plt.show()  


# In[ ]:


sns.barplot(x='Sex', y='Survived', hue='Pclass', data=traindf)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Pclass and Sex")
plt.show()


# ## **Preprocessing**

# In[ ]:


traindf['sexn'] = traindf['Sex'].apply(lambda x: 1 if x == 'female' else 0 )


# In[ ]:


import math
traindf['age_f'] = traindf['Age'].apply(lambda x: x if not math.isnan(x) else 0)
traindf['age_nan'] = traindf['Age'].apply(lambda x: 1 if math.isnan(x) else 0)


# In[ ]:


features = ['age_f', 'sexn', 'age_nan', 'Pclass', 'Fare']


# In[ ]:


traindf[features].head()


# ### Feature selection and Training 

# In[ ]:



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

forest = RandomForestClassifier(n_estimators=100,
                                criterion='gini',
                                max_depth=5,
                                min_samples_split=10,
                                min_samples_leaf=5,
                                random_state=0)
X_train, X_test, y_train, y_test = train_test_split(traindf[features], traindf["Survived"])
X_train.head()
forest.fit(X_train, y_train)
print("Random Forest score: {0:.2}".format(forest.score(X_test, y_test)))


# In[ ]:


plt.bar(np.arange(len(features)), forest.feature_importances_)
plt.xticks(np.arange(len(features)), features, rotation='vertical', ha='left')
plt.tight_layout()


# In[ ]:


X_test

# This is an example! Also a bad practise :D
#AGE, SEX, AGE_NAN, PClass, FARE
testcase = np.array([[25, 0, 0, 1, 2]])
prediction = forest.predict(testcase)[0]
pproba = forest.predict_proba(testcase)[0]
print("Prediction for test case: %s (perish -> %.2f, surv -> %.2f)" %
      ('PERISH' if prediction == 0 else 'SURVIVED!', pproba[0], pproba[1]))


# In[ ]:




