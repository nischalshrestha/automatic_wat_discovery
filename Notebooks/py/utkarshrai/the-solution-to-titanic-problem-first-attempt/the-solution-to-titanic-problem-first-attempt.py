#!/usr/bin/env python
# coding: utf-8

# This was my first solution to the Titanic Problem that scored 0.8 on Kaggle. This is not my final submission; in my final submission I've broken the dataframes by titles and applied random forest to each set independently thereby maintaining the ratio of survived adult males and women that isn't correct in this code. However this gives you a fair idea of who the machine thinks should've been saved on the boat.

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns


# In[ ]:


to_test=data.Survived


# In[ ]:


data.drop('Survived',1,inplace=True)
df=data.append(test)
df.reset_index(inplace=True)
df.drop('index',inplace=True,axis=1)


# In[ ]:


df['Title'] = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())


# In[ ]:


grouped = df.groupby(['Sex','Pclass','Title'])


# In[ ]:


grouped.median()


# In[ ]:





# In[ ]:





# In[ ]:


df["Age"] = df.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


df.head()


# In[ ]:


df["Fare"] = df.groupby(['Sex','Pclass','Title'])['Fare'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


df.head()


# In[ ]:


df['Sex'] = df['Sex'].map({'male':1,'female':0})
df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
df['Alone'] = df['FamilySize'].map(lambda s : 1 if s == 1 else 0)
df['Couple'] = df['FamilySize'].map(lambda s : 1 if s==2 else 0)
df['Family'] = df['FamilySize'].map(lambda s : 1 if 3<=s else 0)


# In[ ]:


df.Embarked.fillna('S',inplace=True)
df.Cabin.fillna('U',inplace=True)
df['Cabin'] = df['Cabin'].map(lambda c : c[0])


# In[ ]:


df.drop('Name',axis=1,inplace=True)


# In[ ]:


class_feature = pd.get_dummies(df['Pclass'],prefix="Pclass")
titles_feature = pd.get_dummies(df['Title'],prefix='Title')
embarked_feature = pd.get_dummies(df['Embarked'],prefix='Embarked')
cabin_feature = pd.get_dummies(df['Cabin'],prefix='Cabin')
df = pd.concat([df,cabin_feature],axis=1)
df = pd.concat([df,class_feature],axis=1)
df = pd.concat([df,titles_feature],axis=1)
df = pd.concat([df,embarked_feature],axis=1)
df.drop('Ticket',inplace=True,axis=1)
df.drop('Pclass',inplace=True,axis=1)
df.drop('Title',inplace=True,axis=1)
df.drop('Cabin',inplace=True,axis=1)


# In[ ]:


df.drop('Embarked',inplace=True,axis=1)


# In[ ]:


df.head()


# In[ ]:





# In[ ]:


data = df.ix[0:890]
test = df.ix[891:1308]


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(data, to_test)


# In[ ]:


features = pd.DataFrame()
features['feature'] = data.columns
features['importance'] = clf.feature_importances_


# In[ ]:


features.sort(['importance'],ascending=False)


# In[ ]:


model = SelectFromModel(clf, prefit=True)
training = model.transform(data)
training.shape


# In[ ]:


test


# In[ ]:


test = test.fillna(method='ffill')


# In[ ]:


testing = model.transform(test)
testing.shape


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
forest = RandomForestClassifier(max_features='sqrt')

parameter_grid = {'max_depth':[5],'n_estimators': [220],'criterion': ['gini','entropy']}

cross_validation = StratifiedKFold(to_test, n_folds=5)

grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(training, to_test)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


# In[ ]:


pipeline = grid_search
output = pipeline.predict(testing).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output1.csv',index=False)


# In[ ]:




