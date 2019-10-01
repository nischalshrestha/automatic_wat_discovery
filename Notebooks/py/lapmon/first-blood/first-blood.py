#!/usr/bin/env python
# coding: utf-8

# > This is the first kaggle made early.Thanks to those kind people who share thoughts on the forum.Help me a lot.
# 
# > Welcome any good suggestion both on improving behaviour or coding style 
# 

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)


# # data process & feature engineering
# 
# ## check data
# 
# 'Age' 'Embarked' has nan
# 
# - Pclass：几等仓
# - cabin: 客舱
# - SibSp：兄弟姐妹/配偶
# - Parch：Parents/Children
# - Embarked：上船港口

# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head(1)


# In[ ]:


df.describe()


# In[ ]:


df.count()


# In[ ]:


df_pred = pd.read_csv('../input/test.csv')
df_pred.count()


# ### feature engineering
# 
# - 'Cabin' as sparse as 20+% ,so drop it;
# - 'Age', 'Fare' need fill_na strategy
# - 'Embarked' has very few nan, so choose easy ways:make a na-feature or just drop

# In[ ]:


df_data = df.copy()
print(float(df['Cabin'].count()) / df.shape[0])
print((float)(df_pred['Cabin'].count()) / df_pred.shape[0])


# #### pclass sex

# In[ ]:


df_data = pd.concat([df_data, pd.get_dummies(df_data['Pclass'], prefix='Pclass')], axis=1)
#data = pd.concat([data, pd.get_dummies(data['Sex'], prefix='Sex')], axis=1)
df_data['Gender'] = df_data['Sex'].map({'female': 0, 'male': 1}).astype(int)


# ### family
# 
# survived factor : family size
# 
# expired by Megan Risdal's submission.split to three category variables

# In[ ]:


#sns.countplot(x="Pclass", data=train, palette="Greens_d")
df_data['Family'] = df_data['SibSp'] + df_data['Parch'] + 1
sns.countplot(x="Family", data=df_data, hue='Survived')


# In[ ]:


df_data['Singleton'] = (df_data['Family'] == 1).astype(int)
df_data['FamilySmall'] = np.logical_and(df_data['Family'] > 1, df_data['Family'] < 5).astype(int)
df_data['FamilyLarge'] = (df_data['Family'] >= 5).astype(int)
df_data['FamilySize'] = df_data['Singleton'] + df_data['FamilySmall']*2 + df_data['FamilyLarge']*3
sns.countplot(x="FamilySize", data=df_data, hue='Survived')


# ### Age 
# 
# #### new feature: Child
# 
# notice the inflection point of the curve

# In[ ]:


sns.distplot(df_data["Age"].dropna(), rug=False, kde=True, hist=True)


# In[ ]:


facet = sns.FacetGrid(df_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, df_data['Age'].max()))
facet.add_legend()


# child > 10 and < 10 seems distribute differently. Add new category feature 'Child'

# In[ ]:


df_data['Child'] = (df_data['Age'] < 10).astype(int)


# #### fill na
# 
# person with large number of fanily maybe likely to be younger, not so clearly, think about it later

# In[ ]:


sns.boxplot(x="Family", y="Age", data=df_data)


# In[ ]:


#age_mean = df_data['Age'].mean()
#df_data['Age'] = df_data['Age'].fillna(df_data['Age'].mean())
print((float)(df_data['Age'].count()) / df_data.shape[0])


# 'Age' missing 20% and due to feature importance(previous experiment result) ,it has an unignore influence on result.So try to process it more precisely: regression.
# 
# process lit later when other feature dealed

# ### fare
# 
# simplely use median or mean maybe not a good idea.
# 
# 'Pclass' 'Sex' "Embarked" related to 'Fare' seems rational 

# In[ ]:


sns.distplot(df_data["Fare"].dropna(), rug = False,kde=True,hist=False)


# In[ ]:


sns.pointplot(x="Pclass", y="Fare", hue='Sex', data=df_data)


# In[ ]:


fare_means = df_data.pivot_table(values='Fare', columns=['Pclass','Sex','Embarked'], aggfunc='mean')
fare_means


# ### Embarked
# 
# only two miss sample,drop or use na column

# In[ ]:


df_data[df_data['Embarked'].isnull()]


# In[ ]:


#data.drop(data.index[[61,829]], inplace=True, axis=0)
df_data = pd.concat([df_data, pd.get_dummies(df_data['Embarked'], prefix='Embarked', dummy_na=True)], axis=1)


# ### regression predict missing age     

# In[ ]:


df_use_reg = df_data[['PassengerId', 'Survived', 'Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1',                    'Pclass_2', 'Pclass_3', 'Gender', 'Singleton', 'FamilySmall', 'FamilyLarge', 'Child',                    'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Embarked_nan']]


# In[ ]:


from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV

X_age=df_use_reg.dropna().as_matrix()[:,3:]
y_age=df_use_reg.dropna().as_matrix()[:,2].astype(float)
X_age_pred=df_use_reg[df_use_reg['Age'].isnull()].as_matrix()[:,3:]
X_age.shape, X_age_pred.shape


# In[ ]:


svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})
svr.fit(X_age, y_age)
df_data['Age']=df_use_reg.apply(lambda x: svr.predict(x[3:].reshape(1, -1)) if pd.isnull(x['Age']) else x['Age'], axis=1)


# ### scale

# In[ ]:


df_X = df_data[['Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1',                    'Pclass_2', 'Pclass_3', 'Gender', 'Singleton', 'FamilySmall', 'FamilyLarge', 'Child',                    'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Embarked_nan']]
# , 'Age_square', 'Fare_square', \
#                   'sex_pclass', 'sex_fare', 'fare_pclass', 'age_pclass', 'fare_age', 'sex_age'
df_X_maxes = df_X.max()
df_use = df_X.apply(lambda x: x/x.max(), axis=0)
df_use = pd.concat([df_data[['PassengerId', 'Survived']], df_use], axis=1)
df_use.head(1)


# # predict

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

X=df_use.as_matrix()[:,2:]
y=df_use.as_matrix()[:,1].astype(int)

est_range = list(range(10, 30, 2)) + list(range(30, 150, 10))
fea_range = np.arange(.5,1,.1).tolist()

parameter_grid = {
    'n_estimators': est_range,
    'max_features': fea_range,
    'max_depth': [5., None]
}
grid_search = GridSearchCV(RandomForestClassifier(n_estimators = 10), parameter_grid,
                            cv=5, verbose=3, scoring='roc_auc')
grid_search.fit(X,y)
model = grid_search.best_estimator_


# ### feature importance

# In[ ]:


feature_importance = model.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 1, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
feature_names = df_use.columns[2:]
plt.yticks(pos, feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[ ]:


df_pred = pd.read_csv('../input/test.csv')
df_pred.head(2)


# In[ ]:


# missing age fare
df_pred.count()


# In[ ]:


df_pred.info()


# In[ ]:


df_pred = pd.concat([df_pred, pd.get_dummies(df_pred['Pclass'], prefix='Pclass')], axis=1)
df_pred['Gender'] = df_pred['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_pred['Fare'] = df_pred[['Fare', 'Pclass', 'Sex', 'Embarked']].apply(lambda x:
                            fare_means[x['Pclass']][x['Sex']][x['Embarked']] if pd.isnull(x['Fare'])
                            else x['Fare'], axis=1)
df_pred['Family'] = df_pred['SibSp'] + df_pred['Parch'] + 1
df_pred['Singleton'] = (df_pred['Family'] == 1).astype(int)
df_pred['FamilySmall'] = np.logical_and(df_pred['Family'] > 1, df_pred['Family'] < 5).astype(int)
df_pred['FamilyLarge'] = (df_pred['Family'] >= 5).astype(int)
df_pred['FamilySize'] = df_pred['Singleton'] + df_pred['FamilySmall']*2 + df_pred['FamilyLarge']*3
df_pred['Child'] = (df_pred['Age'] < 10).astype(int)
df_pred = pd.concat([df_pred, pd.get_dummies(df_pred['Embarked'], prefix='Embarked', dummy_na=True)], axis=1)

df_pred_use_X = df_pred[df_X.columns]

df_pred_use_X['Age']=df_pred_use_X.apply(lambda x: svr.predict(x[1:].reshape(1, -1)) if pd.isnull(x['Age']) else x['Age'], axis=1)

df_pred_use_X = df_pred_use_X / df_X_maxes
df_pred_use_X.head(1)


# In[ ]:


test_data = df_pred_use_X.values
output = model.predict(test_data[:,:])
ids = df_pred['PassengerId'].astype(int)

result = np.c_[ids, output.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('03_10.csv', index=False)


# In[ ]:




