#!/usr/bin/env python
# coding: utf-8

# credit for feature engineering: https://www.kaggle.com/pmarcelino/data-analysis-and-feature-extraction-with-python

# In[ ]:


import pandas as pd

from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'
OUTPUT_PATH = '../output/submission.csv'


# In[ ]:


df_raw = pd.read_csv(TRAIN_PATH)
print(df_raw.info() )
df_raw.sample(5)


# In[ ]:


df = df_raw.copy()


# In[ ]:


df.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Name'], inplace=True)


# In[ ]:


# წერტილამდე მეჩავს ყველა ასოს
df['Status'] = df_raw['Name'].str.extract('(\w+)\.', expand=False)
print(df['Status'].value_counts() )
df['Status'] = df['Status'].str.replace(r'Dr|Rev|Col|Mlle|Major|Lady|Sir|Don|Capt|Mme|Jonkheer|Countess|Ms', 'Rare')
status_age_dict = df.groupby('Status')['Age'].mean().to_dict()
status_age_dict


# In[ ]:


no_age_index = df[df['Age'].isnull()].index
df.loc[no_age_index, 'Age'] = df['Status'].loc[no_age_index].map(status_age_dict)


# In[ ]:


print(df.info())
'მართლაც შეივსო Age ველი'


# In[ ]:


df.drop(index=df.loc[df['Embarked'].isnull()].index, inplace=True)


# In[ ]:


def categorize_columns(df, colnames):
    for colname in colnames:
        df[colname] = df[colname].astype('category')
    return

categorize_columns(df, ['Pclass', 'Sex', 'Embarked', 'Status'])



# In[ ]:


df['Relatives'] = df['SibSp'] + df['Parch']
df.drop(columns=['SibSp', 'Parch'], inplace=True)


# In[ ]:


sns.heatmap(df.corr() )


# In[ ]:


sns.barplot(df['Pclass'], df['Survived'])
plt.show()
sns.barplot(df['Sex'], df['Survived'])
plt.show()
sns.barplot(df['Embarked'], df['Survived'])
plt.show()
sns.barplot(df['Status'], df['Survived'])
plt.show()


# # do it

# In[ ]:


df_train_X, df_test_X, df_train_y, df_test_y = train_test_split(df.loc[:, df.columns != 'Survived'], df['Survived'], test_size=300)


# In[ ]:


stdscaler = StandardScaler()
df_train_X.loc[:, df_train_X.dtypes != 'category'] = stdscaler.fit_transform(df_train_X.loc[:, df_train_X.dtypes != 'category'])
df_test_X.loc[:, df_train_X.dtypes != 'category'] = stdscaler.transform(df_test_X.loc[:, df_train_X.dtypes != 'category'])
df_train_X = pd.get_dummies(df_train_X, drop_first=True)
df_test_X = pd.get_dummies(df_test_X, drop_first=True)

# make sure I did not fuck up train/test sameness:
df_train_X.columns.tolist() == df_test_X.columns.tolist()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

single_tree = DecisionTreeClassifier(
#     max_depth=50, 
#     min_impurity_decrease=0.3, 
#     min_samples_leaf=100
)
single_tree_grid_search = GridSearchCV(cv=10, estimator=single_tree, n_jobs=4, scoring='accuracy', param_grid={
    'max_depth': [2, 4, 8, 16, 32, 64],
    'min_samples_split': [0.1, 0.3, 0.6, 0.9],
    'min_impurity_decrease': [0.0, 0.001, 0.01, 0.1, 1.0, 10.0],
})
single_tree_grid_res = single_tree_grid_search.fit(df_train_X.values, df_train_y.values)


# In[ ]:


single_tree_grid_res.best_params_


# In[ ]:


single_tree_grid_res.best_score_


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    **single_tree_grid_res.best_params_,
    max_features='sqrt' # sqrt > log2
)
rf_cv_score = cross_val_score(rf, df_train_X.values, df_train_y.values, cv=10, n_jobs=4)
print('mean ' + str(rf_cv_score.mean() ) + ', +-' + str(rf_cv_score.std() ) )


# In[ ]:


rf_grid_search = GridSearchCV(cv=10, estimator=rf, n_jobs=4, scoring='accuracy', param_grid={
    'n_estimators': [2, 4, 8, 16, 32, 64, 128, 256]
})
rf_grid_res = rf_grid_search.fit(df_train_X.values, df_train_y.values)


# In[ ]:


rf_grid_res.best_params_


# In[ ]:


rf_grid_res.best_score_


# In[ ]:


pd.DataFrame({'FeatureName': df_train_X.columns.tolist(), 'Importance': rf_grid_res.best_estimator_.feature_importances_}).sort_values(by='Importance', ascending=False)


# In[ ]:


rf_grid_search = GridSearchCV(cv=10, estimator=rf, n_jobs=4, scoring='accuracy', param_grid={
    'n_estimators': [2, 4, 8, 16, 32, 64, 128, 256],
    'max_depth': [4, 8, 16],
    'min_impurity_decrease': [0.0, 0.001, 0.01],
    'min_samples_leaf': [1, 2, 4]
})
rf_grid_res = rf_grid_search.fit(df_train_X.values, df_train_y.values)


# In[ ]:


rf_grid_res.best_params_


# In[ ]:


rf_grid_res.best_score_


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


adab = AdaBoostClassifier()
adab_scores = cross_val_score(adab, df_train_X.values, df_train_y.values)
print('mean ' + str(adab_scores.mean() ) + ', +-' + str(adab_scores.std() ) )


# In[ ]:


adab_grid_search = GridSearchCV(cv=10, estimator=adab, n_jobs=4, scoring='accuracy', param_grid={
    'n_estimators': [2, 4, 8, 16, 32, 64, 128, 256]
})
adab_grid_res = adab_grid_search.fit(df_train_X.values, df_train_y.values)


# In[ ]:


adab_grid_res.best_params_


# In[ ]:


adab_grid_res.best_score_


# ### single tree is the best

# In[ ]:


best_estimator = single_tree_grid_res.best_estimator_


# In[ ]:


best_estimator.score(df_test_X.values, df_test_y.values)


# ### output

# In[ ]:


df_submit_raw = pd.read_csv(TEST_PATH)
print(df_submit_raw.info() )
df_submit_raw.sample(5)


# In[ ]:


df_submit = df_submit_raw.copy()
df_submit.drop(columns=['Ticket', 'Cabin', 'Name'], inplace=True)
df_submit['Fare'].fillna(df['Fare'].mean(), inplace=True)
no_age_index = df_submit[df_submit['Age'].isnull()].index
df_submit['Status'] = df_submit_raw['Name'].str.extract('(\w+)\.', expand=False)
df_submit['Status'] = df_submit['Status'].str.replace(r'Dr|Rev|Col|Mlle|Major|Lady|Sir|Don|Capt|Mme|Jonkheer|Countess|Ms', 'Rare')
df_submit['Status'] = df_submit['Status'].str.replace(r'Rarea', 'Rare')
df_submit.loc[no_age_index, 'Age'] = df_submit['Status'].loc[no_age_index].map(status_age_dict)
# df_submit.drop(index=df_submit.loc[df_submit['Embarked'].isnull()].index, inplace=True)
categorize_columns(df_submit, ['Pclass', 'Sex', 'Embarked', 'Status'])
df_submit['Relatives'] = df_submit['SibSp'] + df_submit['Parch']
df_submit.drop(columns=['SibSp', 'Parch'], inplace=True)

df_submit_data = df_submit.drop(columns=['PassengerId'] )
df_submit_data.info()


# In[ ]:


df_submit_data.loc[:, df_submit_data.dtypes != 'category'] = stdscaler.transform(df_submit_data.loc[:, df_submit_data.dtypes != 'category'])
df_submit_data = pd.get_dummies(df_submit_data, drop_first=True)

# make sure I did not fuck up train/test sameness:
df_submit_data.columns.tolist() == df_test_X.columns.tolist()


# In[ ]:


predictions = best_estimator.predict(df_submit_data)


# In[ ]:


submission = pd.DataFrame({'PassengerId': df_submit['PassengerId'], 'Survived': predictions})
submission.head(5)


# In[ ]:


submission.to_csv('submission.csv')


# In[ ]:




