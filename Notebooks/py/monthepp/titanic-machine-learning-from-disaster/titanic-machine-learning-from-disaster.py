#!/usr/bin/env python
# coding: utf-8

# > **Problem overview**
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 

# In[ ]:


# import library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# import model function from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# import model selection from sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# import model evaluation classification metrics from sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


# > **Acquiring training and testing data**
# 
# We start by acquiring the training and testing datasets into Pandas DataFrames.

# In[ ]:


# acquiring training and testing data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


# visualize head of the training data
train_df.head(n=3)


# In[ ]:


# visualize tail of the testing data
test_df.tail(n=3)


# In[ ]:


# combine training and testing dataframe
train_df['DataType'], test_df['DataType'] = 'training', 'testing'
test_df.insert(1, 'Survived', np.nan)
data_df = pd.concat([train_df, test_df])
data_df.head(n=3)


# > **Feature exploration, engineering and cleansing**
# 
# Here we generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution together with exploring some data.

# In[ ]:


# describe training and testing data
data_df.describe(include='all')


# In[ ]:


# feature extraction: surname
data_df['Surname'] = data_df['Name'].str.extract(r'([A-Za-z]+),', expand=False)


# In[ ]:


# feature extraction: title
data_df['Title'] = data_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
data_df['Title'] = data_df['Title'].replace(['Capt', 'Rev'], 'Crew')
data_df['Title'] = data_df['Title'].replace('Ms', 'Miss')
data_df['Title'] = data_df['Title'].replace(['Col', 'Countess', 'Don', 'Dona', 'Jonkheer', 'Lady', 'Major', 'Mlle', 'Mme', 'Sir'], 'Royal')


# In[ ]:


# feature extraction: age
data_df['Age'] = data_df['Age'].fillna(data_df.groupby(by=data_df['Title'])['Age'].transform('mean'))


# In[ ]:


# feature extraction: is woman and is child
data_df['IsWoman'] = data_df['Sex'].apply(lambda x: 1 if x == 'female' else 0)
data_df['IsChild'] = data_df['Title'].apply(lambda x: 1 if x == 'Master' else 0)


# In[ ]:


# feature extraction: family size
data_df['FamilySize'] = data_df['SibSp'] + data_df['Parch'] + 1


# In[ ]:


# feature extraction: is alone
data_df['IsAlone'] = data_df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)


# In[ ]:


# feature extraction: ticket string
data_df['TicketString'] = data_df['Ticket'].str.extract(r'([A-Za-z]+)', expand=False)


# In[ ]:


# feature extraction: has ticket string
data_df['HasTicketString'] = data_df['TicketString'].apply(lambda x: 0 if pd.isnull(x) else 1)


# In[ ]:


# feature extraction: fare per person
data_df['Fare'] = data_df['Fare'].fillna(data_df.groupby(by=data_df['Pclass'])['Fare'].transform('mean'))
data_df['FarePerPerson'] = data_df['Fare'] / data_df['FamilySize']


# In[ ]:


# feature extraction: has fare
data_df['HasFare'] = data_df['Fare'].apply(lambda x: 0 if x == 0 else 1)


# In[ ]:


# feature extraction: cabin string
data_df['Cabin'] = data_df['Cabin'].fillna(0)
data_df['CabinString'] = data_df['Cabin'].str.extract(r'([A-Za-z]+)', expand=False)


# In[ ]:


# feature extraction: has cabin
data_df['HasCabin'] = data_df['CabinString'].apply(lambda x: 0 if pd.isnull(x) else 1)


# In[ ]:


# feature extraction: embarked
data_df['Embarked'] = data_df['Embarked'].fillna(data_df['Embarked'].value_counts().idxmax())


# In[ ]:


# feature extraction: tour
data_df['Tour'] = np.where(data_df['FamilySize'] == 1, '-1', data_df['Ticket'])


# In[ ]:


# feature extraction: woman and child tour
temp_df = data_df.groupby('Tour')['IsWoman', 'IsChild'].transform('sum')
temp_df.loc[data_df['Tour'] == '-1', ['IsWoman', 'IsChild']] = 0
data_df['WomanChildTour'] = (temp_df['IsWoman'] >= 1) & (temp_df['IsChild'] >= 1)


# In[ ]:


# feature extraction: average survived for the same tour
data_df['SurvivedTour'] = data_df.groupby('Tour')['Survived'].transform('mean')
data_df.loc[(data_df['Tour'] == '-1') | (data_df['WomanChildTour'] == False), 'SurvivedTour'] = -1
data_df['SurvivedTour'] = data_df['SurvivedTour'].fillna(-1)


# In[ ]:


# feature extraction: survived
data_df['Survived'] = data_df['Survived'].fillna(0).astype('int')


# In[ ]:


data_df.head(n=3)


# After extracting all features, it is required to convert category features to numerics features, a format suitable to feed into our Machine Learning models.

# In[ ]:


# verify dtypes object
data_df.info()


# In[ ]:


# convert dtypes object to category
col_obj = data_df.select_dtypes(['object']).columns
data_df[col_obj] = data_df[col_obj].astype('category')
data_df.info()


# In[ ]:


# convert dtypes category to category codes
col_cat = data_df.select_dtypes(['category']).columns
data_df[col_cat] = data_df[col_cat].apply(lambda x: x.cat.codes)
data_df.info()


# In[ ]:


data_df.head(n=3)


# > **Analyze and identify patterns by visualizations**
# 
# Let us generate some correlation plots of the features to see how related one feature is to the next. To do so, we will utilize the Seaborn plotting package which allows us to plot very conveniently as follows.
# 
# The Pearson Correlation plot can tell us the correlation between features with one another. If there is no strongly correlated between features, this means that there isn't much redundant or superfluous data in our training data. This plot is also useful to determine which features are correlated to the observed value.

# In[ ]:


# compute pairwise correlation of columns, excluding NA/null values and present through heat map
corr = data_df[data_df['DataType'] == 1].corr()
fig, ax = plt.subplots(figsize=(20, 15))
heatmap = sns.heatmap(corr, annot=True, cmap=plt.cm.RdBu, fmt='.1f', square=True);


# The pairplots is also useful to observe the distribution of the training data from one feature to the other.

# In[ ]:


# plot pairwise relationships in a dataset
pairplot = sns.pairplot(data_df[data_df['DataType'] == 1], diag_kind='kde', diag_kws=dict(shade=True), hue='Survived')


# The pivot table is also another useful method to observe the impact between features.

# In[ ]:


# pivot table: women and children in the same tour
pivottable = pd.pivot_table(data_df[(data_df['DataType'] == 1) & (data_df['IsAlone'] == 0)], aggfunc=np.mean,
                            columns=None, index=['WomanChildTour', 'Tour'], values='Survived').applymap(lambda x: 1 if x in (0, 1) else 0)
pivottable = pivottable.groupby(level='WomanChildTour').mean()
pivottable.style.background_gradient(cmap='Blues')


# In[ ]:


# pivot table
pivottable = pd.pivot_table(data_df[data_df['DataType'] == 1], aggfunc=np.mean,
                            columns=['Sex'], index=['IsAlone'], values='Survived')
pivottable.style.background_gradient(cmap='Blues')


# > **Model, predict and solve the problem**
# 
# Now, it is time to feed the features to Machine Learning models.

# In[ ]:


# select all features to evaluate the feature importances
x = data_df[data_df['DataType'] == 1].drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'DataType', 'Tour', 'SurvivedTour'], axis=1)
y = data_df[data_df['DataType'] == 1]['Survived']


# In[ ]:


x.head(n=3)


# In[ ]:


# set up random forest classifier to find the feature importances
forestclf = RandomForestClassifier(max_depth=99, n_estimators=2000, random_state=0).fit(x, y)
feat = pd.DataFrame(data=forestclf.feature_importances_, index=x.columns, columns=['FeatureImportances']).sort_values(['FeatureImportances'], ascending=False)


# In[ ]:


# plot the feature importances
fig, ax = plt.subplots(figsize=(20, 5))
plt.title('Feature Importances')
plt.bar(feat.index, feat['FeatureImportances'])
plt.axhline(0.02, color="grey")
ax.set_xticklabels(feat.index, rotation='vertical')
ax.set_yscale('log')
plt.tight_layout()
plt.show()


# In[ ]:


# list all features
data_df.columns


# In[ ]:


# list feature importances
feat[feat['FeatureImportances'] > 0.02].index


# In[ ]:


# select the important features
x = data_df[data_df['DataType'] == 1][feat[feat['FeatureImportances'] > 0.02].index]
y = data_df[data_df['DataType'] == 1]['Survived']


# In[ ]:


x.head(n=3)


# In[ ]:


# perform train-test (validate) split
x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=0, test_size=0.25)


# In[ ]:


# model prediction
logreg = LogisticRegression().fit(x_train, y_train)
logreg_ypredict = logreg.predict(x_validate)
logreg_f1score, logreg_auc = f1_score(y_validate, logreg_ypredict), roc_auc_score(y_validate, logreg_ypredict)
logreg_cvscores = cross_val_score(logreg, x, y, cv=5, scoring='accuracy')
print('logistic regression\n  f1 score: %0.4f, auc: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(logreg_f1score, logreg_auc, logreg_cvscores.mean(), 2 * logreg_cvscores.std()))

treeclf = DecisionTreeClassifier(max_depth=20, min_samples_split=5, splitter='best').fit(x_train, y_train)
treeclf_ypredict = treeclf.predict(x_validate)
treeclf_f1score, treeclf_auc = f1_score(y_validate, treeclf_ypredict), roc_auc_score(y_validate, treeclf_ypredict)
treeclf_cvscores = cross_val_score(treeclf, x, y, cv=5, scoring='accuracy')
print('decision tree classifier\n  f1 score: %0.4f, auc: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(treeclf_f1score, treeclf_auc, treeclf_cvscores.mean(), 2 * treeclf_cvscores.std()))

forestclf = RandomForestClassifier(max_depth=20, min_samples_split=5, n_estimators=250, random_state=0).fit(x_train, y_train)
forestclf_ypredict = forestclf.predict(x_validate)
forestclf_f1score, forestclf_auc = f1_score(y_validate, forestclf_ypredict), roc_auc_score(y_validate, forestclf_ypredict)
forestclf_cvscores = cross_val_score(forestclf, x, y, cv=5, scoring='accuracy')
print('random forest classifier\n  f1 score: %0.4f, auc: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(forestclf_f1score, forestclf_auc, forestclf_cvscores.mean(), 2 * forestclf_cvscores.std()))


# > **Supply or submit the results**
# 
# Our submission to the competition site Kaggle is ready. Any suggestions to improve our score are welcome.

# In[ ]:


# model selection
model = forestclf

# prepare testing data and compute the observed value
x_test = data_df[data_df['DataType'] == 0][feat[feat['FeatureImportances'] > 0.02].index]
y_test = pd.DataFrame(model.predict(x_test), columns=['Survived'])


# In[ ]:


# overwrite same tour as training set
temp_df = data_df.loc[(data_df['DataType'] == 0) & (data_df['SurvivedTour'] > -1)]['SurvivedTour']
y_test.loc[temp_df.index, 'Survived'] = temp_df.apply(lambda x: 0 if x < 0.5 else 1).astype('int')


# In[ ]:


# summit the results
out = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_test['Survived']})
out.to_csv('submission.csv', index=False)


# In[ ]:




