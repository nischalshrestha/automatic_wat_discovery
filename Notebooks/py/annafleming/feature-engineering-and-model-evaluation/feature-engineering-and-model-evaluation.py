#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().magic(u'matplotlib inline')
from datetime import datetime
sns.set()
import warnings
warnings.filterwarnings('ignore')


# In[3]:


original_train_df = pd.read_csv('../input/train.csv')
original_test_df = pd.read_csv('../input/test.csv')
original_train_df.shape, original_test_df.shape


# I'm going to combine `Train` and `Test` datasets into one in order to efficently perform data clean up.

# In[4]:


combined_df = original_train_df.append(original_test_df)
combined_df.head()


# ## Meet the data

# In[5]:


original_train_df['Survived'].value_counts()


# It appears that both of the classes are well represented in the training dataset with the majority of `Survived` = 0

# We are going to delete the features `PassengerId` and `Ticket`, since thay shouldn't have any predictive power. 

# In[7]:


combined_df = combined_df.drop(['PassengerId', 'Ticket'], axis=1)


# Let's now look at each feature.

# #### Age

# In[9]:


sns.kdeplot(combined_df[combined_df.Survived == 1]['Age'].dropna(), label='Survived', shade=True)
sns.kdeplot(combined_df[combined_df.Survived == 0]['Age'].dropna(), label='Died', shade=True)


# Looks like younger passengers had a better survival rate, then any other age group, while the adults between approximately 20 and 30 were less likely to survive.  

# #### Cabin
# 
# Since `Cabin` value is going to be somewhat unique, we are going to create a new feature `Deck` using the first letter from `Cabin`. At first, we are going to fill the missing data with the value `U`.  We are also going to remove the feature `Cabin` from the dataset.

# In[10]:


combined_df['Cabin'] = combined_df['Cabin'].fillna('U').astype(str)
combined_df['Deck'] = combined_df['Cabin'].apply(lambda x: x[0])
combined_df = combined_df.drop('Cabin', axis=1)


# In[11]:


sns.countplot(y="Deck", hue="Survived", data=combined_df)


# Most of the examples don't  have a record of the `Cabin` (`Deck`), so having the feature might not help with the prediction. We're still going to keep it for now.

# #### Embarked	

# In[12]:


combined_df["Embarked"].value_counts()


# In[13]:


sns.countplot(y="Embarked", hue="Survived", data=combined_df)


# It looks like most of the passengers embarked in `S` and there is no clear correlation with the output label.

# #### Fare

# In[14]:


sns.kdeplot(combined_df[combined_df.Survived == 1]['Fare'].dropna(), label='Survived', shade=True)
sns.kdeplot(combined_df[combined_df.Survived == 0]['Fare'].dropna(), label='Died', shade=True)


# People with more expensive tickets appear to have a better survival rate. Also we need to note that the distribution is skewed to the right and most of the ticket vary between 0 and 100. 

# #### Name
# 
# Since `Name` is unique in every record and can't help us with the predictions, we are going to extract the name prefix.

# In[15]:


def extract_prefix(name):
    return name.split(',')[1].split('.')[0].strip()

combined_df['Prefix'] = combined_df['Name'].apply(extract_prefix)

combined_df = combined_df.drop('Name', axis=1)

combined_df['Prefix'].value_counts()


# Since we have a lot of low count prefixes, we are going to combine them into similar groups.

# In[16]:


prefix_mapping = {'Ms': 'Miss', 
                  'Mlle': 'Miss', 
                  'Mme': 'Mrs', 
                  'Col': 'Sir',
                  'Major': 'Sir', 
                  'Dona' : 'Lady', 
                  'the Countess': 'Lady',
                  'Capt': 'Sir',  
                  'Don': 'Sir',  
                  'Jonkheer': 'Sir'}
combined_df['Prefix'] = combined_df['Prefix'].replace(prefix_mapping)
combined_df['Prefix'].value_counts()


# In[17]:


sns.countplot(y="Prefix", hue="Survived", data=combined_df)


# From the plot, it looks like children (`Master`) and females had greater survival rate.

# #### Parch, SibSp

# In[18]:


combined_df['Parch'].value_counts()


# In[19]:


combined_df['SibSp'].value_counts()


# In[20]:


(combined_df['SibSp'] + combined_df['Parch']).value_counts()


# It looks like the majority of the people traveled alone. We are going to keep the features unchanged.

# #### Pclass

# In[21]:


sns.countplot(y="Pclass", hue="Survived", data=combined_df)


# It appears that 1st class had better chances to survive, for the 2nd class it's roughly 50/50 and 3rd class passengers had quite low chance to survive.

# #### Sex

# In[22]:


sns.countplot(y="Sex", hue="Survived", data=combined_df)


# ## Missing values

# In[23]:


combined_df.isnull().sum()


# ##### Age
# 
# We are going to use median age for `Prefix` to impute missing values.

# In[24]:


median_ages = combined_df[['Age', 'Prefix']].groupby('Prefix').median()
median_ages


# In[25]:


for title in median_ages.index:
    title_mask = (combined_df['Prefix'] == title) & (combined_df['Age'].isnull()) 
    median_value = float(median_ages.loc[title])
    combined_df.loc[title_mask,'Age'] = median_value


# ##### Embarked
# 
# We are going to use the most common value `S` to fill missing values.

# In[26]:


combined_df["Embarked"] = combined_df["Embarked"].fillna('S')


# ##### Fare

# In[27]:


combined_df[combined_df['Fare'].isnull()]


# In[28]:


median_fare = combined_df[(combined_df['Embarked'] == 'S') & (combined_df['Pclass'] == 3)]['Fare'].median()
print(median_fare)
combined_df["Fare"] = combined_df["Fare"].fillna(median_fare)


# ## Binning
# 
# Binning is one way to make linear models more powerful on continuous data. We are going to apply binning to `Age` and `Fare`, and delete the original features.

# In[29]:


# Age
bins_count = 7
combined_df['Age_Bins'] = pd.qcut(combined_df['Age'], bins_count, labels=list(range(1,bins_count + 1)))

# Fare
bins_count = 5
combined_df['Fare_Bins'] = pd.qcut(combined_df['Fare'], bins_count, labels=list(range(1,bins_count + 1)))

combined_df = combined_df.drop(['Age', 'Fare'], axis=1)


# ## Create dummy features
# 
# Since some of the features are encoded with numbers, we are going to need to specifically list all the categorical columns. Setting `drop_first` paremeter to `True` will help us to get rid of collinearity.

# In[30]:


categorical_columns = ['Embarked', 'Pclass', 'Sex', 'Deck', 'Prefix', 'Age_Bins', 'Fare_Bins']
combined_df = pd.get_dummies(combined_df, columns=categorical_columns, drop_first=True)


# In[31]:


combined_df.columns


# ### Splitting `combined_df` back on training and testing datasets

# In[35]:


train_df = combined_df[~combined_df['Survived'].isnull()]
test_df = combined_df[combined_df['Survived'].isnull()]
train_df['Survived'] = train_df['Survived'].astype(int)
test_df = test_df.drop('Survived', axis=1)


# ## Model testing

# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE


# First I'm going to write two small functions that are going to help us with model comparison.

# In[37]:


def test_models(df, label, models, scoring):
    num_folds = 5
    seed = 7
    X = df.drop(label, axis=1).values
    y = df[label].values
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=.30, random_state=seed)
    results = {}
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        results[name] = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        
    for name in results:
        print("%s: %s" % (name, results[name].mean()))
        
    return results


# In[38]:


def show_model_comparison_plot(results):
    fig = plt.figure() 
    fig.suptitle('Algorithm Comparison') 
    ax = fig.add_subplot(111) 
    plt.boxplot(results.values()) 
    ax.set_xticklabels(results.keys())
    ax.set_ylim(0,1)
    plt.show()


# In[39]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC())) 
results = test_models(train_df, 'Survived', models , 'accuracy')
show_model_comparison_plot(results)


# Looks like 'K-Nearest Neighbors' and 'SVM' came back with the best results. Let's perform some parameter tuning to further improve accuracy.

# ### Parameter tuning

# In[40]:


def param_tuning(df,label, model, scoring, parameters):
    num_folds = 5
    seed = 7
    X = df.drop(label, axis=1).values
    y = df[label].values
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=.30, random_state=seed)
    grid = GridSearchCV(model, parameters, cv=5)
    grid.fit(X_train, Y_train)
    return grid.best_params_, grid.best_score_


# #### SVM

# In[41]:


params_svc = {'svc__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
              'svc__C': [0.001, 0.01, 0.1, 1, 10, 100]}
model = make_pipeline(SVC())
param_tuning(train_df, 'Survived', model , 'accuracy', params_svc)


# #### K-Nearest Neighbors

# In[42]:


params_knn = {'kneighborsclassifier__n_neighbors': [3, 4, 5, 6, 7]}
model = make_pipeline(KNeighborsClassifier())
param_tuning(train_df, 'Survived', model, 'accuracy', params_knn)


# After the parameter tuning our winner is `SVC` with C=1 and gamma=0.1. Let's see if Ensemble models are going to be able to beat it.

# ### Ensemble models

# In[43]:


ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))
ensembles.append(('XGB', XGBClassifier()))
results = test_models(train_df, 'Survived', ensembles , 'accuracy')
show_model_comparison_plot(results)


# Let's see parameter tuning is going help `GradientBoostingClassifier` come up with a better score.

# In[44]:


gb_grid_params = {'learning_rate': [0.1, 0.05, 0.02],
              'max_depth': [3, 4, 5],
              'min_samples_split': [2, 5,10] }

model = GradientBoostingClassifier()
param_tuning(train_df, 'Survived', model, 'accuracy', gb_grid_params)


# Parameter tuning did make a slight improvement, but `SVC` still holds the best accuracy score.

# We used all the features that we engineered for model training. Let's see now if performing feature selection before training is going to lead to better results.

# ### Feature selection

# We are going to use our best performing model (SVC(C=1, gamma=0.1)) for evaluating feature selection techniques. Let's prepare the model, the dataset, and a function to run the comparison efficiently.

# In[45]:


model = SVC(C=1,gamma=0.1)
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

def test_model(df, features, label, model, scoring):
    num_folds = 5
    seed = 7
    X = df[features].values
    y = df[label].values
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=.30, random_state=seed)
    
    kfold = KFold(n_splits=num_folds, random_state=seed)
    result = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    return result


# Let's run the model with all the feature one more time.

# In[46]:


all_columns = train_df.drop('Survived', axis=1).columns
test_model(train_df, all_columns, 'Survived', model , 'accuracy').mean()


# #### Univariate feature selection

# In[47]:


select = SelectPercentile(percentile=90) 
select.fit(X, y)
mask = select.get_support()
univ_features = np.array(all_columns)[mask]
print('Univariate feature selection')
print('------')
print('Excluded features')
print(np.array(all_columns)[~mask])
print('Accuracy %s' % test_model(train_df, univ_features, 'Survived', model , 'accuracy').mean())


# #### Model-Based Feature Selection 

# In[52]:


select = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=7),
        threshold="mean")
select.fit(X, y)
mask = select.get_support()
mb_features = np.array(all_columns)[mask]
print('Model-Based Feature Selection')
print('------')
print('Excluded features')
print(np.array(all_columns)[~mask])
print('Accuracy %s' % test_model(train_df, mb_features, 'Survived', model , 'accuracy').mean())


# #### Iterative Feature Selection 

# In[54]:


print("Total features %s" % len(all_columns))


# In[55]:


select = RFE(RandomForestClassifier(n_estimators=100, random_state=7),
                 n_features_to_select=18)
select.fit(X, y)
mask = select.get_support()
rfe_features = np.array(all_columns)[mask]
print('Iterative Feature Selection')
print('------')
print('Excluded features')
print(np.array(all_columns)[~mask])
print('Accuracy %s' % test_model(train_df, rfe_features, 'Survived', model , 'accuracy').mean())


# So none of the Feature selection algorithms helped us to improve our accuracy score, which means that all the features that we engineered previously contribute to predicting the correct result.
