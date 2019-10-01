#!/usr/bin/env python
# coding: utf-8

# # Titanic for beginners
# it is basic introduction to Kaggle.
# 
# ## Workflow
# 1. Import Necessary Libraries
# 2. Acquire training and testing data.
# 3. Analyze, Visualize data
#   1. Outlets (errors or possibly innacurate values) ?
#   2. Create new feature?
# 4. Clearning data
# 5. Choosing the Best Model
# 6. Creating Submission File

# ## 1. Import Necessary Libraries

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
import seaborn as sns
import sklearn
from sklearn import ensemble, linear_model, naive_bayes, neighbors, svm, tree
import subprocess

get_ipython().magic(u'matplotlib inline')


# ## 2. Acquire training and testing data

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]


# ## 3. Analyze, Visualize data
# ### Take a look on data

# In[ ]:


train_df.head()


# In[ ]:


train_df.tail()


# ### Analyze features
# - **Which features are categorical?** - Survived, Sex, and Embarked. Ordinal: Pclass.
# - **Which features are numerical?** - Age, Fare. Discrete: SibSp, Parch.
# - **Which features are mixed data types?** - Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.
# - **Which features may contain errors or typos?** - Name feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets, and quotes used for alternative or short names.
# - **Which features contain blank, null or empty values?** - Cabin > Age > Embarked features contain a number of null values in that order for the training dataset. Cabin > Age are incomplete in case of test dataset.
# - **What are the data types for various features?** - Seven features are integer or floats. Six in case of test dataset. Five features are strings (object).
# - **What is the distribution of numerical feature values across the samples?**
#   - Total samples are 891 or 40\% of the actual number of passengers on board the Titanic (2,224).
#   - Survived is a categorical feature with 0 or 1 values.
#   - Around 38\% samples survived representative of the actual survival rate at 32%.
#   - Most passengers `(> 75\%)` did not travel with parents or children.
#   - Nearly 30\% of the passengers had siblings and/or spouse aboard.
#   - Fares varied significantly with few passengers `(<1\%)` paying as high as $512.
#   - Few elderly passengers (<1\%) within age range 65-80.
#   - TODO: *distribution of age and distribution survived index by age*
#   - TODO: *what is survived index of single persons (without children, parents, siblings or spouse)?*
#   - TODO: *what is distribution of age of single persons?*
#   - TODO: *what is survived index of not single persons which has more stronger relative (wife and husband, child vs parent and etc)*
# - **What is the distribution of categorical features?**
#   - Names are unique across the dataset (count=unique=891)
#   - Sex variable as two possible values with `65\%` male (top=male, freq=577/count=891).
#   - Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
#   - Embarked takes three possible values. S port used by most passengers (top=S)
#   - Ticket feature has high ratio (22\%) of duplicate values (unique=681).
#   

# In[ ]:


print('# features:')
print(train_df.columns.values)
print('_'*40)
print('# data types:')
train_df.info()
print('_'*40)
test_df.info()


# In[ ]:


# numberical features
train_df.describe()


# In[ ]:


# categorical features
train_df.describe(include=['O'])


# ### Assumtions based on data analysis
# - **Correlating** - check correlaction of each feature with survive index
# - **Completing** - try to complete significant feature (**Age**, **Embarked**)
# - **Correcting**
#   - **Ticket** feature may be dropped from our analysis as it contains high ratio of duplicates (22%) and there may not be a correlation between Ticket and survival.
#   - **Cabin** feature may be dropped as it is highly incomplete or contains many null values both in training and test dataset.
#   - **PassengerId** may be dropped from training dataset as it does not contribute to survival.
#   - **Name** feature is relatively non-standard, may not contribute directly to survival, so maybe dropped.
# - **Creating**
#   - We may want to create a new feature called **Family** based on Parch and SibSp to get total count of family members on board.
#   - We may want to engineer the **Name** feature to extract Title as a new feature. *ME: does it really influent on survive index?*
#   - We may want to create new feature for **Age bands**. This turns a continous numerical feature into an ordinal categorical feature.
#   - We may also want to create a **Fare range** feature if it helps our analysis.
# - **Classifying**
#   - Women (**Sex**=female) were more likely to have survived
#   - Children (**Age**<?) were more likely to have survived
#   - The upper-class passengers (**Pclass**=1) were more likely to have survived

# ### Analyze by pivoting features
# - **Pclass** We observe significant correlation (>0.5) among Pclass=1 and Survived (classifying #3). We decide to include this feature in our model.
# - **Sex** We confirm the observation during problem definition that Sex=female had very high survival rate at 74% (classifying #1).
# - **SibSp** and **Parch** These features have zero correlation for certain values. It may be best to derive a feature or a set of features from these individual features (creating #1).

# In[ ]:


def chance_to_survive_by_feature(feature_name):
    return train_df[[feature_name, 'Survived']]        .groupby([feature_name])        .mean()        .sort_values(by='Survived', ascending=False)    

chance_to_survive_by_feature('Pclass')


# In[ ]:


chance_to_survive_by_feature('Sex')


# In[ ]:


chance_to_survive_by_feature('SibSp')


# In[ ]:


chance_to_survive_by_feature('Parch')


# ## Visualization

# - Infants (Age <=4) had high survival rate.
# - Oldest passengers (Age = 80) survived.
# - Large number of 15-25 year olds did not survive.
# - Most passengers are in 15-35 age range.

# In[ ]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20);


# - Pclass=3 had most passengers, however most did not survive. Confirms our classifying assumption #2.
# - Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies our classifying assumption #2.
# - Most passengers in Pclass=1 survived. Confirms our classifying assumption #3.
# - Pclass varies in terms of Age distribution of passengers.

# In[ ]:


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[ ]:


ordered_embarked = train_df.Embarked.value_counts().index

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend();


# In[ ]:


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend();


# - Higher fare paying passengers had better survival. Confirms our assumption for creating (#4) fare ranges.
# - Port of embarkation correlates with survival rates. Confirms correlating (#1) and completing (#2).

# # Clearning data
# ## Drop features

# In[ ]:


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
print("After ", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


# ## Create new feature
# ### Create 'Title' and drop 'Name'

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])


# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir',                                                  'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()


# In[ ]:


train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


# ## Convert Sex type to number

# In[ ]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# ## Completing a numerical continuous feature
# Methods
# - A simple way is to generate random numbers between mean and standard deviation.
# - More accurate way of guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender, and Pclass. Guess Age values using median values for Age across sets of Pclass and Gender feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...
# - Combine methods 1 and 2. So instead of guessing age values based on median, use random numbers between mean and standard deviation, based on sets of Pclass and Gender combinations.
# 
# 
# ### Age
# 

# In[ ]:


grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[ ]:


guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[ ]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()


# In[ ]:


train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()


# ## Create feature "FamilySize"

# In[ ]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[[
    'FamilySize', 
    'Survived',
]].groupby([
    'FamilySize'
], as_index=False)\
.mean()\
.sort_values(by='Survived', ascending=False)


# ## Create feature "IsAlone"

# In[ ]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[[
    'IsAlone', 
    'Survived',
]]\
.groupby(['IsAlone'], as_index=False)\
.mean()


# In[ ]:


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()


# ## Create artificial feature combining "Pclass and Age."

# In[ ]:


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# ## Complete missed values of feature "Embarked"

# In[ ]:


freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[[
    'Embarked', 
    'Survived',
]]\
.groupby(['Embarked'], as_index=False)\
.mean()\
.sort_values(by='Survived', ascending=False)


# ## Convert feature "Embarked" to numeric

# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# ## Complete one missed value for feature "Fare"

# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


# In[ ]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[[
    'FareBand', 
    'Survived',
]]\
.groupby(['FareBand'], as_index=False)\
.mean()\
.sort_values(by='FareBand', ascending=True)


# In[ ]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


# # Model, predict and solve
# Binary classification

# In[ ]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# ## Logistic regression
#  logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier

# In[ ]:


models = []

models.append({
    'classifier': linear_model.LogisticRegression,
    'name': 'Logistic Regression',
})
models.append({
    'classifier': svm.SVC,
    'name': 'Support Vector Machines',
})
models.append({
    'classifier': neighbors.KNeighborsClassifier,
    'name': 'k-Nearest Neighbors',
    'args': {
        'n_neighbors': 3,
    },
})
models.append({
    'classifier': naive_bayes.GaussianNB,
    'name': 'Gaussian Naive Bayes',
})
models.append({
    'classifier': linear_model.Perceptron,
    'name': 'Perceptron',
    'args': {
        'max_iter': 5,
        'tol': None,
    },
})
models.append({
    'classifier': svm.LinearSVC,
    'name': 'Linear SVC',
})
models.append({
    'classifier': linear_model.SGDClassifier,
    'name': 'Stochastic Gradient Descent',
    'args': {
        'max_iter': 5,
        'tol': None,
    },
})
models.append({
    'classifier': tree.DecisionTreeClassifier,
    'name': 'Decision Tree',
})
models.append({
    'classifier': ensemble.RandomForestClassifier,
    'name': 'Random Forest',
    'args': {
        'n_estimators': 100,
    },
})

#acc_log


# ## All Models

# In[ ]:


def process_model(model_desc):
    Model = model_desc['classifier']
    model = Model(**model_desc.get('args', {}))
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = round(model.score(X_train, Y_train) * 100, 2)
    return {
        'name': model_desc['name'],
        'accuracy': accuracy,
        'model': model,
    }

models_result = list(map(process_model, models))
models_result = sorted(models_result, key=lambda res: res['accuracy'], reverse=True)

#print(models_result)

# plot bars
models_result_df = pd.DataFrame(models_result, columns=['accuracy', 'name'])
ax = sns.barplot(data=models_result_df, x='accuracy', y='name')
ax.set(xlim=(0, 100))

# show table
models_result_df


# In[ ]:


# use keras (tensorflow) for full-convolutional deep NN


# # Submit result
# get the best model and submit the result

# In[ ]:


# submission.to_csv('../output/submission.csv', index=False)
the_best_result = models_result[0]
Y_pred = the_best_result['model'].predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': Y_pred,
})
submission.to_csv('submission.csv', index=False)

