#!/usr/bin/env python
# coding: utf-8

# The aim of this notebook is to go through all the commonest techniques in participating in a Kaggle competition. The dataset used in this notebook is from the Kaggle competition [Titanic: Machine Learning from Disaster]( https://www.kaggle.com/c/titanic).
# 
# I'm still a beginner, so any advice or suggestion is well accepted!

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
get_ipython().magic(u'matplotlib inline')

RANDOM_SEED = 4321
np.random.seed = RANDOM_SEED


# Importing the dataset
# --
# 
# We import the train and test csv file:

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.info()
train.head(10)


# As for the training, we can see that there are some people with an empty age. The Cabin column is at most empty and only 2 entries have a missing Embarked feature.
# 
# Let us consider the test set:

# In[ ]:


test.info()


# Also the test set has similar properties of the training test. We will deal with missing values later.
# 
# Let us explore the dataset a bit:
# 
# ## Survivals
# 
# As we can expect, most of the people abroad did not survive.

# In[ ]:


plt.title('Number of people survived the Titanic', y=1.1, size=15)
sns.countplot('Survived', data=train)


# ## Sex:

# In[ ]:


plt.title('Survival count between sex', size=20, y=1.1)
sns.countplot(x = 'Survived', hue='Sex', data=train)


# So we can see that most of the survived people were women. Let's also analyze the survival rate:

# In[ ]:


plt.title('Survival rate between sex', size=20, y=1.1)
sns.barplot(x='Sex', y='Survived', data=train)


# Since we must deal with numerical feature, we should convert male/female in a binary vector 0/1.

# In[ ]:


for df in [train, test]:
    df['Sex'] = df['Sex'].apply(lambda x : 1 if x == 'male' else 0)


# So females has an overall higher chance to survive.
# 
# ## PClass:
# 
# Now let's analyze the Pclass feature:

# In[ ]:


plt.figure(figsize=(12, 12))
plt.subplot(2,2,1)
plt.title('Survival rate / Pclass', size=15, y=1.1)
sns.barplot(x='Pclass', y = 'Survived', data=train, palette='muted')
plt.subplot(2,2,2)
plt.title('Count survival / Pclass', size=15, y=1.1)
sns.countplot(x='Pclass', hue = 'Survived', data=train, palette='muted')


# ## Embarked
# 
# Now let's dive into the embarked feature

# In[ ]:


train.head(10)


# In[ ]:


sns.countplot(train['Embarked'])
train['Embarked'].describe()


# It seems that most of the people embarked in Southampton. Since it is the most frequent, we will fill the two missing values with 'S'

# In[ ]:


train['Embarked'] = train['Embarked'].fillna('S')
plt.figure(figsize=(12, 12))
plt.subplot(2,2,1)
sns.barplot(y='Survived', x='Embarked', data=train)
plt.subplot(2,2,2)
sns.countplot(x='Survived', hue='Embarked', data=train)


# In[ ]:


sns.boxplot(x='Embarked', y='Fare', data=train)


# In[ ]:


sns.countplot(hue='Pclass', x='Embarked', data=train)


# There is a little correlation between the embarkment harbor and the survival rate. 
# 
# Since we want a numerical feature, we will convert the embarked in one hot vector

# In[ ]:


train = pd.get_dummies(train, columns=['Embarked'])
test = pd.get_dummies(test, columns=['Embarked'])


# ## Fare

# In[ ]:


sns.distplot(train['Fare'])
train['Fare'].describe()


# Fare is really right skewed (few people paying a very high fare and most people paying a low fare). We can see that in the train we have all the values, but one value is missing in test set. Fill it with the median:

# In[ ]:


test['Fare'] = test['Fare'].fillna(test['Fare'].median())


# Since the fare is so oddly distributed, is seems reasonable to introduce categories:

# In[ ]:


for df in [train, test]:
    df['Fare'] = pd.qcut(df['Fare'], 4, labels=[0, 1, 2, 3])

train.head(5)


# ## Cabin
# 
# Cabin is a feature that has very few non null entries. Thus, we can drop this column and only take into account whether the value was present or not.

# In[ ]:


for df in [train, test]:
    df['Cabin'] = df['Cabin'].fillna('NaN')
    df['HasCabin'] = df['Cabin'].apply(lambda x : 0 if x == 'NaN' else 1)

train, test = train.drop(['Cabin'], axis=1), test.drop(['Cabin'], axis=1)


# ## Parch and SibSp
# 
# **Parch** is the abbreviation of 'parent/children' and represent the sum of parents and children.
# 
# **SibSp** is the abbreviation of 'sibling/spouse' and represent the sum of brothers/sisters/wife/husband.
# 
# This said, we can compute the family size as the sum of the two above + 1

# In[ ]:


for df in [train, test]:
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1


# We have also added a feature TravelAlone, this is to say if the passenger has some family member on board or not.

# In[ ]:


sns.barplot(x='FamilySize', y='Survived' , data=train)


# Those with 2-4 family members had a slightly more chance to survive

# In[ ]:


def filter_family_size(x):
    if x == 1:
        return 'Solo'
    elif x < 4:
        return 'Small'
    else:
        return 'Big'

for df in [train, test]:
    df['FamilySize'] = df['FamilySize'].apply(filter_family_size)

train = pd.get_dummies(train, columns=['FamilySize'])
test = pd.get_dummies(test, columns=['FamilySize'])


# ## Name
# 
# Let's see if we can extract something from the name. Here is a list of names from the dataset:

# In[ ]:


train['Name'].head(10)


# One thing that can be useful is the title, which can be found soon after the first comma. Since there are a lot of rare titles, we will join the uncommon ones (such as countess, dona, jonkeer, ...) in 6 labels

# In[ ]:


# Filter the name
def get_title(x):
    y = x[x.find(',')+1:].replace('.', '').replace(',', '').strip().split(' ')
    if y[0] == 'the':    # Search for the countess
        title = y[1]
    else:
        title = y[0]
    return title

def filter_title(title, sex):
    if title in ['Countess', 'Dona', 'Lady', 'Jonkheer', 'Mme', 'Mlle', 'Ms', 'Capt', 'Col', 'Don', 'Sir', 'Major', 'Rev', 'Dr']:
        if sex:
            return 'Rare_male'
        else:
            return 'Rare_female'
    else:
        return title

for df in [train, test]:
    df['NameLength'] = df['Name'].apply(lambda x : len(x))
    df['Title'] = df['Name'].apply(get_title)
    
train.groupby('Title')['PassengerId'].count().sort_values(ascending=False)


# In[ ]:


for df in [train, test]:
    df['Title'] = df.apply(lambda x: filter_title(x['Title'], x['Sex']), axis=1)


# In[ ]:


sns.countplot(y=train['Title'])
train.groupby('Title')['PassengerId'].count().sort_values(ascending=False)


# In[ ]:


plt.figure(figsize=(8, 8))
sns.barplot(x='Sex', y='Survived', hue='Title', data=train)


# We can see that all the ladies with some rare title have all survived.
# 
# As we did before, we must convert the name column in a one hot vector:

# In[ ]:


train = pd.get_dummies(train, columns=['Title'])
test = pd.get_dummies(test, columns=['Title'])

train = train.drop(['Name', 'Sex'], axis=1)
test = test.drop(['Name', 'Sex'], axis=1)

#name_mapping = {'Mr' : 0, 'Mrs' : 1, 'Miss' : 2, 'Master' : 3, 'Rare' : 4}
#for df in [train, test]:
#    df['Name'] = df['Name'].apply(lambda x : name_mapping[x])


# ## Ticket
# 
# Have a look at the ticket column:

# In[ ]:


train['Ticket'].describe()


# In[ ]:


train['Ticket'].head(20)


# There are 681 unique tickets among 891 tickets, so there is some repetition. Let us add a column DuplicatedTicket that take into account that a certain ticket was duplicated.

# In[ ]:


for df in [train, test]:
    df['TicketLetter'] = df['Ticket'].apply(lambda x : str(x)[0])
    #df['TicketLength'] = df['Ticket'].apply(lambda x : len(x))


# In[ ]:


train.groupby(['TicketLetter'])['Survived'].mean().sort_values(ascending=False)


# In[ ]:


sns.barplot(x = 'TicketLetter', y='Survived', data=train)
df_count = train.groupby(['TicketLetter'],as_index=True)['PassengerId'].count().sort_values(ascending=False)
print(df_count)


# In[ ]:


def filter_ticket(x):
    if x in ['9', '8', '5', 'L', '6', 'F', '7', '4', 'W', 'A']:
        return 'Rare'
    elif x in ['C', 'P', 'S', '1']:
        return 'Frequent'
    elif x == '2':
        return 'Common'
    elif x == '3':
        return 'Commonest'

for df in [train, test]:
    df['TicketCategory'] = df['TicketLetter'].apply(filter_ticket)


# In[ ]:


sns.barplot(x='TicketCategory', y='Survived', data=train)


# In[ ]:


def simplify_ticket(df):
    df = pd.get_dummies(df, columns=['TicketCategory'])
    df = df.drop(['TicketLetter', 'Ticket'], axis=1)
    return df

train,test = simplify_ticket(train), simplify_ticket(test)


# In[ ]:


train.columns


# ## Age:
# 
# We have seen that there are some missing values for the age. It is worthy to study its distribution:

# In[ ]:


plt.title('Age distribution', size=20, y=1.1)
sns.distplot(train['Age'].dropna())
train['Age'].describe()


# So the age is not that far from a Gaussian distribution. The median is 28 years, while the average is more like 29-30 years. 
# 
# Let's keep track of whether the age was missing or not with a binary value, and then we will fill the missing values.
# We will fill the missing values generating a random number according to a Gaussian distribution.
# 
# **Note:** we must reflect the changes to the test data as well!

# In[ ]:


for df in [train, test]:
    df['MissingAge'] = df['Age'].apply(lambda x : 1 if np.isnan(x) else 0)


# In[ ]:


col_to_drop = ['PassengerId', 'Age', 'Survived', 'MissingAge']

from sklearn.metrics import mean_squared_error, r2_score

x_train_not_age_nan = train.loc[train['MissingAge'] == 0, :]
y_train_not_age_nan = x_train_not_age_nan['Age']

x_train_not_age_nan = x_train_not_age_nan.drop(col_to_drop, axis=1, errors='ignore')


rpart_params = {
    'criterion' : ['mse'],
    'splitter' : ['best'],
    'max_features' : ['auto', 'sqrt', 'log2', None],
    'max_depth' : [2, 3, 4],
    'min_samples_split' : [2, 3],
    'min_samples_leaf' : [1, 2],
    'max_leaf_nodes' : [3, 4, None],
    'random_state' : [RANDOM_SEED],
    'presort' : [True]
}

model = DecisionTreeRegressor()
age_grid = GridSearchCV(model, rpart_params, n_jobs=-1).fit(x_train_not_age_nan, y_train_not_age_nan.values.ravel())
print('Best model CV score: ', age_grid.best_score_)
age_estimator = age_grid.best_estimator_.fit(x_train_not_age_nan, y_train_not_age_nan.values.ravel())


# In[ ]:


for df in [train, test]:
    age_fill = age_estimator.predict(df.loc[df['MissingAge'] == 1, :].drop(col_to_drop, axis=1, errors='ignore'))

    df.loc[df['MissingAge'] == 1, 'Age'] = age_fill

plt.title('Age distribution after filling NaN', size=20, y=1.1)
sns.distplot(train['Age'])
train['Age'].describe()


# Ok, the mean is slightly changed but the distribution is quite the same.
# 
# To simplify the feature space, we will subdivide the age in 5 slots:

# In[ ]:


for df in [train, test]:
    df['Age'] = pd.cut(df['Age'], 5, labels=[0, 1, 2, 3, 4])


# ## Final refinements
# 
# The only final refinement is to drop the unused columns and convert the Sex column in a binary digit

# In[ ]:


train_y = train['Survived'].ravel()
train_x = train.drop(['Survived', 'PassengerId'], axis=1)

test_x = test.drop(['PassengerId'], axis=1)


for df in [train_x, test_x]:
    for col in df.columns:
        df[col] = df[col].astype('int')


# We have all numerical features and hence we are ready to start with the model selection.

# # Correlation between all features:

# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(12, 12))
plt.title("Feature correlation", y=1.05, size=15)
sns.heatmap(train_x.corr(), linewidths=0.1, square=True, vmax=1.0, annot=True, cmap=colormap)


# In[ ]:


var_correlations = {c: np.abs(train['Survived'].corr(train_x[c])) for c in train_x.columns}

corr_dataframe = pd.DataFrame(var_correlations, index=['Correlation']).T.sort_values(by='Correlation')
plt.title('Correlation between feature and survival rate', y=1.1, size=15)
plt.barh(range(corr_dataframe.shape[0]), corr_dataframe['Correlation'].values, tick_label=train_x.columns.values)


# # Creating the validation set

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_validation, y_train, y_validation = train_test_split(train_x, train_y, test_size=0.3, random_state=RANDOM_SEED)
x_test = test_x.copy()

x_train.index = np.arange(len(x_train))
x_validation.index = np.arange(len(x_validation))


# # Model selection
# 
# ## Logistic regression
# 
# Logistic regression is the simplest classification method. To train the logistic regression, we will use a bunch of parameters and perform a grid search using a crossvalidation with the default 3 folds.
# 
# Original parameters are:
# 
#     lr_params = {
#         'C' : [0.01, 0.03, 0.1, 0.3, 1, 2, 3],
#         'fit_intercept' : [True, False],
#         'max_iter' : [5000],
#         'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
#         'tol' : [1e-4],
#         'random_state' : [RANDOM_SEED]
#     }

# In[ ]:


lr_params = {
    'C' : [1],
    'fit_intercept' : [True],
    'max_iter' : [5000],
    'solver' : ['newton-cg'],
    'tol' : [1e-4],
    'random_state' : [RANDOM_SEED]
}

log_regression = linear_model.LogisticRegression()
acc_scorer = make_scorer(accuracy_score)
log_reg_models = GridSearchCV(log_regression, lr_params, scoring=acc_scorer, n_jobs=-1)
log_reg_models = log_reg_models.fit(x_train, y_train)

lr_best = log_reg_models.best_estimator_
lr_best = lr_best.fit(x_train, y_train)

lr_model = {
    'Name' : 'Logistic regression', 
    'CVScore' : log_reg_models.best_score_, 
    'CVStd' : log_reg_models.cv_results_['std_test_score'][log_reg_models.best_index_],
    'Result_train' : lr_best.predict(x_train),
    'Result_test' : lr_best.predict(x_test),
    'Model' : lr_best
}


# In[ ]:


print('Best model - avg:', 
      lr_model['CVScore'],
      '+/-', 
      lr_model['CVStd'])
print()
print(log_reg_models.best_estimator_)


# ## Random forest classifier
# 
# Random forests are a powerful classification method, but require a discrete time to train. We will do the same thing as before, using cross validation to find the best model with a grid search over a bunch of different parameters
# 
# Originally written with these parameters:
# 
#     rf_params = {
#         'n_estimators' :  [50, 100, 400, 700, 1000],
#         'max_features' : ['log2', 'sqrt', 'auto'],
#         'criterion' : ['entropy', 'gini'],
#         'min_samples_split' :  [2, 4, 10, 12, 16],
#         'min_samples_leaf' : [1, 5, 10],
#         'random_state' : [RANDOM_SEED]
#     }
# 
# 
# Then shorted out to save time.

# In[ ]:


rf_params = {
    'n_estimators' :  [50],
    'max_features' : ['log2'],
    'criterion' : ['gini'],
    'min_samples_split' :  [16],
    'min_samples_leaf' : [1],
    'random_state' : [RANDOM_SEED]
}

random_forest = ensemble.RandomForestClassifier()
acc_scorer = make_scorer(accuracy_score)
rf_models = GridSearchCV(random_forest, rf_params, scoring=acc_scorer, n_jobs=-1)
rf_models = rf_models.fit(x_train, y_train)

rf_best = rf_models.best_estimator_
rf_best = rf_best.fit(x_train, y_train)

rf_model = {
    'Name' : 'Random forest', 
    'CVScore' : rf_models.best_score_, 
    'CVStd' : rf_models.cv_results_['std_test_score'][rf_models.best_index_],
    'Result_train' : rf_best.predict(x_train),
    'Result_test' : rf_best.predict(x_test),
    'Model' : rf_best
}


# In[ ]:


best_idx = rf_models.best_index_
print('Best model - avg:', 
      rf_model['CVScore'],
      '+/-', 
      rf_model['CVStd'])
print()
print(rf_models.best_estimator_)


# In[ ]:


feature_importances = [(x, y) for x,y in zip(rf_best.feature_importances_, x_train.columns.values)]

feature_importances.sort(key = lambda x : x[0])
plt.figure(figsize=(8, 6))
plt.barh(range(len(feature_importances)), [x[0] for x in feature_importances], tick_label = [x[1] for x in feature_importances])


# XGBoost classifier
# --
# 
# This is a very good general purpose classifier. The training process will take about 1 minute.
# 
# Originally with this parameters:
# 
#     xgb_params = {
#         'max_depth' : [2, 3, 4, 5, 6],
#         'learning_rate' : [0.05, 0.01, 0.005],
#         'n_estimators' : [100, 300, 600],
#         'seed' : [RANDOM_SEED]
#     }

# In[ ]:


xgb_params = {
    'max_depth' : [5],
    'learning_rate' : [0.05],
    'n_estimators' : [100],
    'seed' : [RANDOM_SEED]
}

xgb_model = xgb.XGBClassifier()
acc_scorer = make_scorer(accuracy_score)
xgb_grid = GridSearchCV(xgb_model, xgb_params, scoring=acc_scorer)
xgb_grid = xgb_grid.fit(x_train, y_train)

xgb_best = xgb_grid.best_estimator_
xgb_best = xgb_best.fit(x_train, y_train)

xgb_model = {
    'Name' : 'XGBoost', 
    'CVScore' : xgb_grid.best_score_, 
    'CVStd' : xgb_grid.cv_results_['std_test_score'][xgb_grid.best_index_],
    'Result_train' : xgb_best.predict(x_train),
    'Result_test' : xgb_best.predict(x_test),
    'Model' : xgb_best
}


# In[ ]:


best_idx = xgb_grid.best_index_
print('Best model - avg:', 
      xgb_model['CVScore'],
      '+/-', 
      xgb_model['CVStd'])
print()
print(xgb_grid.best_estimator_)


# In[ ]:


xgb.plot_importance(xgb_best)


# ## Support vector classifier
# 
# SVC are also very good classifier, but requires a normalization phase in order to converge.

# In[ ]:


train_test = pd.concat([x_train, x_test, x_validation], ignore_index=True)
train_test_normalized = preprocessing.scale(train_test)
x_train_normalized = train_test_normalized[:len(x_train), :]
x_test_normalized = train_test_normalized[len(x_train):len(x_train) + len(x_test), :]
x_validation_normalized = train_test_normalized[len(x_train) + len(x_test):, :]


# And now for the training
# 
#     svm_params = {
#         'C' : [0.1, 0.3, 0.8, 0.9, 1.0, 2.0],
#         'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
#         'tol' : [1e-3, 1e-4],
#         'degree' : [2, 3, 4, 5],
#         'random_state' : [RANDOM_SEED]
#     }

# In[ ]:


svm_params = {
    'C' : [0.3],
    'kernel' : ['rbf'],
    'tol' : [1e-3],
    'degree' : [2],
    'random_state' : [RANDOM_SEED]
}
acc_scorer = make_scorer(accuracy_score)
svc = SVC()
svc_classifiers = GridSearchCV(svc, svm_params, scoring=acc_scorer, n_jobs=-1)
svc_classifiers = svc_classifiers.fit(x_train_normalized, y_train)

svc_best = svc_classifiers.best_estimator_
svc_best = svc_best.fit(x_train_normalized, y_train)

svc_model = {
    'Name' : 'SVC', 
    'CVScore' : svc_classifiers.best_score_, 
    'CVStd' : svc_classifiers.cv_results_['std_test_score'][svc_classifiers.best_index_],
    'Result_train' : svc_best.predict(x_train_normalized),
    'Result_test' : svc_best.predict(x_test_normalized),
    'Model' : svc_best
}


# In[ ]:


best_idx = svc_classifiers.best_index_
print('Best model - avg:', 
      svc_model['CVScore'], 
      '+/-', 
      svc_model['CVStd'])
print()
print(svc_classifiers.best_estimator_)


# # Ada boost
# 
# Also here, the parameters were:
# 
#     ada_params = {
#         'n_estimators' : [20, 50, 100, 500, 1000],
#         'learning_rate' : [0.1, 0.4, 0.3, 0.9, 1],
#         'algorithm' : ['SAMME', 'SAMME.R'],
#         'random_state' : [RANDOM_SEED]
#     }

# In[ ]:


ada_params = {
    'n_estimators' : [100],
    'learning_rate' : [0.1],
    'algorithm' : ['SAMME.R'],
    'random_state' : [RANDOM_SEED]
}

acc_scorer = make_scorer(accuracy_score)
ada_class = ensemble.AdaBoostClassifier()
ada_classifiers = GridSearchCV(ada_class, ada_params, scoring=acc_scorer, n_jobs=-1)
ada_classifiers = ada_classifiers.fit(x_train, y_train)

ada_best = ada_classifiers.best_estimator_
ada_best = ada_best.fit(x_train, y_train)

ada_model = {
    'Name' : 'Ada boost', 
    'CVScore' : ada_classifiers.best_score_, 
    'CVStd' : ada_classifiers.cv_results_['std_test_score'][ada_classifiers.best_index_],
    'Result_train' : ada_best.predict(x_train),
    'Result_test' : ada_best.predict(x_test),
    'Model' : ada_best
}


# In[ ]:


best_idx = ada_classifiers.best_index_
print('Best model - avg:', 
      ada_model['CVScore'], 
      '+/-', 
      ada_model['CVStd'])
print()
print(ada_classifiers.best_estimator_)


# # Ensemble
# 
# We have seen that overall the XGBClassifier is the one that performs the best.

# In[ ]:


class Ensemble:
    def __init__(self, models, svc_last=True):
        self.models = models[:]
        self.svc_last = svc_last
    
    def fit(self, X_train, y_train, X_train_normalized):
        fitted_models = [m['Model'].fit(X_train, y_train) for m in self.models]
        for i in range(len(fitted_models)):
            self.models[i]['Model'] = fitted_models[i]
    
    def predict(self, X_test, X_test_normalized):
        predictions = []
        for m in self.models:
            if m['Name'] == 'SVC':
                predictions.append(m['Model'].predict(X_test_normalized))
            else:
                predictions.append(m['Model'].predict(X_test))
        
        df = pd.DataFrame(np.array(predictions), index=[m['Name'] for m in self.models])
        
        return df.apply(lambda x : 0 if np.sum(x) <= 2 else 1)

    def __repr__(self):
        return "Ensemble(" + ', '.join([m['Name'] for m in self.models]) + ")"


# In[ ]:


ens = Ensemble([rf_model, ada_model, xgb_model, lr_model, svc_model])
predictions = ens.predict(x_test, x_test_normalized).values


# In[ ]:


ensemble_model = {
    'Name' : 'Ensemble', 
    'CVScore' : 0, 
    'CVStd' : 0,
    'Result_train' : [],
    'Result_test' : predictions,
    'Model' : ens
}

answer_df = pd.DataFrame()
answer_df['PassengerId'] = test['PassengerId']
answer_df['Survived'] = predictions

answer_df.to_csv('results_ensemble.csv', index=False)


# # Stacking

# In[ ]:


def get_stacked(x, x_normalized, models):
    predictions = []
    for m in models:
        if m['Name'] == 'SVC':
            predictions.append(m['Model'].predict(x_normalized))
        else:
            predictions.append(m['Model'].predict(x))
    stack = pd.DataFrame(np.array(predictions).T, columns=[m['Name'] for m in models])
    return pd.concat([x, stack], axis=1)

stacking_models = [rf_model, ada_model, svc_model]
train_stacked = get_stacked(x_train, x_train_normalized, stacking_models)
test_stacked = get_stacked(x_test, x_test_normalized, stacking_models)


# These are the initial parameters I used for GridSearch:
# 
#     xgb_params = {
#         'max_depth' : [2, 3, 4, 5, 6],
#         'learning_rate' : [0.05, 0.01],
#         'n_estimators' : [30, 100, 300, 600],
#         'seed' : [RANDOM_SEED]
#     }

# In[ ]:


xgb_params = {
    'max_depth' : [2],
    'learning_rate' : [0.05],
    'n_estimators' : [30],
    'seed' : [RANDOM_SEED]
}

xgb_stacked = xgb.XGBClassifier()
acc_scorer = make_scorer(accuracy_score)
xgb_stacked_grid = GridSearchCV(xgb_stacked, xgb_params, scoring=acc_scorer)
xgb_stacked_grid = xgb_stacked_grid.fit(train_stacked, y_train)

stacked_best = xgb_stacked_grid.best_estimator_.fit(train_stacked, y_train)

stacked_model = {
    'Name' : 'Stacking', 
    'CVScore' : xgb_stacked_grid.best_score_, 
    'CVStd' : xgb_stacked_grid.cv_results_['std_test_score'][xgb_stacked_grid.best_index_],
    'Result_train' : xgb_stacked_grid.predict(train_stacked),
    'Result_test' : xgb_stacked_grid.predict(test_stacked),
    'Model' : stacked_best
}


# In[ ]:


print('Best model - avg:', 
      stacked_model['CVScore'], 
      '+/-', 
      stacked_model['CVStd'])
print()
print(stacked_best)


# # Check the validation

# In[ ]:


models = [lr_model, rf_model, ada_model, xgb_model, svc_model, ensemble_model, stacked_model]
models_df = pd.DataFrame(models, index=[m['Name'] for m in models])


# In[ ]:


x_validation_stacked = get_stacked(x_validation, x_validation_normalized, stacking_models)

def get_validation_predictions(x):
    if x['Name'] == 'SVC':
        return x['Model'].predict(x_validation_normalized)
    elif x['Name'] == 'Ensemble':
        return x['Model'].predict(x_validation, x_validation_normalized)
    elif x['Name'] == 'Stacking':
        return x['Model'].predict(x_validation_stacked)
    else:
        return x['Model'].predict(x_validation)

models_df['ValidationScore'] = models_df.apply(lambda x : accuracy_score(get_validation_predictions(x), y_validation), axis=1)

models_df['ValidationScore']


# # Result output
# 
# Overall, it seems that the SVC generalizes better, so we will use that SVC to output the result. 
# 
# Note that here a statistical test should be carried out, because the accuracy alone is usually not enough. One should use cross validation instead of holdout and see if there is a statistical significance of a given model over the others.

# In[ ]:


best_model = svc_model['Model']
best_model = best_model.fit(np.concatenate([x_train_normalized, x_validation_normalized]), np.concatenate([y_train, y_validation]))

predictions = best_model.predict(x_test_normalized)

result = pd.DataFrame()
result['PassengerId'] = test['PassengerId']
result['Survived'] = predictions

result.to_csv('results.csv', index=False)


# # References
# 
# - [Titanic Random Forest: 82.78%](https://www.kaggle.com/zlatankr/titanic/titanic-random-forest-82-78/run/806902)
# - [Scikit-Learn ML from Start to Finish](https://www.kaggle.com/jeffd23/titanic/scikit-learn-ml-from-start-to-finish/run/320209)
# - [A Journey through Titanic](https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic/run/447794)
