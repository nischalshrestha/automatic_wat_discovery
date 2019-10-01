#!/usr/bin/env python
# coding: utf-8

# # Predicting Survival in the Titanic
# By Luis Alberto Denis

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# ## Load data

# In[ ]:


raw_train = pd.read_csv('../input/train.csv')
raw_test = pd.read_csv('../input/test.csv')


# In[ ]:


raw_train.head()


# In[ ]:


raw_train.info()
print('-'*40)
raw_test.info()

Observation: The info method shows that the training set [Age, Cabin and Embarked] and the test set [Age, Fare, Cabin] contain missing values. 
# In[ ]:


raw_train.describe()

Observation: The minimum value in the feature Age is represented as a fraction, it means there were babies with less than one year old aboard the Titanic.
# In[ ]:


correlation = raw_train.corr()['Survived']


# In[ ]:


correlation.sort_values(ascending=False)

Observation: The coefficient of determination indicates the feature Survived is correlated with Fare and Pclass, these could be good features for the model.
# ## Survival Analysis

# In[ ]:


raw_train['Survived'].value_counts()

Observation: In the training set, there are 342 passengers who survived, this tells us that the survival rate was 38.88%.
# ## Feature Name Analysis

# In[ ]:


raw_train['Name'].unique().size

Observation: Every passenger had a different name, if we leave this feature as it is, it wont be of use.
# In[ ]:


raw_train['Name'].head()

Observation: Each name has a title in it. This could be useful to know if a passenger was married, or if the passenger belonged to royalty.
# ## Feature Sex Analysis

# In[ ]:


raw_train[['Sex', 'PassengerId']].groupby(['Sex'], as_index=True).count()

Observation: In the training dataset there are 577 men, which represents 64.75% of all passengers and crew members.
# In[ ]:


raw_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean()

Observation: Females have a survival rate much higher than males, so sex == female is a good indicative of survival. 
# In[ ]:


data_sex = raw_train[['Sex', 'Pclass', 'Survived']].groupby(['Sex', 'Pclass'], as_index=True).mean()

data_sex.loc['female'].plot(kind='bar', ylim=[0, 1], title='Female Survival Rate', legend=False)
data_sex.loc['male'].plot(kind='bar', ylim=[0, 1], title='Male Survival Rate', legend=False)

Observation: Almost every female from 1st and 2nd Pclass survived.
Observation: No matter the Pclass males have a low chance of survival. (Survival in 1st Pclass is less than 40%)
# ## Feature Pclass Analysis

# In[ ]:


data_pclass = raw_train[['Pclass', 'PassengerId']].groupby(['Pclass'], as_index=True).count()
data_pclass


# In[ ]:


data_pclass.plot(kind='pie', y='PassengerId', legend=False, title='Pclass percentage of total', figsize=(5, 5))

Observation: There are more 3rd Pclass passengers than 1st and 2nd Pclass together.
# In[ ]:


raw_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

Observation: 62.96% of passengers in 1st Pclass survived, maybe the feature Pclass would be a good predictor variable.
# ## Feature Fare Analysis

# In[ ]:


raw_train['Fare'].hist(bins=100, color='red')


# In[ ]:


raw_train['Fare'].hist(bins=100, normed=True, alpha=0.5, color='red')
raw_train['Fare'].plot(kind='kde', style='k--')


# In[ ]:


raw_train['Fare'].skew()

Observation: The Fare distribution is very skewed. This can lead to overweight very high values in the model, even if it is scaled. It is better to transform it with the log function to reduce the skew. [Thanks to Yassine Ghouzam]
# In[ ]:


plt.boxplot(raw_train['Fare'], vert=False)


# In[ ]:


raw_train.loc[raw_train['Fare'] > 100, 'Pclass'].value_counts()

Observation: There are 53 passengers who their fares are considered outliers for the distribution, but as you can see all of them belonged to 1st class. Knowing this, these passengers won't be removed from the training set (Some people travel with a lot of luxury). 
# ## Feature SibSp Analysis

# In[ ]:


raw_train['SibSp'].value_counts()


# In[ ]:


raw_train['SibSp'].hist(bins=20, color='orange')

Observation: 68.23% of the passengers in the training set did not have any Siblings/Spouses aboard the Titanic and 23.45% travelled with only one.
# In[ ]:


raw_train['SibSp'].hist(bins=20, color='orange', normed=True, alpha=0.5)
raw_train['SibSp'].plot(kind='kde', color='black')


# In[ ]:


data_sibsp = raw_train[['SibSp', 'Survived']].groupby('SibSp', as_index=True).mean()
data_sibsp.plot(kind='bar', color='orange', ylim=[0, 1])

Observation: Being in the Titanic with 1 or 2 SibSp can bring up the survival rate.
# In[ ]:


data_sibsp = raw_train[['SibSp', 'Sex', 'Survived']].groupby(['Sex', 'SibSp'], as_index=True).mean()
data_sibsp.loc['male'].plot(kind='bar', color='orange', title='Male SibSp', ylim=[0, 1])
data_sibsp.loc['female'].plot(kind='bar', color='orange', title='Female SibSp', ylim=[0, 1])

Observation: Indeed, men with 1 or 2 SibSp aboard have the best chance to survive. For women, more than 2 SibSp aboard can drop their survival rate.
# ## Feature Parch Analysis

# In[ ]:


raw_train['Parch'].value_counts()


# In[ ]:


raw_train['Parch'].hist(bins=20, color='magenta')

Observation: Almost 700 passengers travelled without Parents/Children.
# In[ ]:


raw_train['Parch'].hist(bins=20, color='magenta', normed=True, alpha=0.5)
raw_train['Parch'].plot(kind='kde', color='black')


# In[ ]:


data_parch = raw_train[['Parch', 'Survived']].groupby('Parch', as_index=True).mean()
data_parch.plot(kind='bar', color='magenta', ylim=[0, 1])

Observation: Passengers with 1, 2 or 3 Parch are the most likely to survive. This seems reasonable, I imagine parents trying to get their sons first into the boats. 
# In[ ]:


data_parch = raw_train[['Parch', 'Sex', 'Survived']].groupby(['Sex', 'Parch'], as_index=True).mean()
data_parch.loc['male'].plot(kind='bar', color='magenta', ylim=[0, 1])
data_parch.loc['female'].plot(kind='bar', color='magenta', ylim=[0, 1])

Observation: For females the picture is crystal clear, being alone or with up to 3 Parch members is positive influence in their survival. So no matter what the females were prioritized. For males is much complicated. The data shows that men being accompanied with 1 or 2 Parch have a survival rate of 30% +.
# ## Feature Age Analysis

# In[ ]:


raw_train['Age'].isnull().sum()

Observation: The 19.86% of the feature values are missing. 
# In[ ]:


raw_train['Age'].hist(bins=100, grid=True, alpha=0.8, color='gray')

Observation: There are many infants (between the ages of 1 and 5), see how the bell shape of the distribution is ruined by the first bars.
# In[ ]:


raw_train['Age'].hist(bins=100, grid=True, alpha=0.8, normed=True, color='gray')
raw_train.loc[raw_train['Survived']==0, 'Age'].plot(kind='kde', color='red', label='Not Survived', legend=True)
raw_train.loc[raw_train['Survived']==1, 'Age'].plot(kind='kde', color='blue', label='Survived', legend=True)

Observation: The line in blue shows two interesting things:
                - Children up to 5 years old have a very good chance of survive.
                - Elderly people have few or none chance of survival (See how the line drops at the end).
# ## Feature Embarked Analysis

# In[ ]:


raw_train['Embarked'].value_counts().plot(kind='bar', title='Amount of passenger per port', color='g')


# In[ ]:


data_embarked = raw_train[['Embarked', 'Survived']].groupby('Embarked').mean()
data_embarked.plot(kind='bar', ylim=[0, 1], legend=False, title='Survival Rate per Port', color='g')

Observation: The high chance of survival that can be seen in this graph for port C could be explained because the majority of people who boarded from this port were 1st class passengers.
# In[ ]:


pd.crosstab(raw_train['Embarked'], raw_train['Pclass']).plot(kind='bar', title='Amount of Passengers per Port')


# ## Feature Ticket Analysis

# In[ ]:


raw_train['Ticket'].unique().size

Observation: There are 681 unique tickets in the training set and 891 passengers, this means that some people shared the same ticket.
# In[ ]:


raw_train['Ticket'].head()

Observation: Seems that Ticket's structure does not follow any pattern. Some tickets are formed by joining an alphanumeric secuence and a numeric one. Others are just a numeric secuence. For this feature to be of any use for a learning algorithm, some feature engineering should be done.
# ## Feature Cabin Analysis

# In[ ]:


raw_train['Cabin'].unique()

Observation: The cabin feature if formed by joining the deck information, and the room number.
# In[ ]:


raw_train.loc[raw_train['Cabin'].isnull(), 'Cabin'].size

Observation: The 77.10% of the cabin's data is missing.
# ## Filling of Missing Values in Fare

# In[ ]:


from sklearn.preprocessing import Imputer


# In[ ]:


fare_imputer = Imputer(strategy='median')
fare_imputer.fit(raw_train['Fare'].values.reshape(-1, 1))

raw_test['Fare'] = fare_imputer.transform(raw_test['Fare'].values.reshape(-1, 1))


# ## Feature LogFare

# In[ ]:


raw_train['LogFare'] = raw_train['Fare'].apply(lambda x: np.log(x) if x > 0 else 0)

raw_test['LogFare'] = raw_test['Fare'].apply(lambda x: np.log(x) if x > 0 else 0)


# In[ ]:


raw_train['LogFare'].hist(bins=100, normed=True, alpha=0.5, color='yellow')
raw_train['LogFare'].plot(kind='kde', style='k--')


# In[ ]:


raw_train['LogFare'].skew()

Observation: After applying the log function to the Fare feature the skewness clearly disappears.
# ## Filling of Missing Values in Cabin

# In[ ]:


raw_train['Cabin'] = raw_train['Cabin'].fillna('U') #Unknown

raw_test['Cabin'] = raw_test['Cabin'].fillna('U')


# ## Feature Deck

# In[ ]:


raw_train['Deck'] = raw_train['Cabin'].apply(lambda x: x[0])

raw_test['Deck'] = raw_test['Cabin'].apply(lambda x: x[0])


# In[ ]:


raw_train['Deck'].value_counts().plot(kind='bar', color='brown')


# In[ ]:


raw_train[['Deck', 'Survived']].groupby('Deck').mean().plot(kind='bar', color='brown')


# ## Feature Title

# In[ ]:


def extract_title_from_name(name):
    for word in name.split():
        if word.endswith('.') and len(word) > 2: return word[:-1]
    return None

raw_train['Title'] = raw_train['Name'].apply(lambda x: extract_title_from_name(x))

raw_test['Title'] = raw_test['Name'].apply(lambda x: extract_title_from_name(x))

raw_train['Title'].unique()


# In[ ]:


raw_test['Title'].unique()

Observation: In the test set appears a title that is not in the training set 'Dona'. We have to include it in the title mapping dict.
# In[ ]:


pd.crosstab(raw_train['Title'], raw_train['Sex'])

Observation: There is a female doctor in the training set. 
# In[ ]:


raw_train.loc[(raw_train['Title']=='Dr') & (raw_train['Sex']=='female')]


# In[ ]:


title_mapping = {'Capt':'Mr', 'Col':'Mr','Don':'Mr','Dona':'Mrs',
                 'Dr':'Mr','Jonkheer':'Mr','Lady':'Mrs','Major':'Mr',
                 'Master':'Master','Miss':'Miss','Mlle':'Miss','Mme':'Mrs',
                 'Mr':'Mr','Mrs':'Mrs','Ms':'Miss','Rev':'Mr','Sir':'Mr',
                 'Countess':'Mrs'}

raw_train.loc[(raw_train['Title']=='Dr') & (raw_train['Sex']=='female'),'Title'] = 'Mrs'
raw_train['Title'] = raw_train['Title'].map(title_mapping)

raw_test['Title'] = raw_test['Title'].map(title_mapping)


# ## Filling of Missing Values in Age

# In[ ]:


raw_train[raw_train['Age'].isnull()]['Title'].value_counts()


# In[ ]:


raw_train['Age'].hist(bins=100, normed=True, alpha=0.5, color='gray')
raw_train['Age'].plot(kind='kde', style='b--')


# In[ ]:


ages = dict()

for title in raw_train['Title'].unique():
    ages[title] = dict()

for title in ages.keys():    
    for pclass in raw_train['Pclass'].unique():
        ages[title][pclass] = raw_train[(raw_train['Title'] == title) & (raw_train['Pclass'] == pclass)]['Age'].median()

ages


# In[ ]:


raw_train['Age'] = raw_train['Age'].fillna(-1)
for index, row in raw_train.iterrows():
    if row['Age'] == -1:
        raw_train.loc[index, 'Age'] = ages[row['Title']][row['Pclass']]
        
raw_test['Age'] = raw_test['Age'].fillna(-1)
for index, row in raw_test.iterrows():
    if row['Age'] == -1:
        raw_test.loc[index, 'Age'] = ages[row['Title']][row['Pclass']]


# In[ ]:


raw_train['Age'].hist(bins=100, normed=True, alpha=0.5, color='gray')
raw_train['Age'].plot(kind='kde', style='r--')

Observation: Maybe the approach is not very effective. Clearly filling the missing values with the median values of 'Title & Pclass' have affected a little bit the age distribution, you can see those 4 high bars [18, 26, 31, 42]
# ## Filling of Missing Values in Embarked

# In[ ]:


raw_train.loc[raw_train['Embarked'].isnull()]

Observation: The features Pclass and Fare could be use to determine the missing Embarked information.
# In[ ]:


filling_data = raw_train.loc[raw_train['Pclass'] == 1, ['Fare', 'Embarked']].groupby('Embarked', as_index=True)
filling_data.boxplot(subplots=False)

Observation: The median value of Fare for those embarked in port C is the closest one to these passengers' Fare.
# In[ ]:


raw_train['Embarked'] = raw_train['Embarked'].fillna('C')


# In[ ]:


raw_train['Embarked'].count()


# ## Feature Male

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


bin_sex = LabelEncoder()
raw_train['Male'] = bin_sex.fit_transform(raw_train['Sex'])

raw_test['Male'] = bin_sex.transform(raw_test['Sex'])


# In[ ]:


raw_train['Male'].value_counts()


# ## Feature FamilyMembers
Observation: There are features like SibSp and Parch which represent family relations. Maybe it would be wise to combine them to have the family information together.
# In[ ]:


raw_train['FamilyMembers'] = raw_train['SibSp'] + raw_train['Parch']

raw_test['FamilyMembers'] = raw_test['SibSp'] + raw_test['Parch']


# In[ ]:


raw_train['FamilyMembers'].value_counts().sort_index().plot(kind='bar',  legend=False, color='c')

Observation: There are more than 500 passengers who boarded the Titanic with no family at all. They represent the 60% of the passengers in the training set.
# In[ ]:


raw_train[['FamilyMembers', 'Survived']].groupby('FamilyMembers').mean().plot(kind='bar', legend=False, color='c')

Observation: This graph shows that there are some groups with higher survival rate than others, it would be wise to join those groups together.
# ## Binning Feature FamilyMembers into FamiySize

# In[ ]:


def binning_family(x):
    if x == 0:
        return 'Alone'
    elif (x > 0) & (x <= 3):
        return 'Small'
    elif (x > 3):
        return 'Large'

raw_train['FamilySize'] = raw_train['FamilyMembers'].apply(lambda x: binning_family(x))

raw_test['FamilySize'] = raw_test['FamilyMembers'].apply(lambda x: binning_family(x))


# In[ ]:


raw_train['FamilySize'].value_counts()


# ## Categorical Embarked into Dummies

# In[ ]:


from sklearn.preprocessing import LabelBinarizer


# In[ ]:


bin_embarked = LabelBinarizer()

ports = bin_embarked.fit_transform(raw_train['Embarked'])
ports_df = pd.DataFrame(ports, columns=['Port_' + p for p in bin_embarked.classes_.tolist()])
raw_train = raw_train.join(ports_df)

ports = bin_embarked.transform(raw_test['Embarked'])
ports_df = pd.DataFrame(ports, columns=['Port_' + p for p in bin_embarked.classes_.tolist()])
raw_test = raw_test.join(ports_df)


# ## Categorical FamilySize into Dummies

# In[ ]:


bin_fsize = LabelBinarizer()

fsize = bin_fsize.fit_transform(raw_train['FamilySize'])
fsize_df = pd.DataFrame(fsize, columns=['FamilySize_' + f for f in bin_fsize.classes_.tolist()])
raw_train = raw_train.join(fsize_df)

fsize = bin_fsize.transform(raw_test['FamilySize'])
fsize_df = pd.DataFrame(fsize, columns=['FamilySize_' + f for f in bin_fsize.classes_.tolist()])
raw_test = raw_test.join(fsize_df)


# ## Categorical Deck into Dummies

# In[ ]:


bin_deck = LabelBinarizer()

decks = bin_deck.fit_transform(raw_train['Deck'])
decks_df = pd.DataFrame(decks, columns=['Deck_' + d for d in bin_deck.classes_.tolist()])
raw_train = raw_train.join(decks_df)

decks = bin_deck.transform(raw_test['Deck'])
decks_df = pd.DataFrame(decks, columns=['Deck_' + d for d in bin_deck.classes_.tolist()])
raw_test = raw_test.join(decks_df)


# ## Scaling of Feature Age

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


age_scaler = MinMaxScaler()
raw_train['Age'] = age_scaler.fit_transform(raw_train['Age'].values.reshape(-1, 1))

raw_test['Age'] = age_scaler.transform(raw_test['Age'].values.reshape(-1, 1))


# In[ ]:


raw_train['Age'].hist(bins=100, normed=True, alpha=0.5)
raw_train['Age'].plot(kind='kde')


# ## Removing Features

# In[ ]:


train = raw_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Title', 'Ticket', 'Cabin', 'Sex', 'Embarked', 'FamilyMembers', 'FamilySize', 'Deck', 'Fare'], axis=1)
test = raw_test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Title', 'Ticket', 'Cabin', 'Sex', 'Embarked', 'FamilyMembers', 'FamilySize', 'Deck', 'Fare'], axis=1)


# In[ ]:


train.head()


# In[ ]:


test.tail()


# ## Correlation with Survived

# In[ ]:


corr_with_survived = train.corrwith(train['Survived'])
corr_with_survived


# ## Model Selection

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


# In[ ]:


y = train['Survived']
X = train.drop('Survived', axis=1)


# In[ ]:


seed = 4

Observation: The used of a seed to set a random state in some classifiers is very important because it allows to replicate the results.
# In[ ]:


gradientb = GradientBoostingClassifier(random_state=seed)
scores_mlp = cross_val_score(gradientb, X, y, cv=10)


# In[ ]:


rforest = RandomForestClassifier(random_state=seed)
scores_rf = cross_val_score(rforest, X, y, cv=10)


# In[ ]:


logistic_r = LogisticRegression(random_state=seed)
scores_lr = cross_val_score(logistic_r, X, y, cv=10)


# In[ ]:


support_v = SVC(random_state=seed)
scores_sv = cross_val_score(support_v, X, y, cv=10)


# In[ ]:


knn = KNeighborsClassifier()
scores_knn = cross_val_score(knn, X, y, cv=10)


# In[ ]:


list_scores = [
    scores_mlp.mean(), 
    scores_rf.mean(), 
    scores_lr.mean(), 
    scores_sv.mean(), 
    scores_knn.mean()
]
list_std = [
    scores_mlp.std(),
    scores_rf.std(),
    scores_lr.std(),
    scores_sv.std(),
    scores_knn.std()
]
columns = [
    'GradientBoosting',
    'RandomForest',
    'LogisticRegression',
    'SupportVector',
    'KNearestNeighbors'
]

scores = pd.DataFrame(columns=columns)
scores.loc['scores'] = list_scores
scores.loc['std'] = list_std
scores


# ## Fine-Tune the Hyperparameters

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_gb = {
    'loss' : ["deviance"],
    'n_estimators' : [75, 100, 200],
    'learning_rate': [0.1, 0.7, 0.05, 0.03, 0.01],
    'max_depth': [5, 8, None],
    'min_samples_leaf': [25, 50, 75, 100],
    'max_features': [1.0, 0.3, 0.1] 
}

grid_search_gb = GridSearchCV(GradientBoostingClassifier(random_state=seed), param_gb, cv=10, n_jobs=-1)
grid_search_gb.fit(X, y)
grid_search_gb.best_params_


# In[ ]:


param_rf = {
    "max_depth": [5, 8, None],
    "min_samples_split": [2, 5, 10, 15, 100],
    "min_samples_leaf": [5, 10, 25, 50],
    "max_features": ['log2', 'sqrt', None],
    "n_estimators": [75, 100, 200]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=seed), param_rf, cv=10, n_jobs=-1)
grid_search_rf.fit(X, y)
grid_search_rf.best_params_


# In[ ]:


param_lr = {
    "penalty": ["l2"],
    "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

grid_search_lr = GridSearchCV(LogisticRegression(random_state=seed), param_lr, cv=10, n_jobs=-1)
grid_search_lr.fit(X, y)
grid_search_lr.best_params_


# In[ ]:


param_sv = {
    "kernel": ['rbf'],
    "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "gamma": [0.001, 0.01, 0.1, 1]
}

grid_search_sv = GridSearchCV(SVC(probability=True, random_state=seed), param_sv, cv=10, n_jobs=-1)
grid_search_sv.fit(X, y)
grid_search_sv.best_params_


# In[ ]:


param_knn = {
    "n_neighbors": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "p": [2, 3],
    "weights": ['uniform', 'distance']
}

grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_knn, cv=10, n_jobs=-1)
grid_search_knn.fit(X, y)
grid_search_knn.best_params_


# In[ ]:


list_scores_tuning = [grid_search_gb.best_score_, grid_search_rf.best_score_, grid_search_lr.best_score_, grid_search_sv.best_score_, grid_search_knn.best_score_]
index_tuning = ['GradientBoosting', 'RandomForest', 'LogisticRegression', 'SupportVector', 'KNearestNeighbors']
scores_tuning = pd.Series(list_scores_tuning, index=index_tuning)
scores_tuning.sort_values(ascending=False)


# ## Evaluating the Model

# In[ ]:


from sklearn.ensemble import VotingClassifier

estimators = [
    ('gb', grid_search_gb.best_estimator_),
    ('rf', grid_search_rf.best_estimator_),
    ('lr', grid_search_lr.best_estimator_),
    ('sv', grid_search_sv.best_estimator_),
    ('knn', grid_search_knn.best_estimator_)
]

voting_classifier = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
cross_val_score(voting_classifier, X, y, cv=10).mean()


# ## Making predictions

# In[ ]:


voting_classifier = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
voting_classifier.fit(X, y)
predictions = voting_classifier.predict(test)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": raw_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('predictions.csv', index=False)

