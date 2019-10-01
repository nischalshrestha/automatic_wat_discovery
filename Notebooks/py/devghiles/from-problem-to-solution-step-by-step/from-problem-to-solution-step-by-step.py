#!/usr/bin/env python
# coding: utf-8

# # Titanic

# In this kernel, I'm working on the Titanic kaggle challenge: https://www.kaggle.com/c/titanic

# ## Load libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# ## Exploratory Data Analysis

# ### Read the data

# In[ ]:


# read all the data in the train file
data = pd.read_csv('../input/train.csv')


# In[ ]:


data.head()


# The data consists of a table of 11 columns:
# * PassengerId: obvious
# * Survived: 0 -> no, 1 -> yes; the target variable
# * Pclass: 1 -> upper, 2 -> middle, 3 -> lower
# * Name: passenger name, probably irrelevant
# * Sex: male, female
# * Age: obvious
# * SibSp: # of siblings or spouses aboard
# * Parch: # of parent or children aboard
# * Ticket: ticket number (I doubt it will be relevant)
# * Fare: depends on the class I guess
# * Cabin: cabin number (I don't think it will be relvant. All the information it may bring is I think -a priori- carried by Pclass and eventually Fare)
# * Embarked: port of embarkation; C = Cherbourg, Q = Queenstown, S = Southampton (I would say it's irrelevant, we'll see if it is or not)

# ### Clean the data

# In[ ]:


data.count()


# There are 891 rows. We see that the columns 'Age', 'Cabin' and 'Embarked' have missing values; we need to fill them. We'll also transform the categorical data into numerical (ordinal) data because it's easier to work with. Finally, we'll drop the 'PassengerId' and 'Name' columns because they don't provide any information necessary for the classification.

# In[ ]:


def preprocess(df):
    # fill in the missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Cabin'].fillna('UNK', inplace=True)
    df['Embarked'].fillna('UNK', inplace=True)
    
    # categorical data encoding
    enc = LabelEncoder()
    columns_to_encode = ['Sex', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    for column in columns_to_encode:
        df[column] = enc.fit_transform(df[column])
    
    # drop the 'PassengerId' and 'Name' columns
    df.drop('PassengerId', axis=1, inplace=True)
    df.drop('Name', axis=1, inplace=True)


# In[ ]:


preprocess(data)
print(data.count())
data.head()


# Now, all the columns have 891 values, and we see numbers, numbers everywhere, that's what we want.

# ### Explore the data

# In[ ]:


data.describe()


# In[ ]:


# empirical distributions of the features and the target
for column in data.columns:
    plt.hist(data[column], weights=np.ones(len(data)) / len(data))
    plt.title('{0} empirical distribution'.format(column))
    plt.xlabel('{0}'.format(column))
    plt.ylabel('Empirical PDF')
    plt.show()


# We can make the following remarks:
# * Around 60% of the passengers survived (and around 40% died). The classes are relatively well balanced, so it makes sense to use the accuracy as the evaluation metric.
# * The lower class (Pclass = 3) makes up more than half of the passenger. The upper and middle class are somewhat similarly represented.
# * Females make up around 65% of the passengers, and males around 35%.
# * The age is distributed following what seems to be a gaussian distribution, with it's mean around 30. We must keep in mind that a significant number of the age values were missing (around 20%) and were replaced by the remaining values's median.
# * The SibSp column follows a Poisson distribution, with the value 0 being highly represented, the value 1 a little bit, and the other ones from there very rarely.
# * The Parch column also follows a Poisson distribution.
# * Tickets follow an almost uniform distribution.
# * Fares also follow a uniform-like distribution, with the exception of some values which are more highly represented.
# * Cabin is a column which has a lot of missing values (around 77%). Those missing values corresond to the pick in the histogram. The other values seem to be distributed uniformly, although we have to rescale and have more data in order to asses that more confidently. In any case, this information is probably useless due to the number of missing values.
# * Embarked has very few missing values. There are 3 ports of embarkation, they represent around 10%, 20% and 70% of the passengers respectively.

# In[ ]:


# correlation
corr = data.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# From the correlation matrix, we see that the 'Survived' variable is highly correlated with 'Sex', 'Pclass', and surprisingly 'Fare'. The correlation with 'Age' is not as high as I expected it, maybe because of the missing values that were replaced by the median.

# In[ ]:


plt.hist(data['Age'].loc[data['Survived'] == 1], color='g', label='Survived')
plt.hist(data['Age'].loc[data['Survived'] == 0], color='r', label='Died')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()


# The 2 distribution almost overlap. I guess age really doesn't play a significant role, although one might notice that for ages < 10, the survival rate is significantly higher; but this category is rare relative to the general population so it won't provide much discriminative information.

# ## Baseline: random forest

# We'll use a random forest classifier with the scikit-learn default hyperparameters as our baseline.

# In[ ]:


X = data.drop('Survived', axis=1).values
y = data['Survived'].values
clf = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# The baseline is about 80%, which is very high for a baseline. But let's not forget that the dataset is relatively small, and the test set used here contains less than 200 entries. Results may vary if I use a larger dataset.
# 
# Now I'll work on improving that.

# ## Classification

# We'll first prepare the data for classification (mostly by doing one-hot encoding). Then, we'll test a bunch of classical classification algorithm using (mostly) the default hyperparameters of scikit-learn. After that, we'll do some hyperparameter tuning using cross-validation on the best performing basic classifiers.

# ### Feature extraction

# Age, SibSp and Parch will remain as they are, their quantitative nature corresponds to what they mean in reality.
# 
# One-hot encoding will be applied to Pclass, Sex and Embarked.
# 
# Ticket, Fare and Cabin represent categorical data with numerous possible values. We're going to keep only the most common values.

# In[ ]:


encoders = dict()


# In[ ]:


def feature_extraction(df, test=False):
    # Age, SibSp and Parch
    X = np.concatenate((df['Age'].values.reshape(-1, 1),
                       df['SibSp'].values.reshape(-1, 1),
                       df['Parch'].values.reshape(-1, 1)),
                       axis=1)
    
    # one-hot encoding of Pclass, Sex and Embarked
    for column in ['Pclass', 'Sex', 'Embarked']:
        if test:
            values = np.array(encoders[column].transform(df[column].values.reshape(-1, 1)).todense())
        else:
            encoders[column] = OneHotEncoder(handle_unknown='ignore')
            values = np.array(encoders[column].fit_transform(df[column].values.reshape(-1, 1)).todense())
        X = np.concatenate((X, values), axis=1)
    
    # For columns Ticket, Fare and Cabin, we only keep the most common values
    num_values_to_keep = {
        'Ticket': 8,
        'Fare': 15,
        'Cabin': 4
    }
    for column in ['Ticket', 'Fare', 'Cabin']:
        if not test:
            counts = Counter(df[column])
            most_common_counts = counts.most_common(num_values_to_keep[column])
            values_to_keep = list(map(lambda x: x[0], most_common_counts))
            encoders[column] = OneHotEncoder(handle_unknown='ignore')
            encoders[column].fit(np.array(values_to_keep).reshape(-1, 1))
        values = np.array(encoders[column].transform(df[column].values.reshape(-1, 1)).todense())
        X = np.concatenate((X, values), axis=1)
    
    return X


# In[ ]:


X = feature_extraction(data)
y = data['Survived'].values


# ### Basic classifiers

# In[ ]:


clfs = {
    'mnb': MultinomialNB(),
    'gnb': GaussianNB(),
    'svm1': SVC(kernel='linear'),
    'svm2': SVC(kernel='rbf'),
    'svm3': SVC(kernel='sigmoid'),
    'mlp1': MLPClassifier(),
    'mlp2': MLPClassifier(hidden_layer_sizes=[100, 100]),
    'ada': AdaBoostClassifier(),
    'dtc': DecisionTreeClassifier(),
    'rfc': RandomForestClassifier(),
    'gbc': GradientBoostingClassifier(),
    'lr': LogisticRegression()
}


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
accuracies = dict()
for clf_name in clfs:
    clf = clfs[clf_name]
    clf.fit(X_train, y_train)
    accuracies[clf_name] = clf.score(X_valid, y_valid)


# In[ ]:


accuracies


# Good results seem to be given by the SVM with RBF kernel and by the 2 layers feedforward neural network. I'll perform hyperparameter optimization on these two models next. Other models also give good results (AdaBoost, Gradient Boosting, Multinomial Naive Bayes, ...) but for now, I'll just work on the SVM and the neural network.

# ### SVM hyperparameters optimization

# In[ ]:


# the kernel is RBF
parameters = {
    'C': [1, 10, 100],
    'gamma': [0.001, 0.01, 0.1]
}


# In[ ]:


svc = SVC()
clf = GridSearchCV(svc, parameters, scoring='accuracy', return_train_score=True)
clf.fit(X, y)


# In[ ]:


pd.DataFrame(clf.cv_results_)


# The best score (for the test) is $0.82$ and it is obtained for $C = 10$ and $\gamma = 0.01$. This is a little bit disappointing knowing that we got the same score in our baseline.

# ### Neural network hyperparameters optimization

# In the first time, we'll tune the neural network's number of layers and number of units per layer. After that, we'll optimize other hyperparameters.

# In[ ]:


parameters = {
    'hidden_layer_sizes': [
        [50,],
        [100,],
        [200,],
        [50, 50],
        [50, 100],
        [100, 50],
        [100, 100],
        [200, 200],
        [100, 100, 100]
    ]
}


# In[ ]:


mlp = MLPClassifier()
clf = GridSearchCV(mlp, parameters, scoring='accuracy', return_train_score=True)
clf.fit(X, y)


# In[ ]:


pd.DataFrame(clf.cv_results_)


# 3 hidden layers with 100 units each seem to be the best architecture, giving a test accuracy of 0.82 and not overfitting very much (small difference between train score and test score). We'll keep this architecture and optimize other hyperparameters.

# In[ ]:


parameters = {
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam']
}


# In[ ]:


mlp = MLPClassifier(hidden_layer_sizes=[100, 100, 100])
clf = GridSearchCV(mlp, parameters, scoring='accuracy', return_train_score=True)
clf.fit(X, y)


# In[ ]:


pd.DataFrame(clf.cv_results_)


# The test accuracy is still around 0.82, the best parameters being 'tanh' for the activation function and 'adam' for the solver, which are the default parameters.

# Since there is no significant improvement over the baseline after all this feature encoding and hyperparameter tuning, I will just use the data without any encoding (as I did for the baseline) and see what happens.

# ### Without data encoding

# In[ ]:


X = data.drop('Survived', axis=1).values
y = data['Survived'].values


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
accuracies = dict()
for clf_name in clfs:
    clf = clfs[clf_name]
    clf.fit(X_train, y_train)
    accuracies[clf_name] = clf.score(X_valid, y_valid)


# In[ ]:


accuracies


# The random forest classifier seems to perform best, so I'll do some hyperparameter optimization on it.

# In[ ]:


parameters = {
    'n_estimators': [10, 50, 100, 150, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'bootstrap': [True, False]
}
rfc = RandomForestClassifier()
clf = GridSearchCV(rfc, parameters, scoring='accuracy', return_train_score=True)
clf.fit(X, y)
pd.DataFrame(clf.cv_results_)


# The best accuracy is again around 0.82. Now, I will perform another hyperparameter optimization on random forests but by using data encoding.

# ### Random forest hyperparameter optimization with data encoding

# In[ ]:


X = feature_extraction(data)
y = data['Survived'].values


# In[ ]:


parameters = {
    'n_estimators': [10, 50, 100, 150, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'bootstrap': [True, False]
}
rfc = RandomForestClassifier()
clf = GridSearchCV(rfc, parameters, scoring='accuracy', return_train_score=True)
clf.fit(X, y)
pd.DataFrame(clf.cv_results_)


# Again, 0.82!
# 
# I'll just use the neural network with the hyperparameters tuned before for the final submission.

# ## Feature engineering all over again

# [This article](https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/) opened my eyes to the fact that the names of the passengers are not useless information. On the contrary, they contain the person's title ('Dr', 'Master', 'Capt'...), and we can expect this information to be discriminative with regards to the survival of a person. The article also gives a certain number of feature engineering ideas that I'll be using here.

# I'm going to read the data again and perform a different preprocessing, extracting a person's title from its name and adding a few columns to the data by re-doing what was done in the article previously cited.

# In[ ]:


# re-read the data
data = pd.read_csv('../input/train.csv')
data.head()


# In[ ]:


data.count()


# The missing values for the columns 'Age', 'Cabin', and 'Embarked' will be filled the same way as I did in the beginning.
# 
# I'll work on the 'Name' column now.

# In[ ]:


data['Name'].head(10)


# The 'Name' column seems to follow the following pattern: {last name}, {title}. {first names}
# 
# I'll use this information to extract the title.

# In[ ]:


titles = []
for name in data['Name'].str.lower():
    right_part = name.split(', ')[1]  # right_part = {title}. {first names}
    title = right_part.split('.')[0]
    titles.append(title)
print(len(titles))


# We see that all rows have titles in the 'Name' column.

# In[ ]:


titles = list(set(titles))
print(len(titles))
titles


# I'm going to re-order the titles according to their nobility. This certainly won't be a perfect order since I'm not very familiar with these kind of titles. After the re-ordering, I can transform the 'Name' column into an ordinal column, easier to work with for classification algorithms.

# In[ ]:


titles = ['unk', 'miss', 'mlle', 'mrs', 'mr', 'ms', 'mme', 'lady', 'dr', 'sir', 'master', 'rev', 'major', 'col', 'capt', 'don', 'jonkheer', 'the countess']
len(titles)


# In[ ]:


title_encoder = LabelEncoder()
tmp = title_encoder.fit_transform(titles)
tmp


# I can now write a function that transforms the name column into an integer value.

# In[ ]:


def get_title_from_name(name):
    right_part = name.split(', ')[1]  # right_part = {title}. {first names}
    title = right_part.split('.')[0]
    known_titles = ['unk', 'miss', 'mlle', 'mrs', 'mr', 'ms', 'mme', 'lady', 'dr', 'sir', 'master', 'rev', 'major', 'col', 'capt', 'don', 'jonkheer', 'the countess']
    return title if title in known_titles else 'unk'


# In[ ]:


def set_titles(df):
    titles = df['Name'].str.lower().apply(get_title_from_name)
    titles = np.array(titles)
    df['Title'] = titles


# In[ ]:


set_titles(data)
data.head()


# I'm going to transform the 'Cabin' column into a 'Deck' column, I think that that's where the discriminative information might be.

# In[ ]:


def cabin_to_deck(cabin):
    if cabin == 'UNK':
        return 'UNK'
    return cabin[0]


# I'm going to re-read the data in order to undo the changes above. Then, I'm going to re-define the preprocess() and feature_extraction() functions to implement some feature engineering. After that, I'll test several classifiers; and finally, I'll optimize the hyperparemeters of one of them.

# In[ ]:


data = pd.read_csv('../input/train.csv')


# In[ ]:


encoders, column_values = dict(), dict()


# In[ ]:


def preprocess(df, test=False):
    # fill in the missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Cabin'].fillna('UNK', inplace=True)
    df['Embarked'].fillna('UNK', inplace=True)
    
    # create columns for feature engineering
    # add the 'Title' column
    set_titles(df)
    
    # add a 'FamilySize' column
    df['FamilySize'] = df['SibSp'] + df['Parch']
    
    # turn the cabin number into Deck
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'UNK']
    df['Deck'] = df['Cabin'].apply(cabin_to_deck)
    
    # define the lists of column values for categorical data,
    # and the corresponding encoders (if train data)
    if not test:
        for column in ['Sex', 'Embarked', 'Deck']:
            column_values[column] = list(set(df[column])) + ['other']
            encoders[column] = LabelEncoder()
            encoders[column].fit(column_values[column])
    
    # adjust certain values if test data
    if test:
        for column in ['Sex', 'Embarked', 'Deck']:
            df[column] = df[column].apply(lambda x: x if x in column_values[column] else 'other')
    
    # drop the 'PassengerId', 'Name' and 'Cabin' columns
    df.drop('PassengerId', axis=1, inplace=True)
    df.drop('Name', axis=1, inplace=True)
    df.drop('Ticket', axis=1, inplace=True)
    df.drop('Cabin', axis=1, inplace=True)


# In[ ]:


preprocess(data)


# In[ ]:


def feature_extraction(df):
    # unchanged columns
    X = np.concatenate((df['Age'].values.reshape(-1, 1),
                        df['Pclass'].values.reshape(-1, 1),
                        df['SibSp'].values.reshape(-1, 1),
                        df['Parch'].values.reshape(-1, 1),
                        df['Fare'].values.reshape(-1, 1),
                        df['FamilySize'].values.reshape(-1, 1)),
                        axis=1)
    
    # ordinal encoding of Sex, Embarked and Deck
    for column in ['Sex', 'Embarked', 'Deck']:
        values = encoders[column].transform(df[column].values.reshape(-1, 1))
        values = np.array(values).reshape(-1, 1)
        X = np.concatenate((X, values), axis=1)
    
    # the other columns will not be used
    return X


# In[ ]:


X = feature_extraction(data)
y = data['Survived'].values


# In[ ]:


clfs = {
    'mnb': MultinomialNB(),
    'gnb': GaussianNB(),
    'svm1': SVC(kernel='linear'),
    'svm2': SVC(kernel='rbf'),
    'svm3': SVC(kernel='sigmoid'),
    'mlp1': MLPClassifier(),
    'mlp2': MLPClassifier(hidden_layer_sizes=[100, 100]),
    'ada': AdaBoostClassifier(),
    'dtc': DecisionTreeClassifier(),
    'rfc': RandomForestClassifier(),
    'gbc': GradientBoostingClassifier(),
    'lr': LogisticRegression()
}


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
accuracies = dict()
for clf_name in clfs:
    clf = clfs[clf_name]
    clf.fit(X_train, y_train)
    accuracies[clf_name] = clf.score(X_valid, y_valid)


# In[ ]:


accuracies


# We'll perform hyperparameter optimization on the gradient boosting classifier.

# In[ ]:


parameters = {
    'loss': ['deviance', 'exponential'],
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5],
    'n_estimators': [50, 100, 150, 200, 300],
    'max_depth': [3, 5, 8, 10]
}
gbc = GradientBoostingClassifier()
clf = GridSearchCV(gbc, parameters, scoring='accuracy', return_train_score=True)
clf.fit(X, y)
pd.DataFrame(clf.cv_results_)


# In[ ]:


print('Best score: {0}'.format(clf.best_score_))
print('Best params: {0}'.format(clf.best_params_))


# ## Submission

# So I'll use gradient boosting as a classification algorithm, with the following hyperparameters:
# * learning rate: 0.2
# * loss: exponential
# * max_depth: 3
# * n_estimators: 100
# 
# I'll train the algorithm on the whole provided dataset, then I'll generate the submission file using the provided test data.

# In[ ]:


# read and preprocess the whole train dataset
data = pd.read_csv('../input/train.csv')
encoders, column_values = dict(), dict()
preprocess(data)

# extract the features from the whole training dataset provided
X = feature_extraction(data)

# format the labels in a sutable way for a classifier
y = data['Survived'].values

# train the classifier
clf = GradientBoostingClassifier(learning_rate=0.2, loss='exponential', max_depth=3, n_estimators=100)
clf.fit(X, y)


# In[ ]:


# read the test data
test_data = pd.read_csv('../input/test.csv')

# we'll save the passenger ids because we need them for the submission file
passenger_ids = test_data['PassengerId'].values

# preprocess the test data
preprocess(test_data, test=True)

# extract the features
X_test = feature_extraction(test_data)

# make the predictions
y_pred = clf.predict(X_test)


# In[ ]:


# save the submission in a file
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': y_pred
})
submission.to_csv('submission.csv', index=False)
submission.head()

