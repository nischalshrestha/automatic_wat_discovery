#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing

# Allows paralellization on OS X and Linux via scikit-learn's n_jobs:
multiprocessing.set_start_method('forkserver')

import numpy as np
import pandas as pd


# In[2]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# ## Preprocessing

# The actual cabin isn't particularly useful on its own, but maybe the letter in the cabin identifier is useful.

# In[3]:


def extract_cabin_letter(val):
    if type(val) is str:
        return val[0]
    return "X"

train['Cabin'] = train['Cabin'].apply(extract_cabin_letter)
test['Cabin'] = test['Cabin'].apply(extract_cabin_letter)


# Let's extract each person's title.

# In[4]:


# "Brobeck, Mr. Karl Rudolf" => "Mr."
def extract_title(name):
    given_name = name.split(",")[1]
    title = given_name.strip().split(" ")[0]
    return title

train['Title'] = train['Name'].apply(extract_title)
test['Title'] = test['Name'].apply(extract_title)

all_titles = np.concatenate((train['Title'].unique(), test['Title'].unique()), axis=0)
all_titles = np.unique(all_titles)
all_titles


# Then we'll extract the prefix for each ticket:

# In[6]:


def extract_ticket_prefix(ticket):
    if ' ' not in ticket:
        return 'regular'
    return ticket.split(' ')[0].replace('.', '').replace('/', '').lower()

train['TicketPrefix'] = train['Ticket'].apply(extract_ticket_prefix)
test['TicketPrefix'] = test['Ticket'].apply(extract_ticket_prefix)

ticket_prefixes = np.concatenate((train['TicketPrefix'].unique(), test['TicketPrefix'].unique()), axis=0)
ticket_prefixes = np.unique(ticket_prefixes)
ticket_prefixes


# Then, we binarize nominal features and impute missing values.  The final feature vector is shown below.

# In[74]:


from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Normalizer, MultiLabelBinarizer, Imputer

train['Embarked'].fillna('n/a', inplace=True)
test['Embarked'].fillna('n/a', inplace=True)

mapper = DataFrameMapper([
    ('Sex', LabelEncoder()),
    (['Pclass'], MultiLabelBinarizer()),
    (['Age'], [Imputer(), Normalizer()]),
    ('SibSp', None),
    ('Parch', None),
    (['Fare'], [Imputer(), Normalizer()]),
    (['Cabin'], MultiLabelBinarizer()),
    (['Title'], MultiLabelBinarizer(classes=all_titles)),
    (['Embarked'], MultiLabelBinarizer()),
    (['TicketPrefix'], MultiLabelBinarizer(classes=ticket_prefixes))
])

training_instances =mapper.fit_transform(train)
training_labels = np.array(train['Survived'])
print("X dimensions:")
print(mapper.transformed_names_)


# # Evaluating Classifiers
# Now, we'll try a range of classifiers and evaluate them using 10-fold cross validation.

# In[75]:


from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import cohen_kappa_score

def evaluate(classifier, training_instances, training_labels):
    metrics = cross_validate(classifier, 
                             training_instances, 
                             training_labels, 
                             cv=10, 
                             n_jobs=-1,
                             scoring=['f1', 'accuracy', 'precision', 'recall'])
    print("Accuracy: %0.4f (+/- %0.4f)" % (metrics['test_accuracy'].mean(), metrics['test_accuracy'].std() * 2))
    print("Mean fit time: %0.4f ms" % (metrics['fit_time'].mean()))
    print("F1: %0.4f" % (metrics['test_f1'].mean()))
    print("Precision: %0.4f" % (metrics['test_precision'].mean()))
    print("Recall: %0.4f" % (metrics['test_accuracy'].mean()))
    


# ## Decision tree classifier

# In[76]:


from sklearn import tree
from sklearn.metrics import accuracy_score

dt = tree.DecisionTreeClassifier()
evaluate(dt, training_instances, training_labels)


# ## SVM

# In[77]:


from sklearn import svm

# svc = svm.SVC(C=1.0, kernel='rbf', cache_size=1000)
svc = svm.SVC(C=2.0, kernel='linear', cache_size=1000)
# svc = svm.SVC(C=2.0, kernel='poly', degree=2, cache_size=1000)
evaluate(svc, training_instances, training_labels)


# ## RandomForest

# In[78]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
evaluate(rf, training_instances, training_labels)


# ## Adaboost

# In[79]:


from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=10)
evaluate(ada, training_instances, training_labels)


# ## XGBoost

# In[80]:


from xgboost import XGBClassifier

xg = XGBClassifier()
evaluate(xg, training_instances, training_labels)


# ## Neural network

# In[87]:


from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(hidden_layer_sizes=(), activation='tanh', solver='adam', batch_size=1)
evaluate(nn, training_instances, training_labels)


# ## Naive bayes

# In[82]:


from sklearn.naive_bayes import GaussianNB, BernoulliNB

nb = BernoulliNB(class_prior=[.68, .31])
evaluate(nb, training_instances, training_labels)


# ## Logistic Regression

# In[83]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
evaluate(lr, training_instances, training_labels)


# ## Voting ensemble

# In[88]:


from sklearn.ensemble import VotingClassifier

voting = VotingClassifier([
    ('logistic', LogisticRegression()),
    ('neuralnet', MLPClassifier(hidden_layer_sizes=(), activation='tanh', solver='adam', batch_size=1)),
    ('xgboost', XGBClassifier()),
    ('randomforest', RandomForestClassifier(n_estimators=100)),
    ('ada', AdaBoostClassifier(n_estimators=10)),
    ('svc', svm.SVC(C=2.0, kernel='linear', cache_size=1000))
])
evaluate(voting, training_instances, training_labels)


# ## Generate predictions for test set
# 
# Finally, we'll generate our predictions for the test set.

# In[89]:


# pre-process:

test_instances = mapper.transform(test)

clf = voting
clf.fit(training_instances, training_labels)
predictions = clf.predict(test_instances)

output = pd.concat([test['PassengerId'], pd.Series(predictions)], axis=1, keys=['PassengerId', 'Survived'])
output.set_index('PassengerId', inplace=True)
output.to_csv("predictions.csv")


# In[90]:


output.head()


# In[20]:


train.set_index('PassengerId').drop('Name', 1).to_csv("prepped_train.csv")


# In[ ]:




