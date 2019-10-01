#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import re
from sklearn import linear_model
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif


# In[ ]:


# Methods to do some work

# A function to get the title from a name.
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# A function to get the title from a name.
def get_last_name(name):
    title_search = re.search('([A-Z])\w+', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(0)
    return np.NaN

def get_family_id(row, map):
    last_name = get_last_name(row['Name'])
    if last_name not in map:
        if len(map) == 0:
            current_id = 1
        else:
            current_id = len(map) + 1
        map[last_name] = current_id
    return ""

def set_family_id(row, map):
    familyId = -1
    if row["FamilySize"] > 0:
        familyId = map[get_last_name(row['Name'])]
    return familyId

def get_cabin_id(row, map):
    last_name = row['Cabin']
    if last_name not in map:
        if len(map) == 0:
            current_id = 1
        else:
            current_id = len(map) + 1
        map[last_name] = current_id
    return ""

def get_ticket_prefix(string):
    parts = string.split()
    if len(parts) == 1:
        return np.NaN
    elif len(parts) == 2:
        return parts[0].replace(".", "")
    else:
        return np.NaN

def get_ticket_prefix_id(row, map):
    prefix = get_ticket_prefix(row['Ticket'])
    if prefix not in map:
        if len(map) == 0:
            current_id = 1
        else:
            current_id = len(map) + 1
        map[prefix] = current_id
    return ""

def get_digits_only(name):
    title_search = re.search('\d+$', name)
    # If the title exists, extract and return it.
    if title_search:
        return int(title_search.group(0))
    return np.NaN

def skipbig(n):
    if n > 1000000:
        return -1
    else:
        return n

def prepare(train, family_id_mapping, get_ticket_prefix_id_mapping, cabin_id_mapping):
    # The titanic variable is available here.
    train["Age"] = train["Age"].fillna(train["Age"].median())
    train["Fare"] = train["Fare"].fillna(train["Fare"].median())
    train.loc[train["Sex"] == "male", "Sex"] = 0
    train.loc[train["Sex"] == "female", "Sex"] = 1
    train["Embarked"] = train["Embarked"].fillna("S")
    train.loc[train["Embarked"] == "S", "Embarked"] = 0
    train.loc[train["Embarked"] == "C", "Embarked"] = 1
    train.loc[train["Embarked"] == "Q", "Embarked"] = 2
    train["SibSp"] = train["SibSp"].fillna(0)
    train["Parch"] = train["Parch"].fillna(0)
    # Generating a familysize column
    train["FamilySize"] = train["SibSp"] + train["Parch"]
    # The .apply method generates a new series
    train["NameLength"] = train["Name"].apply(lambda x: len(x))
    # Get all the titles and print how often each one occurs.
    titles = train["Name"].apply(get_title)

    # Map each title to an integer.  Some titles are very rare, and are compressed into the same 
    # codes as other titles.
    title_mapping = {
        "Mr": 1,
        "Miss": 2,
        "Mrs": 3,
        "Master": 4,
        "Dr": 5,
        "Rev": 6,
        "Major": 7,
        "Col": 8,
        "Mlle": 9,
        "Mme": 10,
        "Don": 11,
        "Lady": 12,
        "Countess": 13,
        "Jonkheer": 14,
        "Sir": 15,
        "Capt": 16,
        "Ms": 17,
        "Dona": 18
    }
    for k, v in title_mapping.items():
        titles[titles == k] = v

    # Add in the title column.
    train["Title"] = titles

    train["FamilyId"] = train.apply(lambda row: set_family_id(row, family_id_mapping), axis=1)
    train["TicketPrefix"] = train.apply(lambda row: get_ticket_prefix_id_mapping[get_ticket_prefix(row['Ticket'])], axis=1)
    train["CabinN"] = train.apply(lambda row: cabin_id_mapping[row['Cabin']], axis=1)


    train["Ticket_s"] = train['Ticket'].apply(get_digits_only).apply(skipbig)
    train["Ticket_s"] = train["Ticket_s"].fillna(0)

    return ""


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    print("Plot train scores " + str(train_scores_mean))
    print("Plot test scores " + str(test_scores_mean))
    return plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
# Family distribution
family_id_mapping = {}
train.apply(lambda row: get_family_id(row, family_id_mapping), axis=1)
test.apply(lambda row: get_family_id(row, family_id_mapping), axis=1);
# Ticket distribution
get_ticket_prefix_id_mapping = {}
train.apply(lambda row: get_ticket_prefix_id(row, get_ticket_prefix_id_mapping), axis=1)
test.apply(lambda row: get_ticket_prefix_id(row, get_ticket_prefix_id_mapping), axis=1);
# cabin distribution
cabin_id_mapping = {}
train.apply(lambda row: get_cabin_id(row, cabin_id_mapping), axis=1)
test.apply(lambda row: get_cabin_id(row, cabin_id_mapping), axis=1);


# In[ ]:


prepare(train, family_id_mapping, get_ticket_prefix_id_mapping, cabin_id_mapping);


# In[ ]:


# TODO tickets

random_state = 170
KMF = ['Ticket_s', 'Pclass', 'CabinN']
kmx = train[KMF]

km_model = KMeans(n_clusters=2, random_state=random_state).fit(kmx)
#print(km_model.score(kmx))
y_pred = km_model.predict(kmx)

plt.figure(figsize=(5, 5))
plt.scatter(train[['Ticket_s']], train[['Pclass']], c=y_pred)
plt.title("Anisotropicly Distributed Blobs")
plt.show()   

train['Ticket_s_g'] = y_pred;


# In[ ]:


predictors = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch', 'FamilySize', 
              'NameLength', 'Title', 'FamilyId', 'CabinN']


# In[ ]:


# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(train[predictors], train["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier

polynomial_features = PolynomialFeatures(degree=1, include_bias=False)
alg = linear_model.LogisticRegression()
#alg = AdaBoostClassifier()
#alg = RandomForestClassifier(n_estimators=300)
#alg = SVC()
#alg = MLPClassifier(hidden_layer_sizes=(24,24,24))

pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("logistic_regression", alg)])
scores = cross_val_score(
    pipeline,
    train[predictors],
    train["Survived"],
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
)
print(scores)
print(scores.mean())


# In[ ]:


from sklearn.metrics import confusion_matrix
# Train the algorithm using all the training data

alg.fit(train[predictors], train["Survived"])
cnf_matrix = confusion_matrix(train["Survived"], alg.predict(train[predictors]))
plot_confusion_matrix(cnf_matrix, classes=[], title='Confusion matrix, without normalization')
plt.show()


# In[ ]:


plot_learning_curve(pipeline, "sdf", train[predictors], train["Survived"], (-0.1, 1.1), cv=3, 
                    n_jobs=1)
plt.show()


# In[ ]:


#Optimal number of features and visualize this
if False:
    rfecv_X = train[predictors]
    rfecv_Y = train["Survived"]
    rfecv = RFECV( estimator = alg , step = 1 , cv = StratifiedKFold( rfecv_Y , 2 ), 
                  scoring = 'accuracy' )
    rfecv.fit( rfecv_X , rfecv_Y )
    plt.figure()
    plt.xlabel( "Number of features selected" )
    plt.ylabel( "Cross validation score (nb of correct classifications)" )
    plt.plot( range( 1 , len( rfecv.grid_scores_ ) + 1 ) , rfecv.grid_scores_ )
    plt.show()


# In[ ]:


#TEST
prepare(test, family_id_mapping, get_ticket_prefix_id_mapping, cabin_id_mapping)
test['Ticket_s_g'] = km_model.predict(test[KMF])


# In[ ]:


# Make predictions using the test set.
predictions = alg.predict(test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
submission.to_csv('titanic.csv', index=False)


# In[ ]:




