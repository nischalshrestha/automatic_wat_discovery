#!/usr/bin/env python
# coding: utf-8

# In[415]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

random_seed = 42

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.base import BaseEstimator, TransformerMixin

import os
print(os.listdir("../input"))


# In[416]:


# from https://stackoverflow.com/a/47167330
# To Impute Categorical Variables
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean',filler='NA'):
        self.strategy = strategy
        self.fill = filler

    def fit(self, X, y=None):
        if self.strategy in ['mean','median']:
            if not all(X.dtypes == np.number):
                raise ValueError('dtypes mismatch np.number dtype is                                  required for '+ self.strategy)
        if self.strategy == 'mean':
            self.fill = X.mean()
        elif self.strategy == 'median':
            self.fill = X.median()
        elif self.strategy == 'mode':
            self.fill = X.mode().iloc[0]
        elif self.strategy == 'fill':
            if type(self.fill) is list and type(X) is pd.DataFrame:
                self.fill = dict([(cname, v) for cname,v in zip(X.columns, self.fill)])
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

def impute_and_to_dataframe(imputer, df, fit_end_index, cols):
    #Fit only on train set
    train = df[:fit_end_index][cols]
    imputer.fit(train)
    #Fill up full set
    imputed = imputer.transform(df[cols])
    return pd.DataFrame(imputed, columns=cols)

#TODO should you always drop_first ?
def columns_into_encoded(df, columns, nofirst=True):
    result = df.copy()
    for c in columns:
        dummies = pd.get_dummies(full[c], prefix=c, drop_first=nofirst)
        result.drop(c, axis=1, inplace=True)
        result = pd.concat([result, dummies], axis = 1)
    return result

def plot_correlation(corr):
    plt.figure(figsize=(20,20))
    sns.heatmap(corr, cmap="RdBu", square=True, vmin=-1.0, vmax=1.0, annot=True, annot_kws = {"fontsize" : 14})
    
def fill_null_with_mean(df):
    for c in df.columns:
        mean = df.loc[:,c].mean()
        x = df.loc[:,c].fillna(mean)
        df.loc[:,c] = x

def fill_null_with_mode(df):
    for c in df.columns:
        mode = df.loc[:,c].mode()[0]
        df.loc[:,c] = df.loc[:,c].fillna(mode)
        
def convert_columns_to_categorical(df, columns):
    for c in df.columns:
        df.loc[:,c] = df.loc[:,c].astype("category")

def feature_ranking(X, y):
    clf = ExtraTreesClassifier()
    clf.fit(X, y)
    importances = clf.feature_importances_
    std = np.std([clf.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("{}. Feature {} ({}) Std Dev: {}".format(f + 1, X.columns[indices[f]], importances[indices[f]], std[indices[f]]))
        
def plot_features(df, features, target=None, plot="barplot", size=(20,20), ncols=2, xtickrot=90):
    cols = ncols
    rows = np.ceil(len(features)/cols)
    
    if plot != "distplot" and target == None:
        raise ValueError("Target Required")
    
    fig = plt.figure(figsize=size)
    for i, feature in enumerate(features):
        fig.add_subplot(rows, cols, i+1)
        plt.xticks(rotation=xtickrot)
        if plot == "barplot":
            sns.barplot(x=feature, y=target, data=df)
        elif plot == "distplot":
            sns.distplot(df[feature].dropna())
        else:
            raise ValueError("Unknown Plot type")
        
def print_scores(models, X_train, y_train, X_valid, y_valid):
    for model in models:
        print("Model: ", model, "Train Score ", model.score(X_train, y_train), "Test Score", model.score(X_valid, y_valid))
        
def print_cv_scores(models, X_train_valid, y_train_valid, cv=10):
    for model in models:
        cv_results = cross_val_score(model, X_train_valid, y_train_valid)
        print("Model: ", model)
        print("Mean Score: ", format(cv_results.mean()))
        print("Median Score: ", format(np.median(cv_results)))
        print("Std Score: ", format(cv_results.std()))
        print("=======================================")

#from https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial
def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

#TODO: Plotting Feature Importances Function
#clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
#clf = clf.fit(train, targets)
#features = pd.DataFrame()
#features['feature'] = train.columns
#features['importance'] = clf.feature_importances_
#features.sort_values(by=['importance'], ascending=True, inplace=True)
#features.set_index('feature', inplace=True)
#features.plot(kind='barh', figsize=(25, 25))


# In[417]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

full = train.append(test, ignore_index=True)
train.shape

titanic = full[:train.shape[0]]

del train, test
print("Datasets:", "full:", full.shape, "titanic:", titanic.shape)


# # Feature Selection and Exploratory Data Analysis

# In[418]:


titanic.describe()


# In[419]:


corr = titanic.corr()
plot_correlation(corr)


# In[420]:


non_numeric_candidates = ["Cabin", "Embarked", "Sex"]
plot_features(titanic, ["Embarked", "Sex", "Pclass", "SibSp", "Parch"],  "Survived")


# In[421]:


numeric_candidates = ["Age", "Fare", "Parch", "Pclass", "SibSp"]
plot_features(titanic,numeric_candidates ,plot="distplot")


# In[422]:


from pandas.plotting import scatter_matrix
scatter_matrix(titanic[numeric_candidates], figsize=(20,20))


# # Feature Engineering

# In[91]:


full.head()


# We arrive at a better way to extract titles:

# In[424]:


def extract_title(name):
    if(name.find(".") < 0):
        return "Unknown"
    half = name.split(".")[0]
    return half.split(" ")[-1] + "."

full["Title"] = full.Name.apply(extract_title)
full.Title.value_counts()


# Processing the ages:
# We look into the training set only to retrieve  median ages for different combinations of Sex, Title and Pclass. This way we get a more accurate way to fill NaN age Values

# In[425]:


median_ages = full[:titanic.shape[0]].groupby(["Sex", "Pclass", "Title"]).median().reset_index()
median_ages = median_ages[["Sex", "Pclass", "Title", "Age"]]

overall_median = full[:titanic.shape[0]]["Age"].median()


# In[426]:


def fill_age(row):
    #Only change NaN values
    if not np.isnan(row["Age"]):
        return row["Age"]
    c = (
        (median_ages['Sex'] == row['Sex']) & 
        (median_ages['Title'] == row['Title']) & 
        (median_ages['Pclass'] == row['Pclass'])
    )
    # if we don't have a median age for this combination, use the overall value
    if(len(median_ages[c].index) == 0):
        return overall_median
    return median_ages[c]['Age'].values[0]

full["Age"] = full.apply(fill_age, axis=1)


# In[427]:


sns.distplot(full["Age"])


# Turn Titles into encoded values and drop names

# In[428]:


full.drop("Name", axis=1, inplace=True)
full = columns_into_encoded(full, ["Title"])


# Fill Missing "Embarked" values with the most frequent value (mode) and encode them

# In[429]:


full["Embarked"] = impute_and_to_dataframe(CustomImputer(strategy="mode"), full, titanic.shape[0],["Embarked"])


# In[430]:


full = columns_into_encoded(full, ["Embarked"])


# Encode "Sex" 

# In[431]:


full = columns_into_encoded(full, ["Sex"])


# Fill Missing Fares with mean

# In[432]:


full["Fare"] = impute_and_to_dataframe(Imputer(), full, titanic.shape[0], ["Fare"])


# In[433]:


full.info()


# Find all possible Cabin Types

# In[434]:


full[full.Cabin.notnull()]["Cabin"].apply(lambda x : x[0]).value_counts()


# In[435]:


full.loc[full.Cabin.notnull(),"Cabin"] = full[full.Cabin.notnull()]["Cabin"].apply(lambda x : x[0])


# In[436]:


imputer = CustomImputer(strategy="fill", filler="U")


# In[437]:


full["Cabin"] = impute_and_to_dataframe(imputer, full, titanic.shape[0], ["Cabin"])


# In[438]:


full = columns_into_encoded(full, ["Cabin"])


# In[439]:


full.info()


# Examine the different Types of Tickets

# In[440]:


full.Ticket.value_counts()


# In[441]:


import re

def transform_tickets(ticket):
    #remove special characters and leading and trailing whitespace
    ticket = re.sub("[^0-9a-zA-Z\s]+", "", ticket)
    ticket = ticket.strip()

    #if only numeric, return 
    if re.search("[^0-9\s]", ticket) == None:
        return "NUMERICAL"
    return ticket.split(" ")[0].upper()


# In[442]:


full["Ticket"] = full.Ticket.apply(transform_tickets)


# In[443]:


full = columns_into_encoded(full, ["Ticket"])


# Handle Family sizes
# TODO: 
# * Research distribution of average family sizes in 1912

# In[444]:


families = pd.DataFrame()
families["FamilySize"] = full["Parch"] + full["SibSp"] + 1

families.describe()


# In[445]:


families.FamilySize.value_counts()


# In[446]:


full["FamilySize"] = families["FamilySize"]


# In[447]:


full.info()


# # Modelling

# ## Create the Datasets
# Split the full dataframe down into train and test again

# In[452]:


X_full = full.copy()
#Drop target features and features that should not be used
X_full.drop(["Survived"], axis=1, inplace=True)
X_train_valid = X_full[:titanic.shape[0]]
y_train_valid = titanic["Survived"]
X_test = X_full[titanic.shape[0]:]


# In[453]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.3, random_state=random_seed)


# In[454]:


feature_ranking(X_train, y_train)


# ## Train the Models

# In[455]:


models = []


# In[456]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

models.append(knn)


# In[457]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)

models.append(knn)


# In[458]:


svc = SVC()
svc.fit(X_train, y_train)
models.append(svc)


# In[459]:


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
models.append(rfc)


# In[460]:


print_cv_scores(models, X_train_valid, y_train_valid, cv=10)


# ## Perform Parameter Tuning with candidate model

# In[461]:


parameters = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }

rfc_cv = GridSearchCV(rfc, scoring="accuracy", param_grid=parameters, cv=10, verbose=1)
#TODO uncomment to perform search
#rfc_cv.fit(X_train_valid, y_train_valid)


# In[462]:


best_params = {'bootstrap': True,
 'max_depth': 8,
 'max_features': 'auto',
 'min_samples_leaf': 3,
 'min_samples_split': 3,
 'n_estimators': 50}


# In[463]:


model = RandomForestClassifier(**best_params)
model.fit(X_train_valid, y_train_valid)


# In[464]:


y_pred = model.predict(X_test).astype(int)
ids = X_test.PassengerId
predictions = pd.DataFrame({"PassengerId" : ids, "Survived" : y_pred})
predictions.to_csv("titanic_predictions.csv", index=False)


# In[ ]:




