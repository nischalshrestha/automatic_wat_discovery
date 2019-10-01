#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Libraries for analysing data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import random as rnd

# Libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# Definitions
pd.set_option('display.float_format', lambda x: '%.0f' % x)
get_ipython().magic(u'matplotlib inline')

# Libraries for machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

train_df = pd.read_csv('../input/train.csv', header=0)
test_df = pd.read_csv('../input/test.csv', header=0)


# In[ ]:


train_df.head(5)


# In[ ]:


train_df.info()
# Age and Embarked have NAn's


# In[ ]:


#VISUALIZING THE DATA 
# Visualizing the data helps to undercover underlying patterns
sns.barplot(x = "Embarked" , y = "Survived", hue= "Sex" , data = train_df)


# In[ ]:


sns.barplot(x= "Pclass" , y = "Survived" , hue = "Embarked" , data = train_df)


# TRANSFORMING FEATURES
# 
# 1.  Age is cleaned and divided into logical human age groups, making it easier to plot
# 2. For the "Cabin" feature,  the first letter is extracted for analysing survival rate and the rest is deleted.
# 3. Fare is divided into quartile ranges for easy handling.
# 4. For "Name" feature the last name and the prefix was extracted.
# 5. All unwanted features are dropped.
# 

# In[ ]:


def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = [1, 8, 7, 6, 5, 3, 2 ,4]
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    
    
def drop_features(df):
    return df.drop(['Ticket', 'Name','Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

train_df = transform_features(train_df)
test_df = transform_features(test_df)
train_df.head()


# In[ ]:


sns.barplot(x="Age", y="Survived", hue="Sex", data=train_df)


# In[ ]:


sns.barplot(x="Cabin", y="Survived", hue="Sex", data=train_df)


# In[ ]:


sns.barplot(x="Fare", y="Survived", hue="Sex", data=train_df)


# **Final Encoding:**
# 
# This step normalizes labels , which converts unique string values to numbers, making data more flexible
# for algorithms. 

# In[ ]:


train_df = train_df.replace({"Sex": { "female" : 2, "male" : 1} })
test_df = test_df.replace({"Sex": { "female" : 2, "male" : 1} })
train_df.head(2)


# In[ ]:


from sklearn import preprocessing
def encode_features(train_df, test_df):
    features = ['Fare', 'Cabin', 'Age', 'Lname', 'NamePrefix']
    df_combined = pd.concat([train_df[features], test_df[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        train_df[feature] = le.transform(train_df[feature])
        test_df[feature] = le.transform(test_df[feature])
    return train_df, test_df
    
train_df, test_df = encode_features(train_df, test_df)
train_df.head()


# **Splitting up the Training Data:**
# 
# First, separate the features(X) from the labels(y).
# X_all: All features minus the value we want to predict (Survived).
# y_all: Only the value we want to predict.
# Second, use Scikit-learn to randomly shuffle this data into four variables. In this case, I'm training 80% of the data, then testing against the other 20%.
# 

# In[ ]:


from sklearn.model_selection import train_test_split

X_all = train_df.drop(['Survived', 'PassengerId'], axis=1)
y_all = train_df['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)


# **Machine Learning Model**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)



# In[ ]:


predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))


# **Validate with KFold**
# 
# Is this model actually any good? It helps to verify the effectiveness of the algorithm using KFold. This will split our data into 10 buckets, then run the algorithm using a different bucket as the test set for each iteration.

# In[ ]:


from sklearn.cross_validation import KFold

def run_kfold(clf):
    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))     
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome)) 

run_kfold(clf)


# **Predict the Actual Test Data**
# 
# 
# And now for the moment of truth. Make the predictions, export the CSV file, and upload them to Kaggle.

# In[ ]:


ids = test_df['PassengerId']
predictions = clf.predict(test_df.drop('PassengerId', axis=1))


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictions.csv', index = False)
output.head()

