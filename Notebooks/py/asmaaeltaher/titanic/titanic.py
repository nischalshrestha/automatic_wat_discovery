#!/usr/bin/env python
# coding: utf-8

# # Steps:
# 1. Important imports
# 2. Load & Prepare data
#         2.1 Analyze data
#         2.2 Analyze fuatures
#         2.3 Clean features
#         2.4 Final encoding
#         2.5 Split data
# 3. Select Model
# 4. Validation
# 5. Predict the Actual Test Data

# # 1. Important imports

# In[ ]:


import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# # 2. Load & Prepare data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ## 2.1 Analyze data

# In[ ]:


train.head()


# In[ ]:


train.info()
print('_'*40)
test.info()


# In[ ]:


train.describe()


# ## 2.2 Analyze fuatures

# In[ ]:


sns.barplot(x='Embarked', y='Survived', hue='Sex', data= train)


# In[ ]:


sns.pointplot(x='Pclass', y='Survived', hue='Sex', data= train)


# ## 2.3 Clean features
# 1. Grouping people into logical human age groups.
# 2. Extract Cabin's letter.
# 3. Fare placed into quartile bins accordingly.
# 4. Extract the name prefix (Mr. Mrs. Etc.).
# 5. Lastly, drop (Ticket and Name)

# In[ ]:


def simplify_age(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabin(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fare(df):
    df.Fare=df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', 'Q1', 'Q2', 'Q3', 'Q4']
    categories = pd.cut(df.Fare, bins, labels = group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df

def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def clean_data(df):
    df = simplify_age(df)
    df = simplify_cabin(df)
    df = simplify_fare(df)
    df = format_name(df)
    df = drop_features(df)
    return df

train = clean_data(train)
test = clean_data(test)
train.head()


# In[ ]:


sns.barplot(x='Age', y='Survived', hue='Sex', data= train)


# In[ ]:


sns.barplot(x='Cabin', y='Survived', hue='Sex', data= train)


# In[ ]:


sns.barplot(x='Fare', y='Survived', hue='Sex', data= train)


# ## 2.4 Final encoding

# In[ ]:


from sklearn import preprocessing
def encode(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'NamePrefix']
    df_all = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_all[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

train, test =encode(train, test)
train.head()


# ## 2.5 Split data

# In[ ]:


from sklearn.model_selection import train_test_split

X = train.drop(['Survived', 'PassengerId'], axis=1)
y = train['Survived']
train_x, test_x, train_y, test_y= train_test_split(X, y, test_size=0.20, random_state=23)


# # 3. Select Model

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.metrics import accuracy_score


# ## 3.1 Random Forest Classifier

# In[ ]:


model_1 =  RandomForestClassifier()
model_1.fit(train_x,train_y)
predictions_1 = model_1.predict(test_x)
print(accuracy_score(test_y, predictions_1))


# ## 3.2 Support Vector Machines

# In[ ]:


model_2 =  SVC()
model_2.fit(train_x,train_y)
predictions_2 = model_2.predict(test_x)
print(accuracy_score(test_y, predictions_2))


# ## 3.3 Gradient Boosting Classifier

# In[ ]:


model_3 =  GradientBoostingClassifier()
model_3.fit(train_x,train_y)
predictions_3 = model_3.predict(test_x)
print(accuracy_score(test_y, predictions_3))


# ## 3.4 K-nearest neighbors

# In[ ]:


model_4 =  KNeighborsClassifier(n_neighbors=3)
model_4.fit(train_x,train_y)
predictions_4 = model_4.predict(test_x)
print(accuracy_score(test_y, predictions_4))


# ## 3.5 Gaussian Naive Bayes

# In[ ]:


model_5 = GaussianNB()
model_5.fit(train_x,train_y)
predictions_5 = model_5.predict(test_x)
print(accuracy_score(test_y, predictions_5))


# ## 3.6 Logistic Regression

# In[ ]:


model_6 = LogisticRegression()
model_6.fit(train_x,train_y)
predictions_6 = model_6.predict(test_x)
print(accuracy_score(test_y, predictions_6))


# # 4. Validate with KFold

# In[ ]:


from sklearn.cross_validation import KFold

def run_kfold(clf):
    kf = KFold(891, n_folds=10)
    outcome = []
    fold=0
    for train_index, test_index in kf:
        fold += 1
        train_x, test_x = X.values[train_index], X.values[test_index]
        train_y, test_y = y.values[train_index], y.values[test_index]
        clf.fit(train_x,train_y)
        predictions= clf.predict(test_x)
        accuracy = accuracy_score(test_y, predictions)
        outcome.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
    mean_outcome = np.mean(outcome)
    return print("Mean Accuracy: {0}".format(mean_outcome))

run_kfold(model_1)
print('_'*20)
run_kfold(model_3)


# # 5. Predict the Actual Test Data

# In[ ]:


ids = test['PassengerId']

preds_3 = model_3.predict(test.drop(['PassengerId'], axis=1)
output=pd.DataFrame({'PassengerId' : ids, 'Survived': preds_3})
test.to_csv( 'titanic_preds_3.csv' , index = False )

preds_1 = model_1.predict(test.drop['PassengerId'], axis=1)
output=pd.DataFrame({'PassengerId' : ids, 'Survived': preds_1})
test.to_csv( 'titanic_preds_1.csv' , index = False )


# In[ ]:




