#!/usr/bin/env python
# coding: utf-8

# # Introduction

# I am here trying to find out how many people on titanic survived from disaster.
# Here goes the flow:
# 
# 1) **Introduction**
# 
#  1. Import Libraries
#  2. Load data
#  3. Run Statistical summeries
#  4. Figure out missing value columns
# 
#  
#  
# 2) **Visualizations**
# 
#  1. Correlation with target variable
# 
# 
# 3) **Missing values imputation**
# 
#  1. train data Missing columns- Embarked,Age,Cabin
#  2. test data Missing columns- Age and Fare
#  
# 
# 4) **Feature Engineering**
# 
#  1. Categorize the age of passengers
#  2. Simplify the cabins
#  3. Simplify and categorize the fares
# 
# 
# 5) **Prediction**
# 
#  1. Split into training & test sets
#  2. Build the model
#  3. Feature importance
#  4. Predictions
#  5. Ensemble : Majority voting
# 
# 6) **Submission**

# # Import libraries

# In[16]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# # Load train & test data

# In[18]:


data = pd.read_csv("../input/train.csv")
data.head()


# In[19]:


test_data = pd.read_csv("../input/test.csv")
test_data.head()


# In[20]:


data.shape


# In[21]:


data.describe()


# In[22]:


data.info()


# In[23]:


null_columns = data.columns[data.isnull().any()]
null_columns


# In[24]:


data.isnull().sum()


# ## Visualizations

# In[25]:


data.hist(bins=10,figsize=(10,10),grid=False)


# In[26]:


sns.barplot(x="Embarked", y="Survived", hue="Pclass", data=data);


# In[27]:


sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data);


# In[28]:


sns.pointplot(x="Pclass", y="Survived",hue="Sex", data=data, palette={"male":"blue", "female":"pink"}, markers=["*", "o"], 
             linestyles=["-", "--"])


# ## Correlation

# In[29]:


data.corr()


# In[30]:


data.corr()["Survived"]


# In[31]:


data[data["Embarked"].isnull()]


# ## Feature Engineering

# In[32]:


def categorize_Age(data):
    data.Age = data.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(data.Age, bins, labels=group_names)
    data.Age = categories
    return data

def simplify_cabin(data):
    data.Cabin = data.Cabin.fillna("N")
    data.Cabin = data.Cabin.apply(lambda x: x[0])
    return data

def simplify_fares(data):
    data.Fare = data.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(data.Fare, bins, labels=group_names)
    data.Fare = categories
    return data

def drop_features(data):
    return data.drop(["Ticket", "Name", "Embarked"], axis=1)

def transform_features(data):
    data = categorize_Age(data)
    data = simplify_cabin(data)
    data = simplify_fares(data)
    data = drop_features(data)
    return data

data = transform_features(data)


# In[33]:


test_data = transform_features(test_data)


# In[34]:


data.head()


# In[35]:


from sklearn import preprocessing

def encode_features(data, test_data):
    features = ['Fare', 'Cabin', 'Age', 'Sex']
    data_combined = pd.concat([data[features], test_data[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(data_combined[feature])
        data[feature] = le.transform(data[feature])
        test_data[feature] = le.transform(test_data[feature])
    return data, test_data

data, test_data = encode_features(data, test_data)


# In[36]:


data.head()


# In[37]:


from sklearn.model_selection import train_test_split

X = data.drop(['PassengerId', 'Survived'], axis=1)
Y = data['Survived']

trainx, testx, trainy, testy = train_test_split(X, Y, test_size=0.2, random_state=0)

trainx.head()


# # Predict Survival

# ## Random Forest

# In[38]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier()
parameters = {'n_estimators': [2,5,10],
             'criterion': ['gini', 'entropy'],
             'max_features': ['log2', 'sqrt', 'auto'],
             'max_depth': [2,3,5,10],
             'min_samples_split': [2,3,5],
             'min_samples_leaf': [1,5,8]}

acc_scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf, parameters, scoring = acc_scorer)
grid_obj = grid_obj.fit(trainx, trainy)

clf = grid_obj.best_estimator_
clf.fit(trainx, trainy)


# In[39]:


predictions = clf.predict(testx)
print(accuracy_score(testy, predictions))


# In[40]:


from sklearn.cross_validation import KFold

N = 891
def run_kfold(clf):
    kf = KFold(N, n_folds = 10)
    accuracies = []
    fold = 0
    for x,y in kf:
        trainx, testx = X.iloc[x], X.iloc[y]
        trainy, testy = Y.iloc[x], Y.iloc[y]
        clf.fit(trainx, trainy)
        pred = clf.predict(testx)
        accuracies.append(accuracy_score(testy, pred))
    mean_accuracy = np.mean(accuracies)
    print(mean_accuracy)

run_kfold(clf)


# In[41]:


ids = test_data['PassengerId']
pred = clf.predict(test_data.drop('PassengerId', axis = 1))
output1= pd.DataFrame({"PassengerId": ids,
                       "Survived": pred
})


# In[42]:


output1.head()


# ## Logistic regression

# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

lr = LogisticRegression(random_state=None)

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)

scores = cross_val_score(lr, trainx, trainy, scoring='f1', cv=cv)

print(scores.mean())


# In[44]:


run_kfold(lr)


# ## Linear Regression

# In[45]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()

def run_kfold_1(clf):
    kf = KFold(N, n_folds = 10)
    accuracies = []
    fold = 0
    for x,y in kf:
        trainx, testx = X.iloc[x], X.iloc[y]
        trainy, testy = Y.iloc[x], Y.iloc[y]
        clf.fit(trainx, trainy)
        pred = clf.predict(testx)
        pred[pred > .5] = 1
        pred[pred <=.5] = 0
        accuracies.append(accuracy_score(testy, pred))
    mean_accuracy = np.mean(accuracies)
    print(mean_accuracy)

run_kfold_1(model)


# ## SVM

# In[48]:


from sklearn.svm import SVC

clf1 = SVC()

parameters = {'C': [1,2,5,10],
             'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
             }

acc_scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf1, parameters, scoring = acc_scorer)
grid_obj = grid_obj.fit(trainx, trainy)

clf1 = grid_obj.best_estimator_
clf1


# In[49]:



cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=None)

scores = cross_val_score(clf1, trainx, trainy, scoring='f1', cv=cv)

print(scores.mean())


# In[50]:


run_kfold(clf1)


# ## Submission

# In[51]:


pred = clf1.predict(test_data.drop("PassengerId", axis=1))


# In[52]:


submission= pd.DataFrame({"PassengerId": test_data["PassengerId"],
                         "Survived": pred
                         })


# In[53]:


submission.head()


# In[54]:


submission.to_csv("titanic_submission.csv", index=False)

