#!/usr/bin/env python
# coding: utf-8

# # Titanic Classification

# ### The following notebook explains in brief how to train a simple machine learning model to tackle the Titanic classification challenge on Kaggle. 
# 
# #### This is my first Kaggle challenge attempt and the work here is much inspired from already existing kernels on Kaggle.

# ### Importing the necessary libraries:

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# ### Importing the dataset:

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
data = pd.concat([train,test],axis=0,sort=False)
data.head()


# In[ ]:


train.info()


#  Let us know the likelihood of survival of passengers based on some of their data. The float values represent the likelihood of their survival.

# In[ ]:


train[['Pclass','Survived']].groupby(train['Pclass']).mean()


# Above, we get Class wise likelihood of Survival. As we can see, the passengers with class 1 tickets are more likely to survive than the remaining two.

# In[ ]:


train[['Sex','Survived']].groupby(train['Sex']).mean()


# Similarly, we see gender wise likelihood of Survival.

# There are two columns named number of siblings/spouses and parents/children. We can use that data and create a new column named Family size and then group them to check the likelihood of survival based on their family sizes:

# In[ ]:


for row in train:
    train['Family size'] = train['SibSp'] + train['Parch'] + 1

for row in test:
    test['Family size'] = test['SibSp'] + test['Parch'] + 1
train[['Family size','Survived']].groupby(train['Family size']).mean()


# Let us check how being a lone passenger contributes to survival. Below we can see that lone passengers are less likely to survive.

# In[ ]:


for row in train:
        train['isAlone'] = 0
        train.loc[train['Family size']==1, 'isAlone'] = 1

for row in test:
        test['isAlone'] = 0
        test.loc[test['Family size']==1, 'isAlone'] = 1
        
train[['isAlone','Survived']].groupby(train['isAlone']).mean()


# The Embarked column records data of port of Embarkation of the passengers. Since there are some missing values in the Embarked column, we fill those missing values by mode ie. most occuring value ie. 'S'.

# In[ ]:


data['Embarked'].groupby(data['Embarked']).count()


# In[ ]:


train['Embarked'] = train['Embarked'].fillna('S')
train[['Embarked','Survived']].groupby(train['Embarked']).mean()

test['Embarked'] = test['Embarked'].fillna('S')
train[['Embarked','Survived']].groupby(train['Embarked']).mean()


# Let us impute the missing values of Fare details by the median of the records and then categorize the fare details into 10 ranges. 

# In[ ]:


train['Fare'] = train['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'],4)

test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['CategoricalFare'] = pd.qcut(test['Fare'],4)
train[['CategoricalFare','Survived']].groupby(train['CategoricalFare']).mean()


# Now let us deal with the Age values:

# In[ ]:


data.info()


# As one can notice, out of total 891, only 714 values of age are recorded while the remaining data is missing. Since the 177 records whose ages are missing still might represent crucial data about other attributes, we simply cannot discard those records. A better method to deal with them is by replacing them by random numbers among the upper and lower range of standard deviation from the mean values of age. Let us do that:

# In[ ]:


for row in train:
    avg = train['Age'].mean()
    std = train['Age'].std()
    null_count = train['Age'].isnull().sum()
    age_null_random_list = np.random.randint(avg - std, avg + std, size=null_count)
    train['Age'][np.isnan(train['Age'])] = age_null_random_list
    train['Age'] = train['Age'].astype(int)

for row in test:
    avg = test['Age'].mean()
    std = test['Age'].std()
    null_count = test['Age'].isnull().sum()
    age_null_random_list = np.random.randint(avg - std, avg + std, size=null_count)
    test['Age'][np.isnan(test['Age'])] = age_null_random_list
    test['Age'] = test['Age'].astype(int)


# In[ ]:


train['CategoricalAge'] = pd.cut(train['Age'],5)

test['CategoricalAge'] = pd.cut(test['Age'],5)
train[['CategoricalAge','Survived']].groupby(train['CategoricalAge']).mean()


# Now that we have imputed some missing values, let us handle the categorical values and handle the ranged values

# In[ ]:


# Mapping Fare
train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2
train.loc[ train['Fare'] > 31, 'Fare'] = 3

train.loc[ train['Age'] <= 16, 'Age'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age'] = 4


test.loc[ test['Fare'] <= 7.91, 'Fare'] = 0
test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1
test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare']   = 2
test.loc[ test['Fare'] > 31, 'Fare'] = 3

test.loc[ test['Age'] <= 16, 'Age'] = 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3
test.loc[ test['Age'] > 64, 'Age'] = 4


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


train.head()


# In[ ]:


train['Sex'] = train['Sex'].map({'female':0,'male':1})
test['Sex'] = test['Sex'].map({'female':0,'male':1})
train['Embarked'] = train['Embarked'].map({'S':0,'C':1,'Q':2})
test['Embarked'] = test['Embarked'].map({'S':0,'C':1,'Q':2})


# Now that all our feature engineering and data cleaning is done, let us select the attributes from the dataset and perform visualization.
# 
# But before that, let us get rid of unnecessary columns. We can get rid of columns such as Passenger Names, the Ticket nos., Cabin records, SibSp & Parch(Since we already used them in Family size), CategoricalFare and CategoricalAge.

# In[ ]:


train = train.drop(['Name','SibSp','Parch','Ticket','Cabin','CategoricalFare','CategoricalAge'],axis = 1)
test = test.drop(['Name','SibSp','Parch','Ticket','Cabin','CategoricalFare','CategoricalAge'], axis = 1)


# In[ ]:


sns.pairplot(train,hue='Survived')


# Sex wise distribution of survival.
# 
# Female --> 0
# Male --> 1

# In[ ]:


sns.countplot(x=train['Sex'],data=train,hue="Survived",orient='v')


# Sex wise distribution of Age.
# 
# Female --> 0
# Male --> 1

# In[ ]:


plt.figure(figsize=(12,10))
sns.countplot(y=train['Age'],data=train,hue='Sex')


# The records representing Embarkation and the Class of the passengers

# In[ ]:


plt.figure(figsize=(15,15))
sns.jointplot(train['Embarked'],train['Pclass'],kind="kde")


# Class wise Fare distribution which tells the relativity among the two attributes:

# In[ ]:


plt.figure(figsize=(12,10))
sns.countplot(x=train['Pclass'],data=train,hue="Fare") 


# In[ ]:


explode =(0,0.05,0.05,0.05)
plt.figure(figsize=(10,10))
plt.pie(train['Fare'].groupby(train['Fare']).sum(),labels=['Category 0','Category 1','Category 2','Category 3'],
        colors=['gold','#e33d3d','#33d9ed','#7ae10c'],
        explode=explode,shadow=True,autopct='%1.1f%%')
plt.show()


# ### Classification Modelling

# We will be creating our classification model on several different classification algorithms including some Ensembling models such as Random Forests, Adaboost, Gradient boosting etc.
# 
# Refer to the links below to know more about the different classification models:
# <pre>
# <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">K nearest Neighbors</a>
# <a href="https://en.wikipedia.org/wiki/Support_vector_machine">Support Vector Machines</a>
# <a href="https://en.wikipedia.org/wiki/Decision_tree">Decision Trees</a>
# <a href="https://en.wikipedia.org/wiki/Random_forest">Random Forests</a>
# <a href="https://en.wikipedia.org/wiki/AdaBoost">Adaboost</a>
# <a href="https://en.wikipedia.org/wiki/Gradient_boosting">Gradient boosting</a>
# <a href="https://en.wikipedia.org/wiki/Naive_Bayes_classifier">Naive Bayes Classification</a>
# <a href="https://en.wikipedia.org/wiki/Logistic_regression">Logistic Regression</a>
# </pre>

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

x = train.iloc[:,[0,2,3,4,5,6,7,8]].values
y = train.iloc[:,1].values

classifiers = [
    KNeighborsClassifier(4),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LogisticRegression()]

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

chart = pd.DataFrame(columns=["Classifier", "Accuracy"])

acc_dict = {}

for train_index, test_index in sss.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(x_train, y_train)
    train_predictions = clf.predict(x_test)
    acc = accuracy_score(y_test, train_predictions)
    if name in acc_dict:
        acc_dict[name] += acc
    else:
        acc_dict[name] = acc

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=["Classifier", "Accuracy"])
    chart = chart.append(log_entry)


# What we have done here is fit the training data to all the models and mapped their accuracies into a bar chart. The plot quite delineates which model is able to classify the Survival of the passengers better.

# In[ ]:


plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=chart, palette="Blues")


# In[ ]:


chart['Accuracy']= chart['Accuracy']*1000
chart


# * As one can notice, the Logistic Regression classifier shows the maximum accuracy in classifying the survival of passengers, hence we will use it to predict our values.

# In[ ]:


final_classifier = LogisticRegression()
final_classifier.fit(train.iloc[:,[0,2,3,4,5,6,7,8]].values,train.iloc[:,1].values)
result = final_classifier.predict(test)


# In[ ]:


result = DataFrame(result)
result


# Therefore, we conclude the titanic classification modelling using Python. The better classifier was Logistic Regression, whereas the algorithm to perform poorly on the data was Support Vector Classifier.  

#   

# ## Thanks for Reading.
