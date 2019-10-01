#!/usr/bin/env python
# coding: utf-8

# **Importing libriaries and loading data**

# In[44]:


# data analysis and wrangling
import pandas as pd
import numpy as np

# visualisation
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[45]:


# read in the files provided by Kaggle
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# **Exploring dataset**

# In[46]:


train_df.columns


# In[47]:


train_df.head()


# In[48]:


train_df.info()


# In[49]:


test_df.info()


# **Features in the dataset:**
# * categorical
#     * nominal: Survived, Sex, Embarked
#     * ordinal: Pclass
# * numerical
#    * continuous: Age, Fare. Discrete: SibSp, Parch
# * mixed data types
#    * alphanumeric: Ticket, Cabin

# In[50]:


# check floats and ints stats
train_df.describe()


# In[51]:


# check objects stats
train_df.describe(include=['O'])


# **Pre-analysis**

# First of all we can drop 'PassengerId feature', because it was given for creating the table and cannot correlate with the survival feature. We are also going to drop the alphanumeric features like 'Ticket' or 'Cabin'. The ticket number seem like a quite unique feature, so it would be probably hard to get any relevant correlation from it.  The problem with cabin is that, we would have to analyse the construction of the ship and even then we can't be sure that everybody were in their cabins during the crash. Probably some cabins we situated more conveniently, so it should be easier to reach the lifeboats, but we can assume that the 'Class' or 'Fare' is going to reflect it. Passenger's names also seems hard to analyse, as they are all unique. One thing we can do is trying to extract the person's title from it and then checking for correlation, but I'm going to skip it.

# ***'PClass'***
# We can see some relation - passengers from 1st class have higher survival ratio. For further analysis we should try to get rid of order between values, because it doesn't reflect the relationship properly.

# In[52]:


# check 'Pclass' feature
class_pivot = train_df.pivot_table(index="Pclass",values="Survived")
class_pivot.plot.bar()
plt.show()


# ***'Sex' ***
# Again, we see an abvious relationship, that females have higher survival ratio. We are going to need to convert it into numerical value.

# In[53]:


sex_pivot = train_df.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()


# ***'Age'***
# There is no obvious relationship between 'Age' and 'Survived' columns, but one may assume that the age was a relevant feature, so it would be good to try to divide age into ranges and then look for the correlations. Plus there are some Nan values, which have to be adressed (we have 714 out of 891 records in the train set).

# In[54]:


survived = train_df[train_df["Survived"] == 1]
died = train_df[train_df["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='green',bins=50)
died["Age"].plot.hist(alpha=0.5,color='red',bins=50)
plt.legend(['Survived','Died'])
plt.show()


# ***'SibSp' and 'Parch'***
# There is a strong variety in those features, we could try to combine 'SibSp' and 'Parch' into 'isAlone' feature to check if people without a family are more willing to survive the crash.

# In[55]:


sib_pivot = train_df.pivot_table(index="SibSp",values="Survived")
sib_pivot.plot.bar()
plt.show()


# In[56]:


parch_pivot = train_df.pivot_table(index="Parch",values="Survived")
parch_pivot.plot.bar()
plt.show()


# ***'Fare'***
# It is not obvious that the feature differs among those who survived and not, but we could try to divide it into bands and then check again for correlation.

# In[57]:


survived = train_df[train_df["Survived"] == 1]
died = train_df[train_df["Survived"] == 0]
survived["Fare"].plot.hist(alpha=0.5,color='green',bins=50)
died["Fare"].plot.hist(alpha=0.5,color='red',bins=50)
plt.legend(['Survived','Died'])
plt.show()


# ***'Embarked'***
# Cherbourg port has the highest ratio of survival, the other two seem more similar. It's worth including it in the model after converting to numerical value.

# In[58]:


embark_pivot = train_df.pivot_table(index="Embarked",values="Survived")
embark_pivot.plot.bar()
plt.show()


# We are looking for the features, which may best correlate with Survived value. Conclusions:
# - get rid of:  'PassengerId', 'Ticket', 'Cabin'
# - to correct: 'Pclass', 'Sex', 'Age', SibSp', 'Parch', Fare', 'Embarked'

# **Cleaning dataset and preparing for machine learning**

# ***Dividing data into ranges for 'Age' and  'Fare'.*** We are going to divide those numerical continuous features into categorical ones, but first we need to fill all the Nan in the 'Age' column.

# In[59]:


def divide_age(df, cut_points, label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"], cut_points, labels=label_names)
    return df

# declare ranges and their labels
age_cut_points = [-1,0,3,12,18,35,60,100]
age_label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

# change both train and test datasets
train_df = divide_age(train_df, age_cut_points, age_label_names)
test_df = divide_age(test_df, age_cut_points, age_label_names)

#plot the results
age_pivot = train_df.pivot_table(index="Age_categories",values='Survived')
age_pivot.plot.bar()
plt.show()


# In[60]:


def divide_fare(df, cut_points, label_names):
    df["Fare_categories"] = pd.cut(df["Fare"], cut_points, labels=label_names)
    return df

# declare ranges and their labels
fare_cut_points = [0,25,50,75,100,520]
fare_label_names = ["Cheap","Low","Medium","High","Premium"]

# change both train and test datasets
train_df = divide_fare(train_df, fare_cut_points, fare_label_names)
test_df = divide_fare(test_df, fare_cut_points, fare_label_names)

#plot the results
fares_pivot = train_df.pivot_table(index="Fare_categories",values='Survived')
fares_pivot.plot.bar()
plt.show()


# ***Convert into one feature: 'SibSp' and 'Parch'***
# We may assume, that when a person was travelling alone he or she has a better chance of surviving the catastrophy, because it was easier to get on a lifeboat. After converting the features we see the contrary - people who were travelling with their families had a higher survival ratio. And it's not because they stand for the majority of the cruise population - only 40% of the passengers travelled with families.

# In[61]:


def is_alone(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    return df

# change both train and test datasets
train_df = is_alone(train_df)
test_df = is_alone(test_df)

#plot the results
is_alone = train_df.pivot_table(index="IsAlone",values='Survived')
is_alone.plot.bar()
plt.show()


# In[62]:


train_df['IsAlone'].value_counts(normalize=True)


# ***Creating dummies for: 'Pclass', 'Sex', 'Embarked', 'Age_categories', 'Fare_categories'***
# 

# In[63]:


def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df,dummies], axis=1)
    return df

for column in ["Pclass","Sex","Embarked", "Age_categories", "Fare_categories"]:
    train_df = create_dummies(train_df, column)
    test_df = create_dummies(test_df, column)


# In[64]:


test_df.head()


# **Splitting the dataset**

# In[65]:


# change the 'real' test data to holdout
holdout = test_df

# select the columns we are interested about making predictions
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'Age_categories_Missing','Age_categories_Infant','Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult', 'Age_categories_Senior', 'Fare_categories_Cheap',
          'Fare_categories_Low', 'Fare_categories_Medium', 'Fare_categories_High', 'Fare_categories_Premium']

# chooose xs and y
all_x = train_df[columns]
all_y = train_df['Survived']

train_x, test_x, train_y, test_y = train_test_split(all_x, all_y, test_size=0.20,random_state=0)


# ** Trying algorithms**

# In[70]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(train_x, train_y)
y_pred = logreg.predict(test_x)
acc_log = round(logreg.score(train_x, train_y) * 100, 2)
print(acc_log)


# In[71]:


# Support Vector Machines
svc = SVC()
svc.fit(train_x, train_y)
y_pred = svc.predict(test_x)
acc_svc = round(svc.score(train_x, train_y) * 100, 2)
print(acc_svc)


# In[72]:


# K-nearest neighbours
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_x, train_y)
y_pred = knn.predict(test_x)
acc_knn = round(knn.score(train_x, train_y) * 100, 2)
print(acc_knn)


# In[73]:


# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(train_x, train_y)
y_pred = gaussian.predict(test_x)
acc_gaussian = round(gaussian.score(train_x, train_y) * 100, 2)
print(acc_gaussian)


# In[74]:


# Perceptron
perceptron = Perceptron()
perceptron.fit(train_x, train_y)
y_pred = perceptron.predict(test_x)
acc_perceptron = round(perceptron.score(train_x, train_y) * 100, 2)
print(acc_perceptron)


# In[75]:


# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(train_x, train_y)
y_pred = linear_svc.predict(test_x)
acc_linear_svc = round(linear_svc.score(train_x, train_y) * 100, 2)
print(acc_linear_svc)


# In[76]:


# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(train_x, train_y)
y_pred = sgd.predict(test_x)
acc_sgd = round(sgd.score(train_x, train_y) * 100, 2)
print(acc_sgd)


# In[77]:


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_x, train_y)
y_pred = decision_tree.predict(test_x)
acc_decision_tree = round(decision_tree.score(train_x, train_y) * 100, 2)
print(acc_decision_tree)


# In[78]:


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_x, train_y)
y_pred = random_forest.predict(test_x)
acc_random_forest = round(random_forest.score(train_x, train_y) * 100, 2)
print(acc_random_forest)


# **Cross-validation**

# In[79]:


names = ['Logistic Regression', 'Support Vector Machines', 'KNN', ' Gaussian Naive Bayes', 'Perceptron', 'Linear SVC', 
         'Stochastic Gradient Decent', 'Decision Tree', 'Random Forest']

estimators = [logreg, svc, knn, gaussian, perceptron, linear_svc, sgd, decision_tree, random_forest]

scores = [acc_log, acc_svc, acc_knn, acc_gaussian, acc_perceptron, acc_linear_svc, acc_sgd, acc_decision_tree, 
          acc_random_forest]

models_summary = pd.DataFrame({'Name':names, 'Score':scores})


# In[82]:


test_list = []
for est in estimators:
    accuracy = cross_val_score(est, all_x, all_y, cv=10).mean()
    test_list.append(accuracy)
print(test_list)

models_summary.loc[:, 'Accuracy'] = pd.Series(test_list, index=models_summary.index)
models_summary.sort_values(by='Accuracy', ascending=False)


# **Predicting on unseen data and creating a submission file**

# In[85]:


svc.fit(all_x,all_y)
holdout_predictions = svc.predict(holdout[columns])


# In[84]:


holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)
submission.to_csv("submission4.csv",index=False)

