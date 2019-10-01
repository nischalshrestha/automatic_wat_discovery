#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Working on the Pandas quickstart tutorial..

# In[ ]:


train_ds= pd.read_csv('../input/train.csv')
print("*"*10 + "USING .tail()" + "*"*10)
print(train_ds.tail(3))
print(train_ds['Name'].tail(9))
print("*"*10 + "USING .index" + "*"*10)
print(train_ds.index)
print("*"*10 + "USING .columns" + "*"*10)
print(train_ds.columns)
print("*"*10 + "USING .values" + "*"*10)
print(train_ds.values)
print("*"*10 + "USING .describe()" + "*"*10)
print(train_ds.describe())


# In[ ]:


print(train_ds.columns.values)


# In[ ]:


train_ds.sort_index(axis= 1, ascending= False)


# In[ ]:


train_ds[9:12]['Name']


# In[ ]:


train_ds.loc[22:29, ['PassengerId', 'Name', 'Age']]


# In[ ]:


train_ds.loc[3:5, :]


# In[ ]:


train_ds.iloc[3:5, :]


# In[ ]:


train_ds.iloc[[4, 2, 90], [2, 4, 5]]


# In[ ]:


train_ds[train_ds.Cabin == 'C123']


# In[ ]:


train_ds[train_ds['Cabin'].isin(['C123', 'D17'])]


# *..Pandas quickstart 10 minute tutorial completed.*

# ## Now off with the dataset on my own..

# In[ ]:


train_ds= pd.read_csv('../input/train.csv')
test_ds= pd.read_csv('../input/test.csv')


# In[ ]:


print(train_ds.info())


# In[ ]:


train_ds.head()


# In[ ]:


print('*' * 10 + '#NaN in Train data' + '*' * 10)
print(len(train_ds) - train_ds.count())
print('*' * 10 + '#NaN in Test data' + '*' * 10)
print(len(test_ds) - test_ds.count())


# **621/891 cabin values are NaN! Also ticket name seems to be useless.  
# Therefore, i'm dropping these..**

# In[ ]:


train_ds.drop('Cabin', axis= 1, inplace= True)
test_ds.drop('Cabin', axis= 1, inplace= True)
train_ds.drop('Ticket', axis= 1, inplace= True)
test_ds.drop('Ticket', axis= 1, inplace= True)


# In[ ]:


train_ds.columns.values


# In[ ]:


train_ds.describe()


# min Fare == 0.00 => *Some people travelled with zero fare on the Titanic!*  
# *Lets look into that..*

# In[ ]:


print("Exactly this no. of passengers travelled with 0 fare:")
print(len(train_ds[train_ds.Fare == 0]))
print("Details of those passengers:")
train_ds[train_ds.Fare == 0]


# In[ ]:


train_ds_0fare= train_ds[train_ds.Fare == 0]
len(train_ds_0fare[train_ds_0fare.Survived==1])


# _Hmm.. interesting._   
# **Most of them failed to survive (only 1/15 did!).**

# In[ ]:


train_ds_0fare[['PassengerId', 'Sex']].groupby(['Sex']).count()


# Also, **all of them are male.**

# In[ ]:


train_ds_0fare[['Pclass', 'PassengerId']].groupby('Pclass').count()


# Also, **these people are spread across all Pclasses (*5 in Pclass(1), 6 in Pclass(2) and 4 in Pclass(3)*).**

# 

# Maybe it will be beneficial if we captured this into a new feature.  
# 
# **For this purpose, I am creating a new column- _FreeTraveller_  where,  
# 1 => Fare ==  0 for that traveller  
# 0 => Fare != 0 for that traveller**

# In[ ]:


train_ds.loc[(train_ds['Fare']==0), 'FreeTraveller']= 1
train_ds['FreeTraveller'].fillna(0, inplace= True)
test_ds.loc[(test_ds['Fare']==0), 'FreeTraveller']= 1
test_ds['FreeTraveller'].fillna(0, inplace= True)


# *Insight- The training accuracies have __failed to improve__ upon the addition of this feature. Upon training with this new feature, no model shows any substantial improvement as compared to before this feature was added. In fact, some accuracies have dropped (though only by a small amount).*

# ### Starting off with the visualisation..
# #### Using seaborn for visualisation
# 
# Following a [tutorial](https://elitedatascience.com/python-seaborn-tutorial) from elitedatascience.com

# In[ ]:


import seaborn as sns


# In[ ]:


sns.lmplot(x= 'Age', y= 'Fare', data= train_ds, fit_reg= False, hue= 'Survived')


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


sns.set_style('whitegrid')
sns.lmplot(x= 'Age', y= 'Fare', data= train_ds, fit_reg= False, hue= 'Survived')
plt.ylim(0, None)
plt.xlim(0, None)


# In[ ]:


train_ds.columns.values


# In[ ]:


sns.swarmplot(x= 'SibSp', y= 'Fare', data= train_ds)
plt.title("Crappy thing to do")


# In[ ]:


g= sns.FacetGrid(train_ds, col= 'Survived')
g.map(plt.hist, 'Age', bins= 10)


# In[ ]:


grid = sns.FacetGrid(train_ds, col='Embarked', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Survived', alpha= .5, ci= None)
grid.add_legend()


# Made the first pass through seaborn. Seems simple to use.

# Lets have a look at the data one more time.

# In[ ]:


train_ds.head(8)


# **Let's combine SibSp and Parch into one feature- _FamilySize_**

# In[ ]:


train_ds['FamilySize']= train_ds['Parch'] + train_ds['SibSp'] + 1
test_ds['FamilySize']= test_ds['Parch'] + test_ds['SibSp'] + 1
train_ds.drop(['SibSp', 'Parch'], axis= 1, inplace= True)
test_ds.drop(['SibSp', 'Parch'], axis= 1, inplace= True)


# In[ ]:


train_ds[['FamilySize', 'Survived']].groupby('FamilySize').mean().sort_values(by= 'Survived', ascending= False)


# ### Now changing the non-numeric data (_Sex, Embarked_) to numeric and completing incomplete fields..

# **SEX:**

# In[ ]:


train_ds['Sex']= train_ds['Sex'].map({'male': 0, 'female': 1}).astype(int)
test_ds['Sex']= test_ds['Sex'].map({'male': 0, 'female': 1}).astype(int)


# In[ ]:


len(train_ds) - train_ds.count()


# In[ ]:


len(test_ds) - test_ds.count()


# **AGE:**

# In[ ]:


mean_age= {'train': np.zeros((2, 3)), 'test': np.zeros((2, 3))} # no. of sexes X no. of Pclasses
for sex in range(0, 2):
    for pcl in range(1, 4):
        mean_age['train'][sex][pcl-1]= train_ds[(train_ds['Pclass']==pcl) &                                      (train_ds['Sex']==sex)]['Age'].mean()
        mean_age['test'][sex][pcl-1]= test_ds[(test_ds['Pclass']==pcl) &                                      (test_ds['Sex']==sex)]['Age'].mean()


# In[ ]:


for sex in range(0, 2):
    for pcl in range(1, 4):
        train_ds.loc[(train_ds['Age'].isnull()) & (train_ds['Sex'] == sex) &                      (train_ds['Pclass'] == pcl), 'Age']= mean_age['train'][sex][pcl-1]
        test_ds.loc[(test_ds['Age'].isnull()) & (test_ds['Sex'] == sex) &                      (test_ds['Pclass'] == pcl), 'Age']= mean_age['test'][sex][pcl-1]


# **FARE:**

# In[ ]:


test_ds['Fare'].fillna(test_ds['Fare'].mean(), inplace= True)


# **EMBARKED:**

# In[ ]:


train_ds['Embarked'].fillna((train_ds['Embarked'].value_counts()).index[0], inplace= True)


# In[ ]:


train_ds['Embarked']= train_ds['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)
test_ds['Embarked']= test_ds['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)


# **NAME:**
# 
# Dropping the name.

# In[ ]:


train_ds.drop(['Name'], axis= 1, inplace= True)
test_ds.drop(['Name'], axis= 1, inplace= True)


# **PASSENGERID:**
# 
# Dropping it from train data set only as we need it in the test dataset when submitting. 

# In[ ]:


train_ds.drop(['PassengerId'], axis= 1, inplace= True)


# *__Let's take one final look at the data..__*

# In[ ]:


print('Train data:')
print('#incomplete rows')
print(len(train_ds) - train_ds.count())
print(train_ds.columns.values)
print(train_ds.head(3))
print('*'* 0)
print('Test data:')
print('#incomplete rows')
print(len(test_ds) - test_ds.count())
print(test_ds.columns.values)
print(test_ds.head(3))


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


X_train= train_ds.drop('Survived', axis= 1)
Y_train= train_ds['Survived']
X_test= test_ds.drop('PassengerId', axis= 1)


# In[ ]:


logreg= LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred= logreg.predict(X_test)
acc_log= round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:


perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_ds["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)

