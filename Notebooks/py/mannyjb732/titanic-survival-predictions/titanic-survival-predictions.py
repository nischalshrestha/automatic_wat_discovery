#!/usr/bin/env python
# coding: utf-8

# <h1>Predicting Titanic Survival Rate</h1>

# In[ ]:


#import packages

import pandas as pd 
import numpy as np 
import seaborn as sn
import matplotlib.pyplot as plt 
get_ipython().magic(u'matplotlib inline')


# Using the Test and Train import the data into using the pandas package. 

# In[ ]:


#import test and train data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# Below we see the our features and data types.  From here we will determine which feautres to use. For our predictions we will not be using Ticket and the Cabin feature.   The column "Name" we will parse out the title as it is going to be used in our predictions to determine survival rate.  

# In[ ]:


train.info()


# Below we see the columns that contains nulls Embarked and Age are the features that we are going to predict the null values to fill the NA. We are going to drop the cabin feature as it contains a lot of null values.   

# In[ ]:


train.isnull().sum()


# Lets visualize the null values on by plotting the data set into a heatmap.

# In[ ]:


plt.figure(figsize=(12,8))
sn.heatmap(train.isnull(), yticklabels=False, cbar=False)
plt.show()


# Lets look at the Statistics of the Titanic Data.   

# In[ ]:


train.describe()


# <h1>Data Visualization</h1>

# Below we can see the corralation between features.   The most corrolated features are Parch and Sibsp, Sex and Survived are Corrolated.  The dark blue shows the fields that have strong corralation.

# In[ ]:



corr = train.corr()
plt.figure(figsize=(13.5,10))
sn.heatmap(corr, annot=True, cmap='seismic_r', linewidths=.5)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sn.countplot('Survived', data=train, palette='RdBu_r')
plt.title('Count of Passenger by Survived')
plt.show()


# In[ ]:


grid = sn.FacetGrid(train, row='Pclass', col='Sex', hue='Sex', size=3.5, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()


# In[ ]:


plt.figure(figsize=(13.5,7))
sn.boxplot('Embarked', 'Fare', data = train)
plt.title('Embarked by Fare')
plt.show()


# In[ ]:


plt.figure(figsize=(13.5, 7))
sn.boxplot('Embarked', 'Age', data=train)
plt.title('Emabrked by Age')
plt.show()


# In[ ]:


grid = sn.FacetGrid(train, row='Sex', col='Survived', hue='Sex', size=3.5, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins = 20)
grid.add_legend()
plt.show()


# <h1>Preparing the Data</h1>

# Lets remove the features we don't need for our predictions.  In this case we are removing the Ticket and Cabin 

# Combined both Train and Test to perform several procedures in the data in order to prepare the data for predictions models.  For example we will be extracting the title from the name feature.  We will be filling the NA values of the Embarked features and the Age feature.  In order for the predictin models to make predictions accurately we will convert the Features into numeric values.   

# In[ ]:


combined = [train, test]


# In[ ]:


for dataset in combined :
    dataset['Embarked'] = dataset['Embarked'].fillna('C')


# In[ ]:


for dataset in combined: 
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# Below we see the titles there several titles that are rare.  Given in the time that Titanic sank rare titles were common given the Era.  We will be convertity those rare titles into a bucket called rare, We know that the following titles Mlle, Ms, and Mme are todays title of Miss and Mrs.  

# In[ ]:


for dataset in combined:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[ ]:


for dataset in combined: 
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)


# In[ ]:


title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
for dataset in combined: 
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

embark_map = {'S': 1, 'Q': 2, 'C': 3}
for dataset in combined:
    dataset['Embarked'] = dataset['Embarked'].map(embark_map)


# In[ ]:


train[['Title', 'Sex']].groupby('Title').count()


# Below we see a Frequency Distribution plot that contains Age values by Pclass and Sex. We can see from Pclass 3 we have the majority of the Males From looking at the Distribution plot I am able to determine that males have a less suvival rate due to the higher male population. 

# From the boxplot we can see by Embarked coe if there are outliers with the fare.  We can quickly determined that embarked Code C has a Fare that is greater than 500.  One Passenger paid 500 dollars.  THe average of fare lies between $5 and $90.  

# In[ ]:



sn.factorplot('Title', data=train, kind='count', size=5, aspect=2)
plt.title('Title Distribution')
plt.show()


# In[ ]:


train.head()


# In[ ]:


test.head()


# Lets guess the age values.  We know that we have 172 null values in the age features therefore we will loop through the feature and guess the age values.  The two features we will be looking at to loop through and guess the age values are Age and Pclass.  There are several ways in doing this, but I am using the a for loop to guess these values.  

# Lets convert hour Sex feature into a number value for example Female = 1 and Male = 0 This will help us guess our age values for our Age feature

# In[ ]:


guess_ages = np.zeros((2,3))

guess_ages

for dataset in combined: 
    for i in range(0,2):
        for j in range(0,3):
            guess_df = dataset[(dataset['Sex']==i) & (dataset['Pclass']==j+1)]['Age'].dropna()
            
            age_guess = guess_df.median() 
            
            guess_ages[i,j] = int(age_guess/0.5 + 0.5)*0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)
        


# In[ ]:


train = train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1) 

test['Fare'] = test['Fare'].fillna(0)


# In[ ]:


train.isnull().sum()


# Now lets view our data to see how it looks. 
# 

# In[ ]:


grid = sn.FacetGrid(train, hue='Sex', size=6.5, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


test.head()


# In[ ]:


#Manchine Learning Models 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


x_train = train.drop('Survived', axis=1)
y_train = train['Survived']

x_test = test.drop('PassengerId', axis=1).copy()


# <h1>Logistic Regression</h1>

# In[ ]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
Y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
acc_log


# In[ ]:


coeff_df = pd.DataFrame(train.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# <h1>Support Vector Machines</h1>

# In[ ]:


svc = SVC()
svc.fit(x_train, y_train)
Y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)
acc_svc


# <h1>Neighbor Classfiers</h1>

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
Y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)
acc_knn


# <h1>Gaussian Naive Bayes</h1>

# In[ ]:


gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
Y_pred = gaussian.predict(x_test)
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)
acc_gaussian


# <h1>Perceptron</h1>

# In[ ]:


perceptron = Perceptron()
perceptron.fit(x_train, y_train)
Y_pred = perceptron.predict(x_test)
acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)
acc_perceptron


# <h1>Linear SVC</h1>

# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
Y_pred = linear_svc.predict(x_test)
acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)
acc_linear_svc


# <h1>Stohastic Gradient Descent</h1>

# In[ ]:


sgd = SGDClassifier()
sgd.fit(x_train, y_train)
Y_pred = sgd.predict(x_test)
acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)
acc_sgd


# <h1>Decision Tree</h1> 

# In[ ]:


#Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
Y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree


# <h1>Random Forrest</h1>

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
Y_pred = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
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


# From the above Model and score We see that Random Forrest and Decision Tree have the highest score.  This means that this two models have the high prediction accuracy.   Below I will use the predictions from Random Forrest.   

# In[ ]:


submission = pd.DataFrame({"Passenger": test["PassengerId"], "Survived": Y_pred})


# In[ ]:




