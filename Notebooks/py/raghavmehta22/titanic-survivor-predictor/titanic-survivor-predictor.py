#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Contents
# 
# 1. Import Necessary Libraries
# 2. Read In and Explore the Data
# 3. Data Analysis
# 4. Data Visualization
# 5. Cleaning Data
# 6. Choosing the Best Model
# 7. Creating Submission File
# 

# In[ ]:


#data analysis libraries 
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
import warnings
warnings.filterwarnings('ignore')


# Now that we have all of the libraries in place, let's take a look at our data. We will first import it and read it with the commands: 'pd.csv_read' and 'describe()'

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#Looking at the closer details
train.describe()


# # Data Analysis
#  We can take a look at the graph as a whole to see the collumns and some variables with the 'head()' functions, or we can use the '.collumns' function to see the collumns on the screen

# In[ ]:


print(train.columns)


# In[ ]:


#To see it on a graph
train.sample(20)


# Let's take a look at how our values were stored shall we? Pandas has a simple built in function, 'dtypes', allowing us to very easily look at the data types.

# In[ ]:


print(train.dtypes)


# Let's see if we have any missing values in any of the columns

# In[ ]:


print(train.isnull().sum())


# It seems like we have many missing values from our DataSet.
# 16.38% of the age column is missing.
# A whooping 70% of the cabin values are missing.
# Only 0.22% of the embarked column is missing which shouldn't hurt our graph too much. Age factor is important for the survival rate, so we should try and fill in those values as much as we can. Cabin values are mostly missing so dropping the table may be the wisest option. We can however figure that higher "fares" would equal to a higher cabin, therefore making a dependence graph can still give us an idea. It is not necessary but could be certainly interesting.
# 
# 

# **Time for predictions!**
# Sex: Since people have the conception of "Women and children first", it's likely that more women survived than men.
# Age: Just thinking rationally would let us see that people who were younger (Not so young that they have to be carried by someone of course) than the most were likely to survive more aswell.
# pclass: This is an interesting one. Higher fares may have let people get cabins from a higher part of the Titanic, which would've allowed easier acces to lifeboats.

# # Graphing
# 
# If you were bored from all the numbers and tables, this part may be more fun for you.

# In[ ]:


sns.barplot(data= train, x="Sex", y="Survived")
print("Percentage of females surviving:", train["Survived"][train.Sex == "female"].value_counts(normalize=True)*100)
print("Number males surviving:", train["Survived"][train.Sex == "male"].value_counts(normalize = True)*100)


# **Age Factor:**

# In[ ]:


train["Age"] = train["Age"].fillna(-0.5)
test["Age"]= test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60,np.inf]
labels = ["Unknown","Small children","Children","Teens","Adolescent","Young Adults","Adults","Elders"]
train["AgeGroups"] = pd.cut(train.Age,labels=labels, bins=bins)
test["AgeGroups"] = pd.cut(test.Age,labels=labels, bins=bins)
sns.barplot(data = train,x= "AgeGroups",y= "Survived")


# **pclass Factor**

# In[ ]:


sns.barplot(data= train, x="Pclass", y="Survived")
print("Percentage of High earners surviving:", train["Survived"][train.Pclass == 1].value_counts(normalize=True)*100)
print("Percentage of middle class earners surviving:", train["Survived"][train.Pclass == 2].value_counts(normalize = True)*100)
print("Percentage of low class earners surviving:", train["Survived"][train.Pclass == 3].value_counts(normalize = True)*100)


# **Family factor**

# In[ ]:


#Family factors
sns.barplot(data =train, x="SibSp",y="Survived")
print("Survivors with one sibling/Spouse:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)
print("Survivors with two sibling/Spouse:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)
print("survivors with three sibling/Spouse:", train["Survived"][train["SibSp"] == 3].value_counts(normalize = True)[1]*100)
print("Survivors with four sibling/Spouse:", train["Survived"][train["SibSp"] == 4].value_counts(normalize = True)[1]*100)


# As more siblings or spouses were present, the probability of survival was also lower. However it is interesting to note that people with 1 sibling/spouse was mos likely to survive is a very interesting that people with no siblings or sposes were less likely to survive than the ones who had one or two.

# **#Parch Factor**

# In[ ]:



sns.barplot(data = train,x="Parch",y="Survived")


# Cabin Feature Cabin features are a little bit tricky. They don't really mean anything, unless we assume that the ones recorded were people that were more important, or of a higher economic class.

# In[ ]:


train["CabinBool"] = (train.Cabin.notnull().astype("int"))
test["CabinBool"] =  (train.Cabin.notnull().astype("int"))
print("Percentage of recorded Cabins that survived:",train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of recorded Cabins that didn't survive:",train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)
sns.barplot(data = train,x="CabinBool",y="Survived")


# 

# # Time to clean up data

# In[ ]:


test.describe(include="all")


# In[ ]:


train = train.drop(["Ticket"],axis = 1)
test = test.drop(['Ticket'], axis = 1)

train = train.drop(["Cabin"],axis = 1)
test = test.drop(["Cabin"],axis = 1)

train = train.drop(["CabinBool"],axis = 1)
test = test.drop(["CabinBool"],axis = 1)


# **EMBARKED**

# In[ ]:


print("Number of people embarking in Southampton (S):",train[train["Embarked"] == "S"].shape[0])

print("Number of people embarking in Cherbourg (C):",train[train["Embarked"] == "C"].shape[0])

print("Number of people embarking in Queenstown (Q):",train[train["Embarked"] == "Q"].shape[0])


# We see that most embarked from Southhampton, so we can fill in the empty values with "S"

# In[ ]:


#replacing the missing values in the Embarked feature with S
train = train.fillna({"Embarked": "S"})


# In[ ]:


#create a combined group of both datasets
combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])


# In[ ]:


#Replace the titles with numbers that would let out computer understand
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


title_map = {"Master":1,"Miss":2,"Mr":3,"Mrs":4,"Rare":5,"Royal":5}
for dataset in combine:
    dataset["Title"] = dataset["Title"].map(title_map)
    dataset["Title"] = dataset["Title"].fillna(0)

train.head()


# In[ ]:


Master_age = train[train["Title"]==1]["AgeGroups"].mode()#Small children (Somehow, don't ask me)
Ms_age = train[train["Title"]==2]["AgeGroups"].mode() #Teens
Mr_age = train[train["Title"]==3]["AgeGroups"].mode() #Young Adults
Mrs_age = train[train["Title"]==4]["AgeGroups"].mode() #Adults
Rare_age = train[train["Title"]==5]["AgeGroups"].mode() #Adult
Royal_age = train[train["Title"]==6]["AgeGroups"].mode() #Adults
#Demonstrating what I'm doing when using .mode
print(Master_age)
print(Ms_age)
print(Mrs_age)
train.head()


# In[ ]:


title_age_map = {1: "Small children", 2: "Teens", 3: "Young Adults", 4: "Adults", 5: "Adults", 6: "Adults"}

for x in range(len(train["AgeGroups"])):
    if train["AgeGroups"][x] == "Unknown":
        train["AgeGroups"][x] = title_age_map[train["Title"][x]]

for x in range(len(test["AgeGroups"])):
    if test["AgeGroups"][x] == "Unknown":
        test["AgeGroups"][x] = title_age_map[train["Title"][x]]


# Now that we filled in the missing values, let's turn those into numbers that our computer can process easily.

# In[ ]:


Age_map = {"Small children": 1, "Children": 2, "Teens": 3, "Adolescent": 4, "Young Adults": 5, "Adults": 6, "Elders": 7}

train["AgeGroups"] = train["AgeGroups"].map(Age_map)
test["AgeGroups"] = test["AgeGroups"].map(Age_map)
train.head()


# Now that we are finally extracted everything we could from the name and the age columns, we can now get rid of them.

# In[ ]:


train = train.drop(["Name"], axis = 1)
test = test.drop(["Name"], axis = 1)
train = train.drop(["Age"], axis = 1)
test = test.drop(["Age"], axis = 1)


# Sex Feature

# In[ ]:


#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()


# Embarked Feature

# In[ ]:


#map each Embarked value to a numerical value that can be read by the computer. Almost there!
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()


# Finally, the fare feature.
# We'll try and categorize this into more logical "bins"

# In[ ]:


print(train.Fare.max())
print(train.Fare.min())


# In[ ]:


#fill in missing Fare value in test set based on mean fare for that Pclass 
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)
        
#map Fare values into groups of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:





# **Training Data**

# In[ ]:


from sklearn.model_selection import train_test_split

predict = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predict, target, test_size = 0.22, random_state = 0)


# Testing Different Models
# I will be testing the following models with my training data (got the list from here):
# 
# * Gaussian Naive Bayes
# * Logistic Regression
# * Support Vector Machines
# * Perceptron
# * Decision Tree Classifier
# * Random Forest Classifier
# * KNN or k-Nearest Neighbors
# * Stochastic Gradient Descent
# * Gradient Boosting Classifier
# For each model, we set the model, fit it with 80% of our training data, predict for 20% of the training data and check the accuracy.

# In[ ]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[ ]:


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[ ]:


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[ ]:


# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# In[ ]:


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# In[ ]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes',  'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian ,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


I decide to use the Gradient Boosting Classifier since it has a higher score overall.


# In[ ]:


#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = gbk.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)

