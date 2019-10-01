#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# preview the data
titanic_df.head()


# In[ ]:


titanic_df.info()
print("----------------------------")
test_df.info()


# In[ ]:


# drop unnecessary columns, these columns won't be useful in analysis and prediction
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)
test_df    = test_df.drop(['Name','Ticket'], axis=1)


# In[ ]:


# Embarked

# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
test_df["Embarked"] = test_df["Embarked"].fillna("S")
#create a new column for Southampton
def get_southampton(passenger):
    Embarked = passenger
    return 1 if (Embarked=='S').bool() else 0
titanic_df['Southampton']=titanic_df[['Embarked']].apply(get_southampton, axis=1)
test_df['Southampton']=test_df[['Embarked']].apply(get_southampton, axis=1)
#create a new column for Cherbourg
def get_cherbourg(passenger):
    Embarked = passenger
    return 1 if (Embarked=='C').bool() else 0
titanic_df['Cherbourg']=titanic_df[['Embarked']].apply(get_cherbourg, axis=1)
test_df['Cherbourg']=test_df[['Embarked']].apply(get_cherbourg, axis=1)
#create a new column for Queenstown
def get_queenstown(passenger):
    Embarked = passenger
    return 1 if (Embarked=='Q').bool() else 0
titanic_df['Queenstown']=titanic_df[['Embarked']].apply(get_queenstown, axis=1)
test_df['Queenstown']=test_df[['Embarked']].apply(get_queenstown, axis=1)


# In[ ]:


# Fare

# only for test_df, since there is a missing "Fare" values
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# convert from float to int
titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare']    = test_df['Fare'].astype(int)

# get fare for survived & didn't survive passengers 
fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0]
fare_survived     = titanic_df["Fare"][titanic_df["Survived"] == 1]

# get average and std for fare of survived/not survived passengers
avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])

# plot
titanic_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))

avgerage_fare.index.names = std_fare.index.names = ["Survived"]
avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)


# In[ ]:


# Age 

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# axis3.set_title('Original Age values - Test')
# axis4.set_title('New Age values - Test')

# get average, std, and number of NaN values in titanic_df
average_age_titanic   = titanic_df["Age"].mean()
std_age_titanic       = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

# plot original Age values
# NOTE: drop all null values, and convert to int
titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
# test_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# fill NaN values in Age column with random values generated
titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1
test_df["Age"][np.isnan(test_df["Age"])] = rand_2

# convert from float to int
titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age']    = test_df['Age'].astype(int)
        
# plot new Age Values
titanic_df['Age'].hist(bins=70, ax=axis2)
# test_df['Age'].hist(bins=70, ax=axis4)


# In[ ]:


# .... continue with plot Age column

# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, titanic_df['Age'].max()))
facet.add_legend()

# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = titanic_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)


# In[ ]:


# Cabin
# It has a lot of NaN values, so it won't cause a remarkable impact on prediction
#create a new column for Cabin
def get_cabin(passenger):
    Cabin = passenger
    return 0 if (Cabin=='').bool() else 1
titanic_df['Room']=titanic_df[['Cabin']].apply(get_cabin, axis=1)
test_df['Room']=test_df[['Cabin']].apply(get_cabin, axis=1)
titanic_df.drop("Cabin",axis=1,inplace=True)
test_df.drop("Cabin",axis=1,inplace=True)


# In[ ]:


# Family

# Instead of having two columns Parch & SibSp, 
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]
titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1
titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0

test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]
test_df['Family'].loc[test_df['Family'] > 0] = 1
test_df['Family'].loc[test_df['Family'] == 0] = 0

# drop Parch & SibSp
titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)
test_df    = test_df.drop(['SibSp','Parch'], axis=1)

# plot
fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

# sns.factorplot('Family',data=titanic_df,kind='count',ax=axis1)
sns.countplot(x='Family', data=titanic_df, order=[1,0], ax=axis1)

# average of survived for those who had/didn't have any family member
family_perc = titanic_df[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)

axis1.set_xticklabels(["With Family","Alone"], rotation=0)


# In[ ]:





# In[ ]:


# Pclass

#create a new column for first class
def get_firstclass(passenger):
    Pclass = passenger
    return 1 if (Pclass==1).bool() else 0
titanic_df['FirstClass']=titanic_df[['Pclass']].apply(get_firstclass, axis=1)
test_df['FirstClass']=test_df[['Pclass']].apply(get_firstclass, axis=1)
#create a new column for second class
def get_secondclass(passenger):
    Pclass = passenger
    return 1 if (Pclass==2).bool() else 0
titanic_df['SecondClass']=titanic_df[['Pclass']].apply(get_secondclass, axis=1)
test_df['SecondClass']=test_df[['Pclass']].apply(get_secondclass, axis=1)
#create a new column for third class
def get_thirdclass(passenger):
    Pclass = passenger
    return 1 if (Pclass==3).bool() else 0
titanic_df['ThirdClass']=titanic_df[['Pclass']].apply(get_thirdclass, axis=1)
test_df['ThirdClass']=test_df[['Pclass']].apply(get_thirdclass, axis=1)
#create a new column for females
def get_female(passenger):
    Sex = passenger
    return 1 if (Sex=='female').bool() else 0
titanic_df['Female']=titanic_df[['Sex']].apply(get_female, axis=1)
test_df['Female']=test_df[['Sex']].apply(get_female, axis=1)
#create a new column for males
def get_male(passenger):
    Sex = passenger
    return 1 if (Sex=='male').bool() else 0
titanic_df['Male']=titanic_df[['Sex']].apply(get_male, axis=1)
test_df['Male']=test_df[['Sex']].apply(get_male, axis=1)


# In[ ]:


# define training and testing sets
titanic_df=titanic_df.drop(['Pclass'],axis=1)
test_df=test_df.drop(['Pclass'],axis=1)
titanic_df=titanic_df.drop(['Sex'],axis=1)
test_df=test_df.drop(['Sex'],axis=1)
titanic_df=titanic_df.drop(['Fare'],axis=1)
test_df=test_df.drop(['Fare'],axis=1)
titanic_df=titanic_df.drop(['Age'],axis=1)
test_df=test_df.drop(['Age'],axis=1)
titanic_df=titanic_df.drop(['Embarked'],axis=1)
test_df=test_df.drop(['Embarked'],axis=1)
X_train = titanic_df.drop("Survived",axis=1)
Y_train = titanic_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)


# In[ ]:


# Support Vector Machines

# svc = SVC()

# svc.fit(X_train, Y_train)

# Y_pred = svc.predict(X_test)

# svc.score(X_train, Y_train)


# In[ ]:


# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)


# In[ ]:


# knn = KNeighborsClassifier(n_neighbors = 3)

# knn.fit(X_train, Y_train)

# Y_pred = knn.predict(X_test)

# knn.score(X_train, Y_train)


# In[ ]:


# Gaussian Naive Bayes

# gaussian = GaussianNB()

# gaussian.fit(X_train, Y_train)

# Y_pred = gaussian.predict(X_test)

# gaussian.score(X_train, Y_train)


# In[ ]:


# get Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(titanic_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
coeff_df


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)

