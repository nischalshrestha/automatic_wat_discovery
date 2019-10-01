#!/usr/bin/env python
# coding: utf-8

# ## Competition Description
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# In[ ]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Load Dataset

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# ## Analyse Data

# **Which features are available in the dataset?**

# In[ ]:


train_df.columns.values


# **Categorical features** - (Classify the samples into sets of similar samples) Survived, Sex, Embarked, Pclass
# 
# **Numerical features** - (Change from sample to sample) Age, Fare, SibSp, Parch
# 
# **Target feature** is Survived

# In[ ]:


#preview the data
train_df.head(10)


# As we can observe - 
# * **Ticket** - is mix of numeric and alphanumeric data types.
# * **Cabin** - is alphanumeric, also has some missing data
# * **Age** - has some missing data

# **Distribution of numerical feature values**

# In[ ]:


train_df.describe()


# We can see here that -
# * Around **38%** samples survived representative of the actual survial rate at **32%**
# * The **Age** feature ranges from 0.42 to 80
# * The **Pclass** feature denotes the class of ticket which are categorized as 1,2,3
# * **Fares** varied significantly with few passengers (<1%) paying as high as $512.

# **Distribution of categorical features**

# In[ ]:


train_df.describe(include=['O'])


# We can observe here that -
# * **Names** are unique across the dataset
# * **Sex** has two values (male, female), with 65% male (577/891)
# * **Cabin** values have several duplicates. Several passengers shared same cabin.
# * **Emabarked** takes three possible values. **S** port used by most passengers (72%)
# * **Ticket** feature have 22% duplicates (unique = 681)

# **Which features contain blank, null or empty values?**

# In[ ]:


train_df.isnull().sum().sort_values(ascending=False)


# * The Cabin feature has 687 missing values, we might need to drop this feature since 77% of it is missing.
# * The Age feature has 177 (19%) missing values
# * The Embarked has 2 missing values, which can be filled easily.

# ## Pivoting data

# To confirm some of our observations and assumptions, we can quickly analyze our feature correlations by pivoting features against each other.
# 
# It makes sense doing so only for features which are categorical (Sex), ordinal (Pclass) or discrete (SibSp, Parch) type.

# In[ ]:


train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# We observe significant correlation among **Pclass=1** and Survived. The upper class passengers were more likely to have survived.

# In[ ]:


train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# We observe that **female** had very high survival rate at **74%**.

# In[ ]:


train_df[['Parch','Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# The **Parch and SibSp** features have zero correlation for certain values. So it may be best to derive a feature or a set of features from these individual features.

# ## Visualize data

# Understanding correlations between numerical features and our goal Survived.

# #### 1. Age and Sex

# In[ ]:


g = sns.FacetGrid(train_df, col='Sex', row='Survived', margin_titles=True)
g.map(plt.hist, 'Age', bins=20)


# **Observations**-
# * Female passengers had much better survival rate than male passengers.
# * Infants (Age <=4 ) had high survival rate.
# * Older passenger (Age = 80) survived.
# * Large number of passengers between 15-25 years old did not survived.
# * Most passengers were between 15-35 age range.

# **Decisions**
# * Consider Age and Sex in our model training
# * Remove null values from Age features
# * Band age groups

# #### 2. Age and Pclass

# In[ ]:


grid = sns.FacetGrid(train_df, col = 'Survived', row = 'Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# **Observations :-**
# * Pclass = 3 had most passengers, however most did not survive.
# * Most passengers in Pclass = 1 survived.
# * Infant passengers in Pclass = 2 and 3 mostly survived.
# * Pclass varies in terms of Age distribution of passengers.
# 
# **Decisions :-**
# * Consider Pclass for model training.

# #### 3. Sex, Embarked and Pclass

# In[ ]:


grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect = 1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# **Observations:-**
# * Female passengers had much better survival rate than males, except in Embarked = C, where males had higher survival rate. This could be correlation between Pclass and Embarked and in turn Pclass and Survived.
# * Males had better survival rate in Pclass=3 when compared with Pclass= 2 for C and Q ports.
# * Ports of embarkation have varying survival rates for Pclass=3 and among male passengers. Correlating
# 
# **Decisions:-**
# * Complete and add Embarked feature to model training.

# #### 4. Embarked, Sex and Fare

# In[ ]:


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size = 2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# **Observations:-**
# * Higher fare paying passengers had better survival.
# * Port of embarkation correlates with survival rates.
# 
# **Decisions:-**
# * Consider Fare feature.

# ## Feature Engineering

# **Features needed to be dropped from the dataset** -
# * PassengerId - as it does not contribute to survival
# * Name - doest not contribute to survivale
# * Cabin - contains 77% missing data (drop from both train and test dataset)
# * Ticket - doest not contribute to survival and also contains 22% duplicates
# 
# First we will drop the Cabin and Ticket features.

# In[ ]:


train_df = train_df.drop(['Cabin','Ticket'], axis = 1)
test_df = test_df.drop(['Cabin', 'Ticket'], axis=1)


# **New features need to be created from existing features** - 
# * Extract titles from passenger names
# * Convert categorical feature like sex to numerical values
# * Age bands - to turn the continous numerical feature into an ordinal categorical feature
# * Feature called Family based on Parch and SibSp to get total count of family members on board
# * Age times class
# * Convert categorical Embarked feature to numeric feature.
# * Fare range feature 

# ### 1. Extract Title feature using regular expressions.

# *Plot Title and Sex*

# In[ ]:


combine = [train_df, test_df]

for data in combine:
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
pd.crosstab(train_df['Title'], [train_df['Sex'], train_df['Survived']])


# In[ ]:


grid = sns.countplot(x='Title', data=train_df)
grid = plt.setp(grid.get_xticklabels(), rotation=45)


# Certain titles mostly survived (Mme, Ms, Lady, Sir) or did not (Rev, Don, Jonkheer).
# 
# We can replace many titles with a more common name or classify them as Rare.

# In[ ]:


for data in combine:
    data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    
train_df[['Title','Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


grid = sns.countplot(x='Title', data=train_df)
grid = plt.setp(grid.get_xticklabels(), rotation=45)


# *Convert categorical titles to ordinal*

# In[ ]:


title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
for data in combine:
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)
    
train_df.head()


# **We can now safely drop the Name feature from training and testing datasets.**
# **We also do not need the PassengerId feature in the training dataset**

# In[ ]:


train_df = train_df.drop(['Name', 'PassengerId'], axis = 1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


# ### 2. Converting a categorical feature (Sex) to numerical value.

# Converting Sex feature to a new feature where female = 1 and male = 0

# In[ ]:


for data in combine:
    data['Sex'] = data['Sex'].map({'female':1, 'male':0}).astype(int)
    
train_df.head()


# ### 3. To turn the continous numerical feature (Age) into an ordinal categorical feature.

# Now we need to tackle the issue with the age features missing values. We will create an array that contains random numbers, which are computed based on the mean age value in regards to the standard deviation and is_null.

# In[ ]:


for data in combine:
    mean = train_df['Age'].mean()
    std = test_df['Age'].std()
    null_count = data['Age'].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size = null_count)
    
    age_slice = data["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    data["Age"] = age_slice
    data["Age"] = train_df["Age"].astype(int)
    
train_df["Age"].isnull().sum()


# In[ ]:


train_df.head()


# Let us create Age bands and determine correlations with Survived.

# In[ ]:


train_df['AgeBand']= pd.cut(train_df['Age'],5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# Let us replace Age with ordinals based on these bands.

# In[ ]:


for data in combine:
    data.loc[data['Age'] <=16, 'Age'] = 0
    data.loc[(data['Age'] >16 ) & (data['Age'] <= 32), 'Age']=1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age']
train_df.head()


# We can now remove the AgeBand feature.

# In[ ]:


train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]


# ### 4. Create new feature - FamilySize based on Parch and SibSp to get total count of family members on board

# In[ ]:


for data in combine:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Create another feature IsAlone, where if FamilySize = 1 then IsAlone = 1 else 0

# In[ ]:


for data in combine:
    data['IsAlone']=0
    data.loc[data['FamilySize'] == 1, 'IsAlone']=1
    
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# Now we can drop Parch, SibSp and FamilySize feature in favour of IsAlone.

# In[ ]:


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()


# ### 5. Age times Class

# In[ ]:


for data in combine:
    data['AgeClass'] = data['Age']* data['Pclass']


# ### 6. Convert categorical Embarked feature to numeric feature.

# **Embarked features has 2 missing values**, we need to fill these with most common occurance.

# In[ ]:


freq = train_df.Embarked.dropna().mode()[0]
freq


# The most frequent occuring value is **S**

# In[ ]:


for data in combine:
    data['Embarked'] = data['Embarked'].fillna(freq)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


#convert categorical feature to numeric
for data in combine:
    data['Embarked'] = data['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
    
train_df.head()


# ### 7. Fare range feature

# In[ ]:


train_df["Fare"].isnull().sum(), test_df["Fare"].isnull().sum()


# Our test dataset contains one missing value for Fare feature, so we need to replace it with most frequently occuring value.

# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)


# Let us create Fare bands and determine correlations with Survived.

# In[ ]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# Convert the Fare feature to ordinal values based on the FareBand.

# In[ ]:


for data in combine:
    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[ data['Fare'] > 31, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head()


# In[ ]:


test_df.head()


# ## Building Machine Learning Models

# Now we are ready to trian our model and predict the required solution. Our problem is a classification and regression problem. We need to identify relation between output **(Survived or not)** with other variables or features **(like Age, sex, Pclass)**

# In[ ]:


X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# **Import libraries**

# In[ ]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# ### 1. Stochastic gradient descent (SGD) learning

# In[ ]:


sgd = SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(acc_sgd, '%')


# ### 2. Logistic Regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log, '%')


# ### 3. Perceptron

# In[ ]:


pcpt = Perceptron(max_iter=5)
pcpt.fit(X_train, Y_train)
Y_pred = pcpt.predict(X_test)

acc_pcpt = round(pcpt.score(X_train, Y_train)*100, 2)
print(acc_pcpt, '%')


# ### 4. Random Forest

# In[ ]:


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)

acc_rf = round(rf.score(X_train, Y_train) * 100, 2)
print(acc_rf, '%')


# ### 5. Decision Tree

# In[ ]:


dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
Y_pred = dt.predict(X_test)

acc_dt = round(dt.score(X_train, Y_train)*100,2)
print(acc_dt, '%')


# ### 6. Linear SVC

# In[ ]:


svc = LinearSVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train)*100, 2)
print(acc_svc, '%')


# ### 7. KNN

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train)*100, 2)
print(acc_knn, '%')


# ### 8. Gaussian Naive Bayes

# In[ ]:


gnb = GaussianNB()
gnb.fit(X_train, Y_train)
Y_pred = gnb.predict(X_test)

acc_gnb = round(gnb.score(X_train, Y_train)*100, 2)
print(acc_gnb, '%')


# ### Which is the best model ?

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_rf, acc_gnb, acc_pcpt, 
              acc_sgd, acc_svc, acc_dt]})
models.sort_values(by='Score', ascending=False)


# Both **Decision Tree and Random Forest** score the same, we choose to use Random Forest, but first let us check, how random forest performs, when we use cross validation.

# In[ ]:


# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)

acc_rf = round(rf.score(X_train, Y_train) * 100, 2)
print(acc_rf, '%')


# ### Submission

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv('submission.csv', index=False)


# #### References
# * __[A journey through Titanic](https://www.kaggle.com/omarelgabry/a-journey-through-titanic)__
# * __[Titanic Data science solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)__
# * __[End to End Project with Python](https://www.kaggle.com/niklasdonges/end-to-end-project-with-python)__

# Our submission to the competition site Kaggle results in scoring 0.77033 ranked 6040.
