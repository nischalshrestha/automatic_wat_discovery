#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

sns.set(style='white', context='notebook', palette='dark')


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Definining the problem**
# 
# Our goal is predict the surival of passengers on the Titanic. This is a prediction and binary classification problem, so we will need to use classification models once we have completed our analysis.

# First start by opening up the datasets

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# **Exploratory Analysis**
# 
# In this section I'll try to answer the following questions:
# 
# 1. How many features are there and how are they encoded? 
# 2. Are there missing values? Can we reasonably infer these values (e.i. impute them)?
# 3. What features are correlated with survival? 
# 4. What features can we combine? 

# In[ ]:


print(train.dtypes)


# In[ ]:


train.describe()


# First, this is my understanding of whether a feature is continuous or discrete (e.i. categorical)
# 1. If something like the "average" of a feature has some meaning, it is usually a continuous variable. For example, we can interpret the mean age of passengers on the Titanic, therefore it is a continuous feature.
# 2. If the "average" doesn't have a reasonable interpretation, it is usually a discrete variable. For example, can we really interpret the mean passenger class? Obviously, passenger class 2.3 doesn't have much meaning, you clearly cant be in Pclass 2.3. A more reasonable way to understand this feature would be to find the proportion of passengers in each Pclass. Therefore, it is a categorical feature.
# 
# We can see the following:
# 
# * Surived is a categorical response variable
# * Pclass and Embarked are categorical features with 3 levels. Pclass however is an ordinal, which means that there is an "order" to the feature. Obviously, Pclass 1 > Pclass 2 > Pclass 3. This is not true for Embarked since we can't infer any meaning from the ports "Q", "C", and "S".
# * Age is a continuous variable. The average age was ~30 years old.
# * SibSp and Parch are categorical
# * Fare is a continuous feature
# * Name, ticket, and cabin are qualitative features

# In[ ]:


print(train.isnull().sum()/train.shape[0])


# We can see that we are Age, Cabin, and Embarked all have missing values. There are roughly 900 observations in the training set and we can see that the Cabin feature is missing ~80% of its values and age is missing ~20% of its values. This corresponds to ~700 missing cabin observations and ~200 missing age features. We'll return to these later, but for now I will say that filling in the missing values for embarked is going to be quite simple and I don't believe that I will be able to extract much useful information from the cabin feature. The age feature seems the most interesting to me since I believe that it is reasonable to infer that some age groups will fare better than others. We've all heard of the phrase "Women and children first", so putting in some work on the age feature will probably end up paying off.
# 
# Next, we will begin to explore our categorical and continuous features graphically.

# In[ ]:


print(test.isnull().sum()/test.shape[0])


# Similar results to the training set, but there is also a missing fare value. We'll deal with this later.

# **Passenger Class**

# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

d=train.groupby(['Pclass'])[['Survived']].mean()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5,ax=axis1).set_title("Survival Rate")

d=train.groupby(['Pclass'])[['Survived']].count()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5,ax=axis2).set_title("Count")


# As expected, the survival rate is ordered by class and most passengers were in third class. Those in higher passengers have a higher surival rate than those in lower classes. Somewhat suprisingly, there are more passengers in first class than in second class.

# **Parch and SibSp**
# 

# In[ ]:


fig, ((axis1,axis2), (axis3, axis4)) = plt.subplots(2,2,figsize=(15,12))

d=train.groupby(['SibSp'])[['Survived']].mean()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis1).set_title("SibSp Survival Rate")

d=train.groupby(['SibSp'])[['Survived']].count()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis2).set_title('SibSp Count')

d=train.groupby(['Parch'])[['Survived']].mean()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis3).set_title("Parch Survival Rate")

d=train.groupby(['Parch'])[['Survived']].count()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis4).set_title('Parch Count')


# It looks like most passengers were alone, and passengers who were alone typically had lower survival rates. I think the surival rate plots above are a little misleading, there are not many observations for for higher values of Parch and SibSp.
# 
# From here it seems fairly natural to combine these features and create a new one for the family size. This will also help with increasing the number of observations for larger families.

# In[ ]:


def family_size(data):
    data['FamilySize'] = data['SibSp'] + data['Parch'] 
    return data


train = family_size(train)
test = family_size(test)


# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

d=train.groupby(['FamilySize'])[['Survived']].mean()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis1).set_title('Survial Rate')

d=train.groupby(['FamilySize'])[['Survived']].count()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis2).set_title('Count')


# It's a little more clear now that if you were alone or in a family greater than 4 your chances of survival were lower. I still have some concerns with there not being many observations for large families. In solve this, I'll bin the families into alone, small, and large.

# In[ ]:


def Bin_family(data):
    data['FamilyBin'] = data['FamilySize'].map({0:0,1:1,2:1,3:1,4:2,5:2,6:2,7:2,8:2,9:2,10:2,11:2}).astype(int)
    return data


train=Bin_family(train)
test=Bin_family(test)


# In[ ]:


fig, (axis1, axis2) = plt.subplots(1,2,figsize=(15,4))

d=train.groupby(['FamilyBin'])[['Survived']].mean()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5,ax=axis1).set_title("Survival Rate")

d=train.groupby(['FamilyBin'])[['Survived']].count()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis2).set_title("Count")


# Now, I think this warrents some more investigation. I'm curious to see if this difference is due to smaller families having a disproportionate number of first class passenger

# In[ ]:


print(train.groupby(['FamilyBin', 'Pclass'])[['Survived']].mean())
print(train.groupby(['FamilyBin', 'Pclass'])[['Survived']].count())


# We can see that my concerns were mostly unfounded, there is quite a large variation in survival rates between family sizes, holding Pclass constant. We can see that 53% of passengers in first class who were alone survived, compared to the 73% (!) survival rate of first class passengers in small families. It also appears that if you were in a large third class family you were almost guarunteed to perish. 
# 
# Now lets see if these differences are perhaps due to families having women in children in them, which would increase the average survival rate of passengers in that family.

# **Gender and Embarked**

# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

d=train.groupby(['Sex'])[['Survived']].mean()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis1).set_title("Survival Rate")

d=train.groupby(['Embarked'])[['Survived']].mean()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis2).set_title("Survival Rate")


# Once again this is the expected result, females had a much higher survival rate than males since women and children had priority when getting off the ship. 
# 
# Interestingly, those who embarked from port "C" tended to have a much higher survival rate than those who embarked from port "Q" or "S". To me, this seems fairly unintuitive Clearly, this needs to be further investigated.

# In[ ]:


print(train.groupby(['Embarked', 'Pclass'])[['Survived']].count())
print(train.groupby(['Embarked', 'Pclass'])[['Survived']].mean())
print(train.groupby(['Embarked'])[['Pclass']].count())


# We can see that port C had a larger proportion of first class passengers than port Q or port S. This seems to explain why passengers who departed from port C had a higher survival rate. As a result, I don't think this predictor is very valuable - the differences seem to be explained by other factors and it doesn't match intuition. 

# **Name**
# 
# Before we can visualize this, we will have to do some feature engineering. 
# 
# We'll start by first extracting the surname from the Name feature. 

# In[ ]:


def titleExtract(data):
    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Mr')
    data['Title'] = data['Title'].replace(['Ms', 'Miss'], 'Miss')
    data['Title'] = data['Title'].replace(['Mlle', 'Mme','Lady', 'Countess', 'Dona'], 'Mrs')
    return data


train=titleExtract(train)
test=titleExtract(test)


# Now we can plot this since we have extracted a categorical feature from a qualitative one.

# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

d=train.groupby(['Title'])[['Survived']].mean()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis1).set_title("Survival Rate")

d=train.groupby(['Title'])[['Survived']].count()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5,ax=axis2).set_title("Count")

print(train.groupby(['Title'])[['Age']].mean())


# From this we can see that there is some variation in survival rates between the different Titles. Those with Miss and Mrs title's appeared to have much higher surivival rates than other titles.  Additionally we can see that those with the "Master" surname are children and those with the "Miss" surname are typically younger than than those with "Mr" or "Mrs".
# 
# In preparation for using this feature in the future, we can assign each category a number.

# In[ ]:


print(train.groupby(['Title', 'Sex'])[['Survived']].mean())
print(train.groupby(['Title', 'Sex'])[['Survived']].count())


# In[ ]:


print(train.groupby(['Title', 'Pclass'])[['Survived']].mean())
print(train.groupby(['Title', 'Pclass'])[['Survived']].count())


# Prior to an edit, I had a seperate title for those with a "special title". After looking at the model significance, it looked like this special title wasn't very useful. Below is what I originally wrote:
# 
# We can see that females with the special title had a 100% survival rate while males with a special title had a surival rate similar to the baseline male value. I think this implies that the "special" title isn't important.  
# 
# As a result I moved those with a "female" special title to Ms and those with a "male" special title to Mr.

# In[ ]:


def mapTitle(data):
    title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4}
    data['Title'] = data['Title'].map(title_map).astype(int)
    return data
   

train=mapTitle(train)
test=mapTitle(test)


# In[ ]:





# **Fare**
# 
# There is one missing value for fare in the test set. Instead of looking for it I'll just make an educated guess on its value based on the passenger's Pclass.

# In[ ]:


def fill_fare_nulls(data):
    data.loc[(data.Fare.isnull())&(data.Pclass==1), 'Fare']=60.3
    data.loc[(data.Fare.isnull())&(data.Pclass==2), 'Fare']=14.25
    data.loc[(data.Fare.isnull())&(data.Pclass==3), 'Fare']=8.05
    return data

test=fill_fare_nulls(test)


# To make it easier to plot I'll bin the fare based on the median values of fare at each Pclass.

# In[ ]:


train.groupby(['Pclass'])[['Fare']].median()


# In[ ]:


train['FareBand'] = pd.cut(train['Fare'], (-1, 8.05, 14.25, 60.2875, 1000), labels=['0','1','2','3'])
test['FareBand'] = pd.cut(test['Fare'], (-1, 8.05, 14.25, 60.2875, 1000), labels=['0','1','2','3']) 
train['FareBand'].astype(int)
test['FareBand'].astype(int)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

d=train.groupby(['FareBand'])[['Survived']].mean()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis1).set_title("Survival Rate")

d=train.groupby(['FareBand'])[['Survived']].count()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis2).set_title("Count")


# So it's quite clear that those who paid more had a higher survival rate - that's not too suprising or interesting at this point. What is interesting is that perhaps those who paid above median price for their ticket may have had a higher survival rate. That is, perhaps those in the "upper" first class did better than those in the "lower" first class.
# 
# A concern with this feature is that is conveys much of the same information as Pclass - we already know that people who paid more tended to have better outcomes. To me, this just confirms what we already know. I think a good solution is to use this information to motivate splitting passenger class based on fare.
# 
# **Revisiting Pclass**
# 
# I think a good way to split passenger class is based on the median fare of each Pclass. We are in a sense splitting passengers based on whether they are in the "upper" or "lower" part of their Pclass.

# In[ ]:


print(train.groupby('Pclass')['Fare'].median())
print(test.groupby('Pclass')['Fare'].median())


# The medians are a little different here, although not by much.

# In[ ]:


def splitClass_Train(data):
    data.loc[(data.Fare>60.3)&(data.Pclass==1), 'PassengerCat']=0
    data.loc[(data.Fare<=60.3)&(data.Pclass==1), 'PassengerCat']=1
    
    data.loc[(data.Fare>14.3)&(data.Pclass==2), 'PassengerCat']=2
    data.loc[(data.Fare<=14.3)&(data.Pclass==2), 'PassengerCat']=3
              
    data.loc[(data.Fare>8.1)&(data.Pclass==3), 'PassengerCat']=4
    data.loc[(data.Fare<=8.1)&(data.Pclass==3), 'PassengerCat']=5
    data['PassengerCat']=data['PassengerCat'].astype(int)
    return data


train=splitClass_Train(train)

def splitClass_Test(data):
    data.loc[(data.Fare>60.00)&(data.Pclass==1), 'PassengerCat']=0
    data.loc[(data.Fare<=60.00)&(data.Pclass==1), 'PassengerCat']=1
    
    data.loc[(data.Fare>15.8)&(data.Pclass==2), 'PassengerCat']=2
    data.loc[(data.Fare<=15.8)&(data.Pclass==2), 'PassengerCat']=3
              
    data.loc[(data.Fare>7.9)&(data.Pclass==3), 'PassengerCat']=4
    data.loc[(data.Fare<=7.9)&(data.Pclass==3), 'PassengerCat']=5
    data['PassengerCat']=data['PassengerCat'].astype(int)
    return data


test=splitClass_Test(test)


# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

d=train.groupby(['PassengerCat'])[['Survived']].mean()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis1).set_title('Surival Rate')

d=train.groupby(['PassengerCat'])[['Survived']].count()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5,ax=axis2).set_title('Count')


# Curiously, we can see that those in the bottom half of first class were typically worse off than those in the upper half of second class. I don't really have an explanation as to why this is the case, perhaps it's just chance or a higher number of female passengers in the "upper" second passenger class. Overall I think this might be a better way of utilizing the fare and Pclass feature.

# **Age**
# 
# I saved age because the goal is to use the title and passenger class features to predict age. 

# In[ ]:


avgAges=[]
avgAges=train.groupby(['Title', 'Pclass'])['Age'].median()
print(avgAges)


# We can see that those in Pclass=1 are generally older than those in Pclass 2 or 3. This matches with intuition since richer people are typically older. There isn't much variation between passenger classes for those with the "Maser (4)" title, so we wont disseminate based on passenger class for those with title 4. Next we fill in the missing age values using the information above. Admittedly, it's a little (very) ugly. I originally implemented with with loops but I found it to be pretty hard to interpret and perhaps equally as ugly.

# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
#Before filling in NA values
train['Age'].hist(bins=20, ax=axis1).set_title('Before Imputing')

def fillAgeNulls(data):
    data.loc[(data.Age.isnull())&(data.Title==1)&(data.Pclass==1), 'Age']=42
    data.loc[(data.Age.isnull())&(data.Title==1)&(data.Pclass==2), 'Age']=31
    data.loc[(data.Age.isnull())&(data.Title==1)&(data.Pclass==3), 'Age']=26
    
    data.loc[(data.Age.isnull())&(data.Title==2)&(data.Pclass==1), 'Age']=30
    data.loc[(data.Age.isnull())&(data.Title==2)&(data.Pclass==2), 'Age']=24
    data.loc[(data.Age.isnull())&(data.Title==2)&(data.Pclass==3), 'Age']=18
    
    data.loc[(data.Age.isnull())&(data.Title==3)&(data.Pclass==1), 'Age']=39
    data.loc[(data.Age.isnull())&(data.Title==3)&(data.Pclass==2), 'Age']=32
    data.loc[(data.Age.isnull())&(data.Title==3)&(data.Pclass==3), 'Age']=31
    
    data.loc[(data.Age.isnull())&(data.Title==4), 'Age']=4
    
    
    return data


train=fillAgeNulls(train)
test=fillAgeNulls(test)

train['Age'].hist(bins=20, ax=axis2).set_title('After Imputing')


# I would like to hear some feedback about this - I'm quite concerned with how the distribution has changed after imputing age. 
# 
# To reduce the amount of noise we will next bin age. This also helps group some "special" age categories together - we expect children and especially young to have a higher survival rate and seniors to have a lower survival rate. 
# 
# The cuts here are somewhat arbitrary and I'm not a huge fan of this implementation - I've essentially grouped being into either a young children, kids, adult, and senior category. 
# 
# I'd be interested in hearing feedback about this.

# In[ ]:



train['AgeBand']=pd.cut(train['Age'], (0, 7, 60, 80), labels=['0','1','2'])
test['AgeBand']=pd.cut(test['Age'], (0, 7, 60, 80), labels=['0','1','2'])
train['AgeBand'].astype(int);
test['AgeBand'].astype(int);

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

d=train.groupby(['AgeBand'])[['Survived']].mean()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis1).set_title('Surival Rate')

d=train.groupby(['AgeBand'])[['Survived']].count()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5,ax=axis2).set_title('Count')


# In[ ]:


print(train.groupby(['AgeBand', 'Sex'])[['Survived']].mean())
print(train.groupby(['AgeBand', 'Sex'])[['Survived']].count())


# Interestingly, if you were a female your survival rate did not appear to depend on whether you were a child, middle aged, or old (note there are only 3 observations for female seniors). Additionally, if you were a male it only mattered if you were young - old and middle aged males have a very similar survival rate. This information however is already encoded 

# **Ticket and other features**

# I haven't explored cabin/ticket in depth, but from what I have seen they are not particularly useful. I'm also worried about introducing more noise into the model. Perhaps in the future I will explore them in depth but for now I will neglect them.

# **Feature Selection**
# 
# First we will drop the features that we used to create other features and the features that were not very useful.
# 
# We used Name, SibSp, Parch, Fare, Age, FareBand, PClass in our feature engineering so we will drop them here.
# 
# Ticket, PassengerId, Cabin, Embarked were either not explored or deemed to be not very useful.

# In[ ]:


def mapSex(data):
    sex_map = {"female": 0, "male": 1,}
    data['Sex'] = data['Sex'].map(sex_map).astype(int)
    return data


train = mapSex(train)
test = mapSex(test)


# In[ ]:


train=train.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin', 'SibSp','Parch','Fare', 'Age', 'Embarked', 'FamilySize','FareBand', 'Pclass', 'Sex'])
test=test.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin','SibSp', 'Parch','Fare','Age', 'Embarked', 'FamilySize','FareBand', 'Pclass', 'Sex'])


# Here we will get a view of the dataset that we will use in our predictions

# In[ ]:


train.head(10)


# In[ ]:


sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix


# Next get the dummy variables for our non-ordinal categorical features. 

# In[ ]:


print(train.dtypes)


# In[ ]:


train=pd.get_dummies(train,columns=['Title'])
test=pd.get_dummies(test,columns=['Title'])


# Before we start modeling, make sure we have no missing values on our training and test set.

# In[ ]:


print(train.isnull().sum())
print("\nTest Set:")
print(test.isnull().sum())


# No missing values so we are ready to model.

# **Modeling**
# 
# A couple things to note before starting this section.
# 
# 1. It's important to cross validate your results and be weary of overfitting. The prediction accuracy that your models have on your training set is usually not a very good indicator of how well it will perform on unseen data. Cross validation scores are an unbiased estimate of your models prediction accuracy that will aid in model selection.
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict 
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import f_regression
from sklearn.ensemble import GradientBoostingClassifier


# Specify the number of folds. My understanding is that 5 or 10 folds is standard. I'll choose 10.

# In[ ]:


train_X=train.drop('Survived', axis=1)
train_Y=train['Survived'].astype(int)
test_X=test
KF=KFold(n_splits=10, random_state=1)

models=[]
modelScores=[]
modelSTD=[]


# I won't go into too much detail about the individual models here. I think there are some pretty good resources on details about them online.
# 
# 
# 
# **Logistic Regression**

# In[ ]:


LR_model = LogisticRegression()

CV=cross_val_score(LR_model,train_X,train_Y,cv=KF, scoring="accuracy")
CV.mean()
models.append('LogisticRegression')
modelScores.append(round(CV.mean(),3))
modelSTD.append(round(CV.std(),3))


# **Trees**

# Decision Tree:
# 
# I've heard in class that decision trees are typically quite overfitted and require pruning. From what I've been told the idea is to fit your tree model and then prune it back. This will hopefully improve its performance on unseen data.

# In[ ]:


DT_model = tree.DecisionTreeClassifier(criterion = "gini", splitter ='random', random_state=1)
DT_model_fit = DT_model.fit(train_X, train_Y)

CV=cross_val_score(DT_model,train_X,train_Y,cv=KF, scoring="accuracy")
CV.mean()
models.append('Decision Tree')
modelScores.append(round(CV.mean(),3))
modelSTD.append(round(CV.std(),3))


# The following code was found [here](https://stackoverflow.com/questions/49428469/pruning-decision-trees) and was modified.

# In[ ]:


from sklearn.tree._tree import TREE_LEAF

def prune_index(inner_tree, index, threshold):
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are shildren, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)

print(sum(DT_model.tree_.children_left < 0))
# start pruning from the root
prune_index(DT_model.tree_, 0, 5)
sum(DT_model.tree_.children_left < 0)


# In[ ]:


CV=cross_val_score(DT_model,train_X,train_Y,cv=KF, scoring="accuracy")
CV.mean()
models.append('Pruned Decision Tree')
modelScores.append(round(CV.mean(),3))
modelSTD.append(round(CV.std(),3))


# In[ ]:


DT_cm=confusion_matrix(train_Y, DT_model.predict(train_X))
DT_cm


# Random Forest:

# We will do a grid search to find the best parameters. Thanks to [this kernel](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling) for the gridsearch code! Note that this code will take 1-3 minutes to run depending on hardware. 

# In[ ]:


from sklearn.model_selection import GridSearchCV

RFC = RandomForestClassifier()


RF_grid = {"max_depth": [20,None],
              "max_features": [2, 3, 5],
              "min_samples_split": [2, 4],
              "min_samples_leaf": [3, 5, 7],
              "bootstrap": [False],
              "n_estimators" :[100, 200],
              "criterion": ["gini"]}

gsRFC = GridSearchCV(RFC,param_grid = RF_grid, cv=10, scoring="accuracy", n_jobs= -1, verbose = 1)

gsRFC.fit(train_X,train_Y)

RFC_best = gsRFC.best_estimator_

# Best score
print(gsRFC.best_score_)
print(gsRFC.best_params_)


# We will use the paramters that gave the best cross validated result:
# 

# In[ ]:


#Random Forest
RF_model = RandomForestClassifier(bootstrap= False,
 criterion = 'gini',
 max_depth = 20,
 max_features=2,
 min_samples_leaf = 7,
 min_samples_split = 4,
 n_estimators = 200, random_state=1)

RF_model.fit(train_X, train_Y)


CV=cross_val_score(RF_model,train_X,train_Y,cv=KF, scoring="accuracy")
models.append('RandomForest')
#modelScores.append(round(RF_model.oob_score_,3))
modelScores.append(round(CV.mean(),3))
modelSTD.append('NA')

featureImportance = pd.concat((pd.DataFrame(train_X.columns, columns = ['Feature']), 
           pd.DataFrame(RF_model.feature_importances_, columns = ['Importance'])), 
          axis = 1).sort_values(by='Importance', ascending = False)[:20]
plt.subplots(figsize=(20,8))
sns.barplot(x=featureImportance['Feature'], y=featureImportance['Importance'], alpha=.5).set_title('Feature Importance')


# Title is quite important primarly because it indicated whether a passenger was male or female.  Unsuprisingly, passenger category was also quite important because it encodes the socioeconomic. Somewhat suprisingly, the age bin of the passenger wasn't too important. I'll keep in the model because I think I've removed a lot of the noise from the dataset.

# In[ ]:


from sklearn.model_selection import GridSearchCV
ETC = RandomForestClassifier()

ET_grid = {"max_depth": [None],
              "max_features": [1, 3, 7],
              "min_samples_split": [2, 3, 7],
              "min_samples_leaf": [1, 3, 7],
              "bootstrap": [False],
              "n_estimators" :[100, 200, 300, 400],
              "criterion": ["gini"]}

gsETC = GridSearchCV(ETC,param_grid = ET_grid, cv=10, scoring="accuracy", n_jobs= -1, verbose = 1)

gsETC.fit(train_X,train_Y)

RET_best = gsETC.best_estimator_

# Best score
print(gsETC.best_score_)
print(gsETC.best_params_)


# In[ ]:


ET_model = ExtraTreesClassifier(bootstrap= False,
 criterion = 'gini',
 max_depth = None,
 max_features=7,
 min_samples_leaf = 3,
 min_samples_split = 2,
 n_estimators = 100, random_state=13)

ET_model_fit = ET_model.fit(train_X, train_Y)

CV=cross_val_score(ET_model,train_X,train_Y,cv=KF, scoring="accuracy")
CV.mean()
models.append('ExtraTrees')
modelScores.append(round(CV.mean(),3))
modelSTD.append(round(CV.std(),3))


# **Support Vector Machines**

# In[ ]:


#SVC
SVC_model=svm.SVC(probability=True)
SVC_model.fit(train_X,train_Y)

CV=cross_val_score(SVC_model,train_X,train_Y,cv=KF, scoring="accuracy")
CV.mean()
models.append('SVC')
modelScores.append(round(CV.mean(),3))
modelSTD.append(round(CV.std(),3))


#NuSVC
NuSVC_model=svm.NuSVC(probability=True)

CV=cross_val_score(NuSVC_model,train_X,train_Y,cv=KF, scoring="accuracy")
CV.mean()
models.append('NuSVC')
modelScores.append(round(CV.mean(),3))
modelSTD.append(round(CV.std(),3))


#LinearSVC
LSVC_model=svm.LinearSVC()

CV=cross_val_score(LSVC_model,train_X,train_Y,cv=KF, scoring="accuracy")
CV.mean()
models.append('LinearSVC')
modelScores.append(round(CV.mean(),3))
modelSTD.append(round(CV.std(),3))


# In[ ]:


ModelComparison=pd.DataFrame({'CV Score':modelScores, 'Std':modelSTD}, index=models)
ModelComparison


# Here we can see a summary of each model showing their cross validated score and the standard deviation. 
# 
# There is no/small difference in accuracy between the Pruned/unpruned decision tree. I went back and make sure that it worked and it indeed does. 
# 
# We can see that SVC and the tree models performed quite well. I'm most confident with the random forest because its score is based off of an out of bag estimate. Even though there is virtually no difference in score between both decision trees, I'll assume the pruned one is a little better because we have remove reduced its variance.

# In[ ]:



test_ID = pd.read_csv('../input/test.csv')
test_ID = test_ID['PassengerId']

Survival_predictions = RF_model.predict(test)
ID=np.arange(892,1310,1)

submission=pd.DataFrame({
        "PassengerId": ID,
        "Survived": Survival_predictions
    })
submission.to_csv('submission.csv', index=False)


# Submitting this results in a score of ~80. Currently, I'm working on implenting a gradient boosted tree which will hopefully produce better results. 
# 
# Please leave any suggestions you have to help me improve this kernel!
