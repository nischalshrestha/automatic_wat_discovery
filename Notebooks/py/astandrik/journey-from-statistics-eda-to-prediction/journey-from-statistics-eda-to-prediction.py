#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stat
get_ipython().magic(u'matplotlib inline')
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# *IMPORT DATASET**
# 
# 1st of all, let's import the dataset

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.shape, test.shape


# **A. DESCRIPTIVE STATISTICS**
# 
# All right, using few lines of code, let's try to describe the data using desctiptive statistics

# In[ ]:


train.describe()


# In[ ]:


test.describe()


# **A.1 Numerical Attributes**
# 
# From above simple code, we can see some numerical attributes described by some simple descriptive statistics. **What do we get here?**
# 1. **Survived**: the sample mean of this training data is 0,38, which could means *only about that percentage of passengers survived from titanic accident*
# 
# 2. **Pclass** (Passenger Class: there are 3 class of passenger. At Q2(50%) and Q3(75%) we could see the value is 3, which could means *there are minimum 50% (or more) passengers which is 3rd class passengers*. It seems logical since lower class usually have cheaper ticket prize, and more quota for that class
# 
# 3. **Age**: from train and test data, the count values seems different from the others. yes, **Age attribute contains missing values**. Another usefull information, the mean/average age on training data is 29 years old, which is 1 years older than the median value of the mean (30 mean and 27 median on test dataset), so what does it mean?
#     
#     it means the distributions of age values have **right skew**, which we expect some outliers in the *higher age value* (on the right size ofthe axis. As we can see, on the training and test dataset max value is 80 and 76 respectively.
#     
# 4. **SibSp and Parch**: these attributes indicate number of SIblings or spouses, and Parent or Children number aboard. From the mean value, seems *majority of the passengers is alone (neither have SibSp or Parch)*. It is interesting that we see the maximum value have 8 SibSp and 9 ParCh, *maybe the oldest person brought his/her entire family on the ship*
# 
# 5. **Fare**: there are huge difference between mean and median value of this attributes, which is logical. *Many passengers from 3rd class which always have lower Fare*, on the other hand, we have so high value on max of Fare here, which seems an outlier that affect the average of this attributes (**again, right skew**). **Fare attribute contain 1 missing value on test dataset**

# In[ ]:


train.describe(include=['O'])


# In[ ]:


test.describe(include=['O'])


# **A.2 Categorical Attributes**
# Now, we're dealing with categorical attributes. From describe method above, we get some new information:
# 1. **Name**: all names are unique (nothing special), *but they contains title*. maybe we can do some feature engineering later to get new attributes which could improve our prediction later.
# 
# 2.  **Sex**: or *gender*. consist of 2 categories, male and female, with both on training and test dataset, male have higher frequency (approximately 60 : 40)
# 
# 3.  **Ticket**: soooo many unique values for this attributes. Maybe I'll just drop this attribute for now and include it for future research
# 
# 4. **Cabin**: so many **missing values** here (*204 filled from 891 possible* on training dataset and *91 filled from 418 possible* on test dataset). *Maybe some passengers*, which we already know, 3rd class or some low Fare paid passenger, **don't have Cabin**.
# 
# 5. **Embarked**: There are **2 missing values** on training dataset. from train and test dataset, we know that most of Passengers embarked from S (*what's this "S" anyway?*).

# **B. EXPLORATORY DATA ANALYSIS**

# In[ ]:


train.head()


# In[ ]:


f,ax = plt.subplots(3,4,figsize=(20,16))
sns.countplot('Pclass',data=train,ax=ax[0,0])
sns.countplot('Sex',data=train,ax=ax[0,1])
sns.boxplot(x='Pclass',y='Age',data=train,ax=ax[0,2])
sns.countplot('SibSp',hue='Survived',data=train,ax=ax[0,3],palette='husl')
sns.distplot(train['Fare'].dropna(),ax=ax[2,0],kde=False,color='b')
sns.countplot('Embarked',data=train,ax=ax[2,2])

sns.countplot('Pclass',hue='Survived',data=train,ax=ax[1,0],palette='husl')
sns.countplot('Sex',hue='Survived',data=train,ax=ax[1,1],palette='husl')
sns.distplot(train[train['Survived']==0]['Age'].dropna(),ax=ax[1,2],kde=False,color='r',bins=5)
sns.distplot(train[train['Survived']==1]['Age'].dropna(),ax=ax[1,2],kde=False,color='g',bins=5)
sns.countplot('Parch',hue='Survived',data=train,ax=ax[1,3],palette='husl')
sns.swarmplot(x='Pclass',y='Fare',hue='Survived',data=train,palette='husl',ax=ax[2,1])
sns.countplot('Embarked',hue='Survived',data=train,ax=ax[2,3],palette='husl')

ax[0,0].set_title('Total Passengers by Class')
ax[0,1].set_title('Total Passengers by Gender')
ax[0,2].set_title('Age Box Plot By Class')
ax[0,3].set_title('Survival Rate by SibSp')
ax[1,0].set_title('Survival Rate by Class')
ax[1,1].set_title('Survival Rate by Gender')
ax[1,2].set_title('Survival Rate by Age')
ax[1,3].set_title('Survival Rate by Parch')
ax[2,0].set_title('Fare Distribution')
ax[2,1].set_title('Survival Rate by Fare and Pclass')
ax[2,2].set_title('Total Passengers by Embarked')
ax[2,3].set_title('Survival Rate by Embarked')


# > Some usefull information:
# * clearly, we can see most passengers are in class 3, which have least survival probability here
# * from Sex attribute, we can see total male Passengers is almost 2 times of female passengers, but lower survival probability *maybe male passengers tend to save their lady first?*
# * from above figure, we can try to input missing age by class
#     * Pclass 1, Age average approximately = 37
#     * Pclass 2, Age average approximately = 29
#     * Pclass 3, Age average approximately = 24
# * also on age attributes, we already clearly see the age distributions follow normal distribution with right skew
# * it seems passenger with Sibling/Spouse, or have parent/children aboard, have higher survival rate than passenger which is alone!

# In[ ]:


train['Cabin'].value_counts().head()


# Now we got new information, some passenger have multiple cabin listed.
# for each passenger, I'll just try to create a new feature called **'Deck'** with first letter from the Cabin as its value.
# if passenger have multiple deck listed, I'll just use the higher class deck (ex: A and D, I'll just use A as the value)
# 
# thanks to this discussion: https://www.kaggle.com/c/titanic/discussion/4693
# 
# "first class had the top decks (A-E), second class (D-F), and third class (E-G). It also makes sense that the people towards the top (higher decks, higher pclass) more likely survived, because they were closer to the lifeboats."

# In[ ]:


g = sns.FacetGrid(col='Embarked',data=train)
g.map(sns.pointplot,'Pclass','Survived','Sex',palette='viridis')
g.add_legend()


# We got a lot information from above visualization, such as:
# * Female passenger Embarked from S and Q have high survival rate
# * Female from class 1 and 2 Embarked from Queenstown absolutely survived!!!
# * Male Embarked from Queenstown with Pclass 1, have lowest survival rate!
# * Male Embarked from Cherbourg with Class 1 and 2 have high survival rate too.

# **C. SUMMARY**
# 
# Before we do this journey further, let's summarize our information and so far and what we should do with them.
# * **Survived:**
#     * The value we should predict using test dataset. It is numerical with binary value 0 (Dead) and 1 (Survived)
#     
# * **Pclass:**
#     * The data type is categorical, level of measurement is qualitative->ordinal, since the level seems like 1>2>3.
#     * Since this is a categorical, maybe we should **get dummies** of this variable.
#     
# * **Name:**
#     * The data type is categorical, level of measurement is qualitative->nominal.
#     * We should include this variable in **Feature Engineering** process to extract the title value which maybe could improve our prediction result.
#     
# * **Sex:**
#     * The data type is categorical, level of measurement is qualitative->nominal.
#     * Since this is a categorical, maybe we should change the value to binary value 0 for male and 1 for female. We'll do this on **Data Preparation** process.
#     
# * **Age:**
#     * The data type is numerical->continous with level of measurement is quantitative->ratio.
#     * we should fill the **missing values**
#     * in order to divide the train data so the machine learning could understand better, I prefer to change the level of measurement to quantitative->interval using the group of age (maybe child, teenagers, young adult, adult) on **Feature Engineering** process.
#     
# * **SibSp & Parch:**
#     * The data type is numerical, level of measurement is quantitative->ratio.
#     * Passenger with Sibling/Spouse, or have parent/children aboard, have higher survival rate than passenger which is alone!
#     * So I'll create a new feature based on this attribute called 'is_alone', I'll do this on **Feature engineering** process.
#     
# * **Ticket:**
#     * *I'' drop this for now.*
#     
# * **Fare:**
#     * The data type is numerical->continous with level of measurement is quantitative->ratio.
#     * There is 1 missing value in test dataset
#     * in order to divide the train data so the machine learning could understand better, I prefer to change the level of measurement to quantitative->interval using the group of Fares (maybe low Fare, Medium Fare, and High Fare or something like that) on **Feature Engineering** process.
#     
# * **Cabin:**
#     * The data type is categorical, level of measurement is qualitative->ordinal, since the level seems like A>B>C>D..
#     * Some passenger have multiple cabin listed.
#     * there are many **missing values** on this attributes, I'll fill it with 'No Cabin' string.
#     * for each passenger, I'll just try to create a new feature called **'Deck'** with first letter from the Cabin as its value on **Feature Engineering** process.
#     * if passenger have multiple deck listed, I'll just use the higher class deck (ex: A and D, I'll just use A as the value)
#     
# * **Embarked:**
#     * The data type is categorical, level of measurement is qualitative->nominal.
#     * Since this is a categorical, maybe we should **get dummies** of this variable.
#     * there are 2 missing values on training dataset

# **D. DEALING WITH MISSING VALUES**
# 
# from the summary above, we should fill missing values in **Age**, 1 value of **Fare** in test, and 2 values of **Embarked** in training. So, let's do this.
# 
# wait, let's check the missing values using heatmap.

# In[ ]:


f,ax = plt.subplots(1,2,figsize=(15,3))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[0])
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[1])


# **D.1 Filling missing values in Age**
# 
# I'll try to input missing Age by Pclass:
# * Pclass 1, Age average approximately = 37
# * Pclass 2, Age average approximately = 29
# * Pclass 3, Age average approximately = 24

# In[ ]:


def fill_age(cols):
    Age = cols[0]
    PClass = cols[1]
    
    if pd.isnull(Age):
        if PClass == 1:
            return 37
        elif PClass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(fill_age,axis=1)
test['Age'] = test[['Age','Pclass']].apply(fill_age,axis=1)


# **D.2 Filling missing values in Fare, Cabin and Embarked**

# In[ ]:


test['Fare'].fillna(stat.mode(test['Fare']),inplace=True)
train['Embarked'].fillna('S',inplace=True)
train['Cabin'].fillna('No Cabin',inplace=True)
test['Cabin'].fillna('No Cabin',inplace=True)


# In[ ]:


f,ax = plt.subplots(1,2,figsize=(15,3))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[0])
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[1])


# **IT'S CLEAR!!!** ready for feature engineering, but, we'll drop Ticket first

# In[ ]:


train.drop('Ticket',axis=1,inplace=True)
test.drop('Ticket',axis=1,inplace=True)


# In[ ]:


train.head()


# **E. FEATURE ENGINEERING**
# 
# from summary above, we will do some work on **Name, Age, SibSP & Parch, Fare, Cabin**. Let's do this!

# In[ ]:


#combine dataset 1st for easier Feature Engineering
train['IsTrain'] = 1
test['IsTrain'] = 0
df = pd.concat([train,test])


# **E.1 Feature Engineering: Name -> Title**

# In[ ]:


df['Title'] = df['Name'].str.split(', ').str[1].str.split('.').str[0]
df['Title'].value_counts()


# for these rare title, we'll convert them to 'Others', except **Mme** will be converted to Mrs, **Ms and Mlle** to Miss

# In[ ]:


df['Title'].replace('Mme','Mrs',inplace=True)
df['Title'].replace(['Ms','Mlle'],'Miss',inplace=True)
df['Title'].replace(['Dr','Rev','Col','Major','Dona','Don','Sir','Lady','Jonkheer','Capt','the Countess'],'Others',inplace=True)
df['Title'].value_counts()


# In[ ]:


df.drop('Name',axis=1,inplace=True)
df.head()


# **E.2 Feature Engineering: Age -> AgeGroup**

# In[ ]:


sns.distplot(df['Age'],bins=5)


# I'll divide age to 5 categories, they are, Child(<=19), Young Adult(>19,<=30), Adult(>30,<=45), Old(>45,<=63), Veteran(>63), 
# 
# with: **child = 0, Young Adult = 1, Adult = 2, Old = 3, Veteran = 4**

# In[ ]:


df['AgeGroup'] = df['Age']
df.loc[df['AgeGroup']<=19, 'AgeGroup'] = 0
df.loc[(df['AgeGroup']>19) & (df['AgeGroup']<=30), 'AgeGroup'] = 1
df.loc[(df['AgeGroup']>30) & (df['AgeGroup']<=45), 'AgeGroup'] = 2
df.loc[(df['AgeGroup']>45) & (df['AgeGroup']<=63), 'AgeGroup'] = 3
df.loc[df['AgeGroup']>63, 'AgeGroup'] = 4


# In[ ]:


sns.countplot(x='AgeGroup',hue='Survived',data=df[df['IsTrain']==1],palette='husl')


# In[ ]:


df.drop('Age',axis=1,inplace=True)
df.head()


# **E.3 Feature Engineering: SibSp & Parch -> IsAlone**

# In[ ]:


df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 #himself
df['IsAlone'] = 0
df.loc[df['FamilySize']==1, 'IsAlone'] = 1


# In[ ]:


#checking correlation with survival rate
f,ax = plt.subplots(1,2,figsize=(15,6))
sns.countplot(df[df['IsTrain']==1]['FamilySize'],hue=train['Survived'],ax=ax[0],palette='husl')
sns.countplot(df[df['IsTrain']==1]['IsAlone'],hue=train['Survived'],ax=ax[1],palette='husl')


# from both figures, I can assume that if a passenger have family onboard, **the survival rate will increase to approximately 50%.**
# 
# because we already have the information using is_alone feature only, *I'll just drop SibSp, Parch, and FamilySize*

# In[ ]:


df.drop(['SibSp','Parch','FamilySize'],axis=1,inplace=True)
df.head()


# **E.4 Feature Engineering: Fare -> FareGroup**

# In[ ]:


f,ax = plt.subplots(1,2,figsize=(16,4))
sns.distplot(df['Fare'],bins=10,ax=ax[0])
sns.swarmplot(x='Pclass',y='Fare',data=df[df['IsTrain']==1],hue='Survived',ax=ax[1],palette='husl')


# In[ ]:


df['FareGroup'] = df['Fare']
df.loc[df['FareGroup']<=50,'FareGroup'] = 0
df.loc[(df['FareGroup']>50) & (df['FareGroup']<=100),'FareGroup'] = 1
df.loc[(df['FareGroup']>100) & (df['FareGroup']<=200),'FareGroup'] = 2
df.loc[(df['FareGroup']>200) & (df['FareGroup']<=300),'FareGroup'] = 3
df.loc[df['FareGroup']>30,'FareGroup'] = 4


# In[ ]:


sns.countplot('FareGroup',data=df[df['IsTrain']==1],hue='Survived',palette='husl')


# In[ ]:


df.drop('Fare',axis=1,inplace=True)
df.head()


# **E.5 Feature Engineering: Cabin -> Deck**

# In[ ]:


df['Deck'] = df['Cabin']
df.loc[df['Deck']!='No Cabin','Deck'] = df[df['Cabin']!='No Cabin']['Cabin'].str.split().apply(lambda x: np.sort(x)).str[0].str[0]
df.loc[df['Deck']=='No Cabin','Deck'] = 'N/A'


# In[ ]:


sns.countplot(x='Deck',hue='Survived',data=df[df['IsTrain']==1],palette='husl')


# Well, now we can see clearly the survival rate based on passenger's Deck

# In[ ]:


df.drop('Cabin',axis=1,inplace=True)
df.head()


# **F. FINAL DATA PREPARATION**
# 
# now after we got the features, lastly on data preprocessing, we need to get dummies on categorical data based on newly fresh baked dataframe, they are: **Embarked, Sex, Pclass, Title, AgeGroup, FareGroup, Deck**. 

# In[ ]:


def process_dummies(df,cols):
    for col in cols:
        dummies = pd.get_dummies(df[col],prefix=col,drop_first=True)
        df = pd.concat([df.drop(col,axis=1),dummies],axis=1)
    return df


# In[ ]:


df = process_dummies(df,['Embarked','Sex','Pclass','Title','AgeGroup','FareGroup','Deck'])


# In[ ]:


df.head()


# Now All Set. Before we continue to prediction section, let's divide again our data to **dataset** (formerly train data) and **holdout** (formerly test data)

# In[ ]:


dataset = df[df['IsTrain']==1]
dataset.drop(['IsTrain','PassengerId'],axis=1,inplace=True)
holdout = df[df['IsTrain']==0]
test_id = holdout['PassengerId']
holdout.drop(['IsTrain','PassengerId','Survived'],axis=1,inplace=True)


# **G. PREDICTION**
# 
# In this section, I'll do some work starts from splitting the dataset and do *cross_validation* with it, and maybe some *parameter tuning* to get better prediction result. Stay tune!!!

# **G.1 Splitting the dataset**

# In[ ]:


X = dataset.drop(['Survived'],axis=1)
y = dataset['Survived'].astype('int')
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)


# **G.2 Cross Validation**
# 
# Here, I'll do cross validation using 3 classifiers, SVM, Random Forest, and Logistic Regression. let's see which one give better result.

# In[ ]:


svc = SVC()
rfc = RandomForestClassifier()
lgr = LogisticRegression()


# In[ ]:


kfold = KFold(n_splits=10, random_state=7)
print('SVM Classifier: ',cross_val_score(svc, X_train, y_train, cv=kfold).mean()*100)
print('Random Forest: ',cross_val_score(rfc, X_train, y_train, cv=kfold).mean()*100)
print('Logistic Regression: ',cross_val_score(lgr, X_train, y_train, cv=kfold).mean()*100)


# > **Result:** Seems Logistic Regression works well here, let's try to submit our 1st prediction using Logistic Regression

# In[ ]:


#lgr.fit(X,y)
#predictions = lgr.predict(holdout) -- We got only 0.76 on submission score, need to improve


# Let's try using Random Forest with n_estimators=1 and 1000

# In[ ]:


#rfc.fit(X,y)
#predictions = rfc.predict(holdout) -- Woa, we got 0.78947 on submission score, we should improve the score
#rfc = RandomForestClassifier()
#rfc.fit(X,y)
#predictions = rfc.predict(holdout) -- 0.799 on submission score.


# **G.3 XGBoost**
# 
# How about using eXtreme Gradient Boosting?

# In[ ]:


#xgb = XGBClassifier()
#xgb.fit(X,y)
#predictions = xgb.predict(holdout) -- we only get 0.77033


# Don' give up, let's try using common parameters based on this blog post: https://machinelearningmastery.com/configure-gradient-boosting-algorithm/

# In[ ]:


xgb = XGBClassifier(n_estimators=500)
xgb.fit(X,y)
predictions = xgb.predict(holdout)


# **H. SUBMISSION**
# 
# Finally, let's submit our result

# In[ ]:


submission = pd.DataFrame({
    'PassengerId': test_id,
    'Survived': predictions
})

submission.to_csv('submission.csv',index=False)


# **MORE TO COME!!!**

# In[ ]:




