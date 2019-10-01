#!/usr/bin/env python
# coding: utf-8

# * # 1. Data Exploration
# <br />
# **Goals:** <br />
# * Load dataframes <br />
# * Know the size of the dataframes <br />
# * Familiarize with the fields <br />
# * Identify the type of variables <br />
# * Perform basic statistic <br />

# こんにちは！パイソンミチコマンです。
# パイソンミチコマンも日々kaggleで与えられたテーマを見て機械学習プログラムの作成に挑戦しています。

# **ライブラリのインポート** <br />
# パイソンはライブラリが非常に充実している。このカーネルでは以下のライブラリを使用する。
# 
# ・matplotlib　グラフ描画ライブラリ。折れ線グラフ、ヒストグラム、散布図などの可視化方法がサポートされているライブラリ。
# Graph drawing library. A library that supports visualization methods such as line graphs, histograms, and scatter plots.
# 
# ・numpy　多次元配列機能や、線形代数やフーリエ変換、疑似乱数生成器などの、高レベルの数学関数が用意されているライブラリ。
# A library that provides high-level mathematical functions such as multidimensional array function, linear algebra, Fourier transform, pseudo random number generator.
# 
# ・pandas　データを変換し解析するためのライブラリ。
# A library for transforming and analyzing data.
# 
# ・math

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#first focusing on what the train dataset tells us


# **パイソンはライブラリが非常に充実している。このカーネルでは以下のライブラリを使用する。** <br />
# ・matplotlib　グラフ描画ライブラリ。折れ線グラフ、ヒストグラム、散布図などの可視化方法がサポートされているライブラリ。
# Graph drawing library. A library that supports visualization methods such as line graphs, histograms, and scatter plots.
# 
# ・numpy　多次元配列機能や、線形代数やフーリエ変換、疑似乱数生成器などの、高レベルの数学関数が用意されているライブラリ。
# A library that provides high-level mathematical functions such as multidimensional array function, linear algebra, Fourier transform, pseudo random number generator.
# 
# ・pandas　データを変換し解析するためのライブラリ。
# A library for transforming and analyzing data.

# In[ ]:


#1. The first five observations
#先頭５行のデータを確認する
train.head(5)


# In[ ]:


#dataframeのサイズを確認する
train.shape
#891のデータと12カラム


# In[ ]:


#12カラムの詳しい情報
train.info()


# In[ ]:


#4. How many missing values?
# Looks like we can't fully rely on Age and Cabin. We need to figure out a way on how to predict the values of these fields especially Age.
# Since Embarked only has 2 missing values we fill them in using mean, median, or mode
train.isnull().sum()


# In[ ]:


#5. Basic Statistics
#基本統計量の算出
train.describe()


# 

# In[ ]:


#6. Number of unique Values
#一意のデータがどれだけあるか調査
train.nunique()


# In[ ]:


train['Survived_C'] = np.where(train['Survived']==0,'No','Yes')
train.head()


# # 2.  Data Visualization
#  <br />
# 
# **More Male passengers survived than Females passengers**  <br />
#  60% of the passengers died, 53% Male and 9% Female  <br />
#  48% of the passengers survived, 12% Male and 26% Female
# 
# 74% Female survival rate  <br />
# 19% Male survival rate

# In[ ]:


#Breakdown of the passengers by gender
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,4))

train['Sex'].value_counts().plot.pie(autopct='%1.f%%', shadow=True, explode=(.1,0), startangle=90, ax=ax[0]).axis('equal')
ax[0].set_title('Passenger Breakdown Based on Sex')

ax1 = sns.countplot(x='Survived_C', hue='Sex', data=train, ax=ax[1])
total = float(len(train))
for p in ax1.patches:
    height = p.get_height()+1
    ax1.text(p.get_x()+p.get_width()/2.,
            height+3,
            '{:1.0f}%'.format((height/total)*100),
            ha="center")
ax1.set_title('Survival Count and Rate Breakdown')

plt.show()


# In[ ]:


sns.barplot(x='Sex', y='Survived', data=train).set_title('Survival Rate based on Sex')
plt.show()


# In[ ]:


train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


#女性の生存率約７４％
#男性の生存率約１８％


# **Point of Embarkation**
# 
# C = Cherbourg; Q = Queenstown; S = Southampton <br />
# *Southampton* embarked 72% of passengers.  <br />
# Visually we can see that Cherbourg is the only port that embarked more survivors than those who died. <br />
# Majority of the passengers embarked from Southampton and Queenstown mostly died.

# In[ ]:


#Filling in missing Embarked values
train[train.Embarked.isnull()]
#both are female Pclass1 survivors, paid $80, and travelled alone (although PassengerID 830 looks like have 2 Names registered?)


# In[ ]:


# Determining how many passengers per observation / per ticket
train['No_of_Passengers_on_Ticket']= train['SibSp'] + train['Parch'] + 1 #+1 for those travelling alone

# Adding a column called 'Group Size' to better segment the observation
# Solo - 1 traveller
# Couple - 2 travellers 
# Mid - 3 to 5 travellers
# Large - 6+ travellers 
train['Group_Size'] = np.where(train.No_of_Passengers_on_Ticket==1, 'Solo',
                                    np.where(train.No_of_Passengers_on_Ticket==2, 'Couple', 
                                             np.where(np.logical_and(train.No_of_Passengers_on_Ticket>2, train.No_of_Passengers_on_Ticket<6),'Mid',
                                                      'Large')))
train[train.Embarked.isnull()]


# In[ ]:


#We know that the missing values are Female passengers, Pclass 1, fare=$80, both travelling alone
Pclass1 = train[(train['Sex']=='female') & (train['Pclass']==1) & (train['Group_Size']=='Solo')]
#Pclass1[['Embarked', 'Fare']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Fare', ascending=False)
Pclass1[['Embarked', 'Fare']].groupby(['Embarked'], as_index=False).describe()


# In[ ]:


#The mean closest to $80 is Southampton. It also has the smaller standard deviation 
Pclass1[['Embarked', 'Fare']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Fare', ascending=False)


# In[ ]:


train.Embarked.fillna('S', inplace=True) #filling in missing Embarked with S


# In[ ]:


f, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,4))

sns.countplot(x='Survived_C', hue='Embarked', data=train, ax=ax[1]).set_title('Survival Count based on Port of Embarkation')

plt.show()


# In[ ]:


train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()


# **Passenger Class**
# 
# *Pclass* is a proxy for socio-economic status (SES)
# 1st ~ Upper;  2nd ~ Middle;  3rd ~ Lower
# 
# 55% of the passengers were considered Pclass 3 (more poor people than rich people) but it also has the lowest survival rate at 24% <br />
# For both genders, Pclass 1 passengers have the highest survival rate at 63% <br />

# In[ ]:


f, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,4))

colors=['forestgreen','steelblue', 'darkorange']

train['Pclass'].value_counts().plot.pie(autopct='%1.1f%%', shadow=True, explode = (0.1,0,0), startangle=90, colors=colors, ax=ax[0]).axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
ax[0].set_title('Proportion of Passengers Per Pclass')

sns.barplot(x='Pclass', y='Survived', ax=ax[1], data=train).set_title('Survival Rate based on Pclass')

plt.show()


# **Pclass and Gender**

# In[ ]:


sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train).set_title('Survival Rate Per Gender and Pclass')
plt.show()


# In[ ]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# **Pclass and Embarked**

# In[ ]:


sns.countplot(x='Embarked', hue='Pclass', data=train).set_title('Survival Count based on Port of Embarkation and Pclass')
plt.show()


# 
# **Group Size of the Survivors**
# 
# Recap of the added Group Size Column:  <br />
# Solo - 1 traveller <br />
# Couple - 2 travellers <br />
# Mid - 3 to 5 travellers <br />
# Large - 6+ travellers <br />
# 
# People who mostly survived are ethe ones travelling alone <br />
# People who travelled in 2 - 4 are the ones who mostly didn't survive<br />

# In[ ]:


data = train.sort_values(['No_of_Passengers_on_Ticket'], ascending=True)

f, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,4))

sns.countplot(x='Group_Size', hue='Survived_C', ax=ax[0], data=data).set_title('Survival Count based on Group Size')
sns.countplot(x='No_of_Passengers_on_Ticket', hue='Survived_C', ax=ax[1], data=data).set_title('Survival Count based on Group Size')
plt.show()


# **Age **
# 
# Since Age has 177 NaN values, we are going to fill in the missing values using Mean Imputation. 

# In[ ]:


train['Age'].fillna(train.Age.mean(), inplace=True)
train.describe()


# Most survivors are in their 30s but it's interesting to see that there's a spike around 1 to 2 years of age. There are also infants and survivors over 80 years old.
# 
# For those who died, it follows pretty much the same curve as the Survivors distribution, wherein lots of thirty year old passengers died.

# In[ ]:


f, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,4))

Gender = ['male', 'female']
Survive = ['Yes', 'No']

for g in Gender:
    survivors = train[(train['Sex']==g) & (train['Survived_C']=='Yes')].Age
    sns.distplot(survivors, hist=False, label=g, ax=ax[0]).set_title('Age Distribution of Survivors')
    deaths = train[(train['Sex']==g) & (train['Survived_C']=='No')].Age        
    sns.distplot(deaths, hist=False, label=g, ax=ax[1]).set_title('Age Distribution of those Who Died')  

plt.show()


# **Age and Group Size** <br />
# Most 30 year olds travelled alone or in two.

# In[ ]:


f, ax = plt.subplots(nrows=1,ncols=2,figsize=(15,4))

sns.pointplot(y='Age', x='No_of_Passengers_on_Ticket', hue='Survived_C', linestyles=['--','-'], markers=['x','o'],               dodge=True, data=train, ax=ax[0]).set_title('Distribution by Age and Passenger Count')
sns.pointplot(y='Age', x='Group_Size', hue='Survived_C', linestyles=['--','-'], markers=['x','o'],               dodge=True, data=train, ax=ax[1]).set_title('Distribution by Age and Group Size')

plt.show()


# **Age and Pclass** <br />
# Most 30 year olds travelled alone or in two.

# In[ ]:


sns.boxplot(y='Age',x='Pclass', data=train).set_title('Age and Pclass')


# **From Visualizing the data, we have observed:** <br />
# 1. There are twice as much male (65%) than female (35%) passengers <br />
# 2. More female survivors than their male counterparts (almost doubled). <br />
# 3. More male deaths than femle (4x more). <br />
# 4. Southampton embarked the most passengers but is also the port with lowest survival rate <br />
# 5. Cherbourg embarked the second largest number of passengers but also have passengers with the highest survival rate.  <br />
# 6. Most survivors travelled alone and in two's and most of them are 30 years olds. <br />
# 7. Socio-economic status played part in survival, having Pclass 1 passengers with the highest surival rate at 63%

# # 3. Data Wrangling
# * Converting Strings into Categorical Values
# * Cleaning / Removing Categorical Values

# In[ ]:


#Dropping 
train.drop(['Survived_C','Cabin'], axis=1, inplace=True)
train.head()


# In[ ]:


Sex = pd.get_dummies(train['Sex'], drop_first=True)
Embarked_New = pd.get_dummies(train['Embarked'], drop_first=True)
Pclass_New = pd.get_dummies(train['Pclass'], drop_first=True)
train_n = pd.concat([train,Sex,Embarked_New,Pclass_New],axis=1)
train_n.drop(columns=['Sex','Pclass','Embarked','Name','PassengerId','Ticket','Group_Size'],axis=1, inplace=True)


# In[ ]:


train_n.rename(columns={'male':'Sex', 'Q':'Queenstown', 'S':'Southampton',2:'Pclass2',3:'Pclass3'}, inplace=True)
train_n.head()


# # 4. Predictions

# **Logistic Regression**
# 
# In logistic regression, the outcome (dependent variable) has only a limited number of possible values. It is used when the response variable is categorical in nature -- which is what exactly what we are trying to predict, whether a passenger will survive or not.
# 

# In[ ]:


# Preparing the data
X = train_n.drop(['Survived'], axis=1)
y = train_n.Survived


# In[ ]:


#building the logistic regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)
logmodel = LogisticRegression()
#fitting the model
logmodel.fit(X_train, y_train)


# In[ ]:


#make predictions
prediction = logmodel.predict(X_test)


# In[ ]:


# Examining the Prediction
#calculate precision and recall
from sklearn.metrics import classification_report
classification_report(y_test,prediction)


# In[ ]:


#look at the confusion matrix to justify Precision and Recall
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction)


# **Observation:**
# We got 78% Precision and Recall -- which is pretty good. Although I think this can still be improved.
# <br />
# *Precision* is the number of True Positives divided by the number of True Positives and False Positives <br />
# *Recal*l is the number of True Positives divided by the number of True Positives and the number of False Negatives. <br />
# 
# 134 - True Positive <br />
# 75 - True Negative <br />
# 19 - False Positive (Type 1 Error) <br />
# 40 - False Negative (Type 2 Error) <br />
# <br />
# *For any system to be able to achieve maximum Precision (no false positive) and maximum Recall (no false negative) there needs to be an absence of type I and II errors.

# In[ ]:


#calculate the accuracy score - from the confusion matrix
#Number of correct predictions
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)
#we have 77% accuracy, which is still pretty good


# **Using the Test Data**

# In[ ]:


#Preparing the data
Sex = pd.get_dummies(test['Sex'], drop_first=True)
Emb = pd.get_dummies(test['Embarked'], drop_first=True)
Pcl = pd.get_dummies(test['Pclass'], drop_first=True)
test_n = pd.concat([test, Sex, Emb, Pcl], axis=1)

#Matching the train data's column labels
test_n.drop(columns=['PassengerId', 'Ticket', 'Pclass', 'Name', 'Sex', 'Cabin', 'Embarked'], axis=1, inplace=True)
test_n['No_of_Passengers_on_Ticket'] = test_n.SibSp + test_n.Parch + 1


# In[ ]:


test_n.rename(columns={'male':'Sex', 'Q':'Queenstown', 'S':'Southampton',2:'Pclass2',3:'Pclass3'}, inplace=True)
test_n.head()

#checking if the test data has null values
#test_n.isnull().sum()

#Dropping the null values of the test data
X_n_test = test_n.dropna(how='any')


# In[ ]:


test_n.shape


# In[ ]:


X_n_test.shape


# In[ ]:


#X_n_test is still a decent sample size, we still have 79% of the actual test data
331/418


# In[ ]:


# Preparing the data
X_train = train_n.drop(['Survived'], axis=1)
y_train = train_n.Survived

#building the logistic regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Train data has a 891 observations, while Testa data has 331 observations
#Using 331/891 = 37% of the Train data as the test size, so that I can compare and plug the Test data later on
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3714, random_state=1)
logmodel = LogisticRegression()

#fitting the model
logmodel.fit(X_train, y_train)


# In[ ]:


#predict from the train set
prediction = logmodel.predict(X_test)


# In[ ]:


#Examining the Prediction
#calculate precision and recall
from sklearn.metrics import classification_report
classification_report(y_test,prediction)


# In[ ]:


#calculate the accuracy score - from the confusion matrix
#Number of correct predictions
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)
#we have 77% accuracy, which is still pretty decent


# In[ ]:


#predict from the test set
prediction2 = logmodel.predict(X_n_test)


# In[ ]:


# Examining the Prediction
#calculate precision and recall
from sklearn.metrics import classification_report
classification_report(y_test,prediction2)


# In[ ]:


#look at the confusion matrix to justify Precision and Recall
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction2)


# In[ ]:


#calculate the accuracy score - from the confusion matrix
#Number of correct predictions
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction2)
#we have 50% accuracy, it's not good... need to improve the prediction model


# # 5. Conclusion
# 
# It's a fun exercise to visualize the data. I'm getting familiar with Kaggle. Although I wish to have better prediction, I'll keep looking for ways to improve my prediction and update this kernel from time to time.
