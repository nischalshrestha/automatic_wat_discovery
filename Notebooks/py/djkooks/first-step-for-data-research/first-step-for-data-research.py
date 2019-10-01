#!/usr/bin/env python
# coding: utf-8

# ## First research
# 
# This is the first step for me of studying data research.
# 
# Because I'm doing this as beginner, there are lot of parts following/referring other kernels...
# - especially https://www.kaggle.com/ash316/eda-to-prediction-dietanic
# - https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner
# - https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# 
# , and some references in web(datacamp tutorial). Thanks to all!
# 
# Most of part in this note is focused on visualizing, and data pre-processing. I'll not focus on logic or algorithm now. It is too complecate to me...

# In[ ]:


# data analysis 
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# hide useless warnings...
import warnings
warnings.filterwarnings('ignore')

# import data to pandas instance from csv file
train_df = pd.read_csv('../input/train.csv')  # training dataframe
test_df  = pd.read_csv('../input/test.csv')   # test dataframe
train_df.head()


# In[ ]:


# There is 891 observations and 12 variables 
train_df.shape
train_df.describe()


# Now, check if there is null value in this table. It needs to be removed before process.

# In[ ]:


train_df.isnull().sum()


# Survived passenger is set as '1', while dead are '0'.
# <br/>In train data, 61.6% has been dead, and survival rate of woman is almost 75% while man are only 19%.
# 

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train_df['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=train_df,ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train_df['Survived'][train_df['Sex']=='male'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
train_df['Survived'][train_df['Sex']=='female'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[0].set_title('Survived (male)')
ax[1].set_title('Survived (female)')

plt.show()


# In[ ]:


# 549 died, 342 survived in train data. Proportion is 61.6162% | 38.3838%
print(train_df['Sex'].value_counts(normalize=True))
train_df.groupby(['Sex','Survived'])['Survived'].count()


# 'Pclass' is class of ticket. You can see survival rate of passengers with Pclass 1 ticket is more higher than Pclass 3, though there are much more people.

# In[ ]:


pd.crosstab([train_df['Sex'],train_df['Survived']],train_df['Pclass'],margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train_df['Pclass'].value_counts().plot.pie(explode=[0,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Number Of Passengers By Pclass')
sns.countplot('Pclass',hue='Survived',data=train_df,ax=ax[1])
ax[1].set_title('Survived vs Dead by Pclass')
plt.show()


# In[ ]:


sns.factorplot('Pclass','Survived',hue='Sex',data=train_df)
plt.show()


# In plot above, it seems more than 90% or female and 40% of male in Pclass 1 has been survived, and passenger's survival rate of other class is much lower than this.
# <br/> In these charts, you can figure out `Sex` and `Pclass` are important features, pretty related with survival rate.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Sex","Age", hue="Survived", data=train_df,split=True,ax=ax[0])
ax[0].set_title('Sex and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Pclass","Age", hue="Survived", data=train_df,split=True,ax=ax[1])
ax[1].set_title('Pclass and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,10))
train_df[train_df['Survived']==0]['Age'].plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
train_df[train_df['Survived']==1]['Age'].plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()


# You can figure out lots of toddlers(under 5), and oldest people(more than 75) has been survived.

# Check the data with Embarked place.
# * C = Cherbourg
# * Q = Queenstown
# * S = Southampton

# In[ ]:


pd.crosstab([train_df['Embarked'],train_df['Pclass']],[train_df['Sex'],train_df['Survived']],margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:


f,ax=plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked',data=train_df,ax=ax[0,0])
ax[0,0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked',hue='Sex',data=train_df,ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked',hue='Survived',data=train_df,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked',hue='Pclass',data=train_df,ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
# plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


# ## Create decission tree model with some categorial data - Sex, Pclass, Embarked
# 
# I'll try to make data model via Decission tree, with some categorial data - Sex, Pclass, Embarked - to compare with result after  data pre-processing with other factors.
# 
# To work with data, null value needs to be removed, and string data needs to be changed as integer.

# In[ ]:


train_df['Embarked'].fillna('S',inplace=True)
test_df['Embarked'].fillna('S',inplace=True)


# In[ ]:


# reform all 'string' data to 'integer'
train_df['Sex'].replace(['male','female'],[0,1],inplace=True)
test_df['Sex'].replace(['male','female'],[0,1],inplace=True)

train_df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
test_df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure


## prediction with logistic regression
train, test = train_test_split(train_df,test_size=0.3,random_state=0)
target_col = ['Pclass', 'Sex', 'Embarked']

train_X=train[target_col]
train_Y=train['Survived']
test_X=test[target_col]
test_Y=test['Survived']

features_one = train_X.values
target = train_Y.values


# In[ ]:


tree_model = DecisionTreeClassifier()
tree_model.fit(features_one, target)
dt_prediction = tree_model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(dt_prediction, test_Y))

# test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
test_features = test_df[target_col].values

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test_df["PassengerId"]).astype(int)
# my_prediction = my_tree_one.predict(test_features)
dt_prediction_result = tree_model.predict(test_features)
dt_solution = pd.DataFrame(dt_prediction_result, PassengerId, columns = ["Survived"])
# print(my_solution)

# Check that your data frame has 418 entries
print(dt_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
dt_solution.to_csv("my_solution_two.csv", index_label = ["PassengerId"]) 


# ## Data pre-processing
# Now I'll go on for processing other factors to find out more relationship between data.

# In[ ]:


train_df['Title'] = train_df['Name'].str.extract('([A-Za-z]+)\.')
pd.crosstab(train_df['Title'], train_df['Sex'])


# These are the titles, included in name. It seems it can be grouped with this. So let's group as
# 
# * 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona' => Rare
# * 'Countess', 'Lady', 'Sir' => Royal
# * 'Mlle', 'Ms', 'Lady' => Miss
# * 'Mme' => Mrs
# * and leave others as now
# 
# ...and update 'Title' column with this.
# And if you see the chart, it sees survival rate of 'Mrs', 'Miss', 'Royal' and 'Master' is higher than 'Rare' and 'Mr'.

# In[ ]:


train_df['Title'] = train_df['Title'].replace(['Capt', 'Col',
'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

train_df['Title'] = train_df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
train_df['Title'] = train_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# pd.crosstab(train_df['Title'], train_df['Sex'])


# ### Make group of Age
# Age is a continuous value, so it needs to be categorial to use. There will be lots of ways to do it, and I'll define as
# 
# - less than 7 : Baby & Kids
# - 8~20: Students
# - 21~30: Young Adults  
# - 31~40: Adults  
# - 41~60: Seniors
# - More than 60: Elders
# 
# But before doing this, we saw that 177 of data has no age value, so it needs to be filled. You can just set all as mean or median value, but in this part it will use 'title' column for this.

# In[ ]:


train_df.groupby('Title')['Age'].mean()


# This is the mean value for each title. Ususally title has relations with how old/young people are, so let's setup mean value matching with passenger's title.

# In[ ]:


train_df.loc[(train_df['Age'].isnull())&(train_df['Title']=='Master'),'Age'] = 5
train_df.loc[(train_df['Age'].isnull())&(train_df['Title']=='Miss'),'Age'] = 22
train_df.loc[(train_df['Age'].isnull())&(train_df['Title']=='Mr'),'Age'] = 33
train_df.loc[(train_df['Age'].isnull())&(train_df['Title']=='Mrs'),'Age'] = 36
train_df.loc[(train_df['Age'].isnull())&(train_df['Title']=='Rare'),'Age'] = 45
train_df.loc[(train_df['Age'].isnull())&(train_df['Title']=='Royal'),'Age'] = 43


# 

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,10))
train_df[train_df['Survived']==0]['Age'].plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
train_df[train_df['Survived']==1]['Age'].plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()


# In[ ]:


train_df['AgeGroup'] = 0
train_df.loc[ train_df['Age'] <= 7, 'AgeGroup'] = 0
train_df.loc[(train_df['Age'] > 7) & (train_df['Age'] <= 18), 'AgeGroup'] = 1
train_df.loc[(train_df['Age'] > 18) & (train_df['Age'] <= 30), 'AgeGroup'] = 2
train_df.loc[(train_df['Age'] > 30) & (train_df['Age'] <= 40), 'AgeGroup'] = 3
train_df.loc[(train_df['Age'] > 40) & (train_df['Age'] <= 60), 'AgeGroup'] = 4
train_df.loc[ train_df['Age'] > 60, 'AgeGroup'] = 5
pd.crosstab(train_df['AgeGroup'], train_df['Survived'])


# In[ ]:


f,ax=plt.subplots(1,1,figsize=(10,10))
sns.countplot('AgeGroup',hue='Survived',data=train_df, ax=ax)
plt.show()


# In[ ]:


train_df['TitleKey'] = 0
title_mapping = {'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Rare': 4, 'Royal': 5}
train_df['TitleKey'] = train_df['Title'].map(title_mapping)


# 

# ### Group with familly members - SibSp, Parch
# Check whether a passenger is alone or with his family members. SibSp is Siblings + Spouse, and Parch is Parent + Child
# 
# * Sibling = brother, sister, stepbrother, stepsister
# * Spouse = husband, wife
# 
# I couldn't figure out the difference of worth between these values, because they are all family and all important. 
# 
# So I'll create a new column 'FamilyMembers'

# In[ ]:


train_df['FamilyMembers'] = train_df['SibSp'] + train_df['Parch'] + 1
pd.crosstab([train_df['FamilyMembers']],train_df['Survived']).style.background_gradient(cmap='summer_r')


# In[ ]:


train_df['IsAlone'] = 0
train_df.loc[train_df['FamilyMembers'] == 1, 'IsAlone'] = 1

f, ax=plt.subplots(1,2,figsize=(20,8))
# sns.barplot('FamilyMembers','Survived',data=train_df,ax=ax[0])
ax[0].set_title('Survived, with family')
train_df['Survived'][train_df['IsAlone']==0].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
train_df['Survived'][train_df['IsAlone']==1].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],shadow=True)
# sns.barplot('IsAlone','Survived',data=train_df,ax=ax[1])
ax[1].set_title('IsAlone Survived')
plt.show()


# 

# In[ ]:


sns.heatmap(train_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# POSITIVE CORRELATION: If an increase in feature A leads to increase in feature B, then they are positively correlated. A value 1 means perfect positive correlation.
# 
# NEGATIVE CORRELATION: If an increase in feature A leads to decrease in feature B, then they are negatively correlated. A value -1 means perfect negative correlation.
# 
# Now from the above heatmap,we can see that the features are not much correlated. The highest correlation is between SibSp and Parch i.e 0.41. So we can carry on with all features.

# In[ ]:


# Do the same things to test data
test_df['Title'] = test_df['Name'].str.extract('([A-Za-z]+)\.')
test_df['Title'] = test_df['Title'].replace(['Capt', 'Col',
'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

test_df['Title'] = test_df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
test_df['Title'] = test_df['Title'].replace(['Mlle', 'Ms', 'Lady'], 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')

test_df.loc[(test_df['Age'].isnull())&(test_df['Title']=='Master'),'Age'] = 5
test_df.loc[(test_df['Age'].isnull())&(test_df['Title']=='Miss'),'Age'] = 22
test_df.loc[(test_df['Age'].isnull())&(test_df['Title']=='Mr'),'Age'] = 33
test_df.loc[(test_df['Age'].isnull())&(test_df['Title']=='Mrs'),'Age'] = 36
test_df.loc[(test_df['Age'].isnull())&(test_df['Title']=='Rare'),'Age'] = 45
test_df.loc[(test_df['Age'].isnull())&(test_df['Title']=='Royal'),'Age'] = 43

test_df['TitleKey'] = 0
title_mapping = {'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Rare': 4, 'Royal': 5}
test_df['TitleKey'] = train_df['Title'].map(title_mapping)

test_df['AgeGroup'] = 0
test_df.loc[ test_df['Age'] <= 7, 'AgeGroup'] = 0
test_df.loc[(test_df['Age'] > 7) & (test_df['Age'] <= 18), 'AgeGroup'] = 1
test_df.loc[(test_df['Age'] > 18) & (test_df['Age'] <= 30), 'AgeGroup'] = 2
test_df.loc[(test_df['Age'] > 30) & (test_df['Age'] <= 40), 'AgeGroup'] = 3
test_df.loc[(test_df['Age'] > 40) & (test_df['Age'] <= 60), 'AgeGroup'] = 4
test_df.loc[ test_df['Age'] > 60, 'AgeGroup'] = 5

test_df['FamilyMembers'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df['IsAlone'] = 0
test_df.loc[test_df['FamilyMembers'] == 1, 'IsAlone'] = 1


# In[ ]:





# In[ ]:


train, test = train_test_split(train_df,test_size=0.3,random_state=0)
target_col = ['Pclass', 'Sex', 'Embarked', 'TitleKey', 'AgeGroup', 'IsAlone']

train_X=train[target_col]
train_Y=train['Survived']
test_X=test[target_col]
test_Y=test['Survived']

features_one = train_X.values
target = train_Y.values

tree_model = DecisionTreeClassifier()
tree_model.fit(features_one, target)
dt_prediction = tree_model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(dt_prediction, test_Y))

# test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
test_features = test_df[target_col].values

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test_df["PassengerId"]).astype(int)
# my_prediction = my_tree_one.predict(test_features)
dt_prediction_result = tree_model.predict(test_features)
dt_solution = pd.DataFrame(dt_prediction_result, PassengerId, columns = ["Survived"])
# print(my_solution)

# Check that your data frame has 418 entries
print(dt_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
dt_solution.to_csv("my_solution_three.csv", index_label = ["PassengerId"]) 


# In[ ]:





# In[ ]:





# In[ ]:


f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(train_df[train_df['Pclass']==1]['Fare'],ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(train_df[train_df['Pclass']==2]['Fare'],ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(train_df[train_df['Pclass']==3]['Fare'],ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()


# Fare of tickets are all different, but mostly higher class ticket are more expensive than lower ones.

# ### Data cleaning...
# Now let's setup all 'string' datas into numeric value

# and make continous values(age, fare) into categorical values by either Binning or Normalisation.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




