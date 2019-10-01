#!/usr/bin/env python
# coding: utf-8

# **2018/03/25**

# # Introduction
# 
# Hello !  This is my first kernel. I want to develop my data science ability from Titanic slowly. So you will make as many analyses and visualizations as possible and see the results of machine learning through them.
# 
# If you take a long look at it in detail, this is how it will proceed
# 
# *  Load Data
#     *  load data
#     *  check missing data    
# *  Visualization
#     
# *  Filling Missing Value & engineering
#     *  Categorical
#     *  Numeric
#     
# *  Modeling

# # Load Data
# 
# Let's load Data

# ## load data

# In[200]:


# Load Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

train = pd.read_csv('../input/train.csv')
test =  pd.read_csv('../input/test.csv')


# After we have gained insight from the training data and created the model, we will use the test data to make predictions.

# In[201]:


train.head(10)


# Check the table below to see what the column values mean.

# **Data Dictionary**
#  <table>
#  <tbody>
#  <tr><th><b>Variable</b></th><th><b>Definition</b></th><th><b>Key</b></th></tr>
#  <tr>
#  <td>survival</td>
#  <td>Survival</td>
# <td>0 = No, 1 = Yes</td>
#  </tr>
#  <tr>
#  <td>pclass</td>
#  <td>Ticket class</td>
#  <td>1 = 1st, 2 = 2nd, 3 = 3rd</td>
#  </tr>
#  <tr>
#  <td>sex</td>
#  <td>Sex</td>
#  <td></td>
#  </tr>
#  <tr>
#  <td>Age</td>
#  <td>Age in years</td>
#  <td></td>
#  </tr>
#  <tr>
#  <td>sibsp</td>
#  <td># of siblings / spouses aboard the Titanic</td>
#  <td></td>
#  </tr>
#  <tr>
#  <td>parch</td>
#  <td># of parents / children aboard the Titanic</td>
#  <td></td>
#  </tr>
#  <tr>
#  <td>ticket</td>
#  <td>Ticket number</td>
#  <td></td>
#  </tr>
#  <tr>
#  <td>fare</td>
#  <td>Passenger fare</td>
#  <td></td>
#  </tr>
#  <tr>
#  <td>cabin</td>
# <td>Cabin number</td>
# <td></td>
# </tr>
# <tr>
# <td>embarked</td>
# <td>Port of Embarkation</td>
# <td>C = Cherbourg, Q = Queenstown, S = Southampton</td>
# </tr>
# </tbody>
# </table>
# 
# **Variable Notes**
# <p><b>pclass</b>: A proxy for socio-economic status (SES)<br> 1st = Upper<br> 2nd = Middle<br> 3rd = Lower<br><br> <b>age</b>: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5<br><br> <b>sibsp</b>: The dataset defines family relations in this way...<br> Sibling = brother, sister, stepbrother, stepsister<br> Spouse = husband, wife (mistresses and fiancés were ignored)<br><br> <b>parch</b>: The dataset defines family relations in this way...<br> Parent = mother, father<br> Child = daughter, son, stepdaughter, stepson<br> Some children travelled only with a nanny, therefore parch=0 for them.</p>

# ## check missing data

# In[202]:


print(train.shape)
print(train.apply(lambda x: sum(x.isnull()),axis=0))


# If you look at the value, there are 177 of the 891 in Age, 681 in Cabin, and two in Embarkd. Let's take care of your values after we review each of the data.

# --------------------------------
# 

# # Visualization

# ## 1. Survived 

# In[203]:


plt.subplots(figsize=(12,8))
sns.countplot('Survived',data=train).set_title('Survived')


# The graph shows that there are many more passengers who did not survive.

# ## Age

# In[204]:


train['Age'].fillna(train['Age'].median(),inplace =True)


# For visualization, first let's process the Age center.

# In[205]:


figure = plt.figure(figsize=(15,8))
plt.hist([train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], stacked=True, color = ['b','g'],bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()


# ## Sex

# In[206]:


print(train[['Sex','Survived']].groupby(['Sex']).count())
print('----------------------------')
print(train.groupby(['Sex','Survived']).size())
plt.subplots(figsize=(12,8))
sns.countplot('Sex',hue='Survived',data =train)


#  The number of men on board the ship is more than twice as many as that of women, but it seems that women survived.
#  Then use Age and Sex to draw a graph.

# In[207]:


plt.subplots(figsize=(15,10))
sns.swarmplot(x='Age',y='Sex',hue='Survived',data=train)


# I could find something very interesting. 
# The graph shows that most of the people rescued are women and children.

# ## Fare

# In[208]:


plt.hist([train[train['Survived']==1]['Fare'],train[train['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
print(train['Fare'].skew())


# The graph shows that Fare is tilted to the left.
# Let's take a quick look at it to see it more clearly.

# In[209]:


train["Fare"] = train["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
plt.hist([train[train['Survived']==1]['Fare'],train[train['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
print(train['Fare'].skew())


# ## PClass

# In[210]:


plt.subplots(figsize = (15,8))
sns.countplot('Pclass',hue='Survived',data =train)
sns.factorplot(x='Age',y='Sex',hue='Survived',row='Pclass',data=train,kind='violin',split=True,size=4,aspect=4)


# If you look at the graph, you can see that people with higher pc resistance are saved first in situations where women are rescued more than men.
# Pclass3 demonstrates that although it accounts for about half of all passengers, it has a lower survival rate than pc 1.

# ## Slibsp & Parch

# In[211]:


fig, ax = plt.subplots(3,1,figsize = (15,8))
sns.countplot('Parch',hue='Survived',data =train,ax=ax[0])
sns.countplot('SibSp',hue='Survived',data=train,ax=ax[1])
train['Fsize'] = train['Parch']+train['SibSp']
sns.countplot('Fsize',hue='Survived',data=train,ax=ax[2])


# In[212]:


print(train.Fsize.value_counts())


# By looking at the graph, you can see that there were more people living with fewer family members
# And you can see there is an Outlier too. Let's take care of it later.

# ## Embarked

# In[213]:


train['Embarked'] = train['Embarked'].fillna(method='ffill')


# In[214]:


plt.subplots(figsize = (15,8))
sns.countplot('Embarked',hue='Survived',data=train)
plt.subplots(figsize = (15,8))
sns.barplot(x='Embarked',y='Survived',data=train)


# According to the graph, C was the most likely to survive.
# S has survived more than C, but the odds are slim because there are so many people who don't survive.

# ------------------------------------

# # Filling Missing Data & engineering

# Recalls the data before learning the machine.
# Subsequent test and training data should be dealt with collectively.
# The reason is that the test and training data have the same column values, and therefore there is also the value of the panel.

# In[215]:


train_dataset = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')

target = train_dataset.Survived
train_dataset.drop('Survived',axis=1,inplace=True)

all = pd.concat([train_dataset,test_dataset])


# In[216]:


all.shape


# In[217]:


rest = []


# In[218]:


all.dtypes


# First, know what Categorical and Numeric data are to learn!

# In[219]:


categorical = [cname for cname in all if all[cname].dtype == 'object']
numerical = [cname for cname in all if all[cname].dtype in ['float64','int64']]


# Let's do the pre-treatment now.

# # Categorical

# ## Name

# In[220]:


all['Title'] = all['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
print(all.Title.value_counts())
plt.subplots(figsize=(15,8))
sns.countplot('Title',data= all)


# Let's tie the rest of it into the rest since it is a major withdrawal and the rest is a minor one.

# In[221]:


all['Title']= all.Title.replace(['Rev', 'Dr', 'Col', 'Major', 'Mlle',
       'Ms', 'Sir', 'Dona', 'Capt', 'Lady', 'Jonkheer', 'the Countess', 'Don',
       'Mme'],'Rest')
all.Title.value_counts()


# In[222]:


all.drop('Name',axis=1,inplace =True)
t_dummy = pd.get_dummies(all.Title,prefix='Title')
all = pd.concat([all,t_dummy],axis=1)

# check = later i drop title
rest.append('Title')


# In[223]:


all.shape


# ## Sex

# In[224]:


all['Sex'] = all.Sex.map({'male':1,'female':0})


# ## Ticket

# In[225]:


all.Ticket


# In[226]:


Ticket = []
for i in list(all.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("UnKnown")
        
all["Ticket"] = Ticket


# In[227]:


all.Ticket.value_counts()


# In[228]:


ticket_dummy = pd.get_dummies(all.Ticket,prefix='Ticket')
all = pd.concat([all,ticket_dummy],axis=1)
rest.append('Ticket')
all.shape


# ## Cabin

# In[229]:


all["Cabin"].isnull().sum()


# In[230]:


all.Cabin.describe()


# In[231]:


all.fillna('U0')
all['Cabin'] = [str(cname)[0] for cname in all.Cabin ]


# In[232]:


all.Cabin.value_counts()


# In[233]:


c_dummy = pd.get_dummies(all.Cabin, prefix='Cabin')
all = pd.concat([all,c_dummy],axis=1)
rest.append('Cabin')


# In[234]:


all.shape


# ## Embarked

# In[235]:


all.Embarked.isnull().sum()


# In[236]:


all.Embarked.value_counts()


# In[237]:


# Fill it up with the most S.
all.Embarked.fillna('S',inplace=True)
e_dummy = pd.get_dummies(all.Embarked,prefix='Embarked')
all = pd.concat([all,e_dummy],axis=1)
rest.append('Embarked')


# OK!!, Now let's go deal with the Numeric data.

# # Numeric

# The most important thing is handling age.
# As I saw before, age and cabin were your most valuable. And because Age is an influencing factor of the Stored.

# ## Age

# In[238]:


mean = train_dataset["Age"].mean()
std = test_dataset["Age"].std()
is_null = all["Age"].isnull().sum()
# compute random numbers between the mean, std and is_null
rand_age = np.random.randint(mean - std, mean + std, size = is_null)
# fill NaN values in Age column with random values generated
age_slice = all["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age
all["Age"] = age_slice
all["Age"] = all["Age"].astype(int)


# In[239]:


sns.factorplot(x="Survived", y = "Age",data = train, kind="box")
sns.factorplot(x="Survived", y = "Age",data = train, kind="violin")


# # SibSp & Parch

# In[240]:


all['Fsize'] = all['SibSp']+all['Parch'] +1 # Including self
plt.subplots(figsize=(15,8))
sns.countplot('Fsize',data=all)


# It is likely to be divided into single people and non-extremity people.

# In[241]:


all['alone'] = all['Fsize'].map(lambda x: 1 if x == 1 else 0)
all['not alone'] = all['Fsize'].map(lambda x: 1 if x >1 else 0)


# In[242]:


rest.append(['Fsize','SibSp','Parch'])


# # Fare

# In[243]:


all['Fare_loc'] = all.Fare.map(lambda i: np.log(i) if i > 0 else 0)


# In[245]:


all.drop('PassengerId',inplace=True,axis=1)


# ----------------------------------
# 

# In[247]:


rest = ['Title', 'Ticket', 'Cabin', 'Embarked','Fsize', 'SibSp', 'Parch']


# In[248]:


all.drop(rest,inplace=True,axis=1)


# In[249]:


all.shape


# ----------------------------------------

# # Modeling

# First, I will use XGBoos  to check it out.
# Afterwards, we will make a model using Tensorflow for comparative analysis.

# 
# ## What is XGBoost
# XGBoost is the leading model for working with standard tabular data (the type of data you store in Pandas DataFrames, as opposed to more exotic types of data like images and videos). XGBoost models dominate many Kaggle competitions.
# 
# To reach peak accuracy, XGBoost models require more knowledge and model tuning than techniques like Random Forest. After this tutorial, you'ill be able to
# 
# * Follow the full modeling workflow with XGBoost
# * Fine-tune XGBoost models for optimal performance
# XGBoost is an implementation of the Gradient Boosted Decision Trees algorithm (scikit-learn has another version of this algorithm, but XGBoost has some technical advantages.) What is Gradient Boosted Decision Trees? We'll walk through a diagram.
# 
# <img src="https://i.imgur.com/e7MIgXk.png" alt="xgboost image">
# 
# We go through cycles that repeatedly builds new models and combines them into an ensemble model. We start the cycle by calculating the errors for each observation in the dataset. We then build a new model to predict those. We add predictions from this error-predicting model to the "ensemble of models."
# 
# To make a prediction, we add the predictions from all previous models. We can use these predictions to calculate new errors, build the next model, and add it to the ensemble.
# 
# There's one piece outside that cycle. We need some base prediction to start the cycle. In practice, the initial predictions can be pretty naive. Even if it's predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.
# 
# This process may sound complicated, but the code to use it is straightforward. We'll fill in some additional explanatory details in the model tuning section below.
# 
# 

# In[272]:


from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score


# In[273]:


train = all.head(891)
test = all.iloc[891:]


# In[276]:


Fold= StratifiedKFold(n_splits=10)

gr_model = GradientBoostingClassifier(random_state=2)
xg_model = XGBClassifier(random_state=2)

gr_model.fit(train,target)
xg_model.fit(train,target)

print(cross_val_score(gr_model,train,target,scoring='accuracy',cv=Fold).mean())
print(cross_val_score(xg_model,train,target,scoring='accuracy',cv=Fold).mean())


# In[ ]:




