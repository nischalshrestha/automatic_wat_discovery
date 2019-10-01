#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# ## A. Data Overview :

# In[ ]:


# 1. Load train & test Dataset 
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

# Profile Report on Dataset 
#pfr = pandas_profiling.ProfileReport(train_data)
#report.to_file("DS_Titanic_AF.html")

print('train data: %s, test data %s' %(str(train_data.shape), str(test_data.shape)) )
train_data.head()
print('_'*40)


# In[ ]:


train_data.head()


# ## B. [AF] Data Analysis - First Observation
# 
# ####  Unique Values:
# #PasengerID Seems to be Unique but we will need to make sure.
# #Name : Can be Unique
# 
# ####  Missing Values:
# * Cabin  { Need to Check all Columns}
# 
# ####  Numerical Columns:
# * Pclass
# * SibSp
# * Parch
# * Fare
# 
# ####  Cateorical Columns:
# * Sex - Male & Female
# * Embarked
# 
# ####  AlphaNumeric Columns:
# * Ticket
# * Cabin
# 
# ####  dTypes MissMatch:
# * Age - It should be in int but available in float
# * Fare
# 
# ####  target Column:
# * Survived 
# 
# ####  Enrich Data:
# 1. SibSp & Parch relates to Family members there cab be a group of individuals or Families.
# -  Family = Parch + SibSp + Self
# - Individual = Find Alone Travellers.
# 2. Sex : We can find how many Male/ Female survivors are there.
# 3. Age: We can find who are Child / Adult/ Old age travellers
# 3. Pclass & Fare : Will give us an idea about Financial Status of travellers 
# 4. Cabin & Ticket : Check if we can find some info from Ticket Numeber
# 5. Embarked : From which destination travellers borded in ship
# 

# In[ ]:


#### 1. Unique Values:
#PasengerID : Yes
#Name :  No
#---------------------------------------------------
# Check If PassengerId is Unique 
if (train_data.PassengerId.nunique() == train_data.shape[0]):
    print('PassengerId is Unique.') 
else:
    print('-[Error]Id is not Unique.')    
    
if (len(np.intersect1d(train_data.PassengerId.values, test_data.PassengerId.values))== 0 ):
    print('train and test datasets are Distinct.')
else:
    print('- [Error] train and test datasets are NOT Distinct.')


# In[ ]:


# Check If PassengerId is Unique 
if (train_data.Name.nunique() == train_data.shape[0]):
    print('Name is Unique.') 
else:
    print('[Error]Id is not Unique.')    
    
if (len(np.intersect1d(train_data.Name.values, test_data.Name.values))== 0 ):
    print('train and test datasets Names are Distinct.')
else:
    print('- [Error] train and test datasets Names are NOT Distinct.')


# ## B. Data Cleaning
# 

# In[ ]:


#### C. Find Missing Values --> DONE
# Age                 
# Cabin            
# Embarked           
# Fare            
#---------------------------------------------------

# [AF] From describe() we come to know that Age contains Some Missing Data. We need to fix this.
# [AF] We will Check All Variables for Missing values in Columns!

# train_data.apply(lambda x: sum(x.isnull().values), axis = 0) # For columns

# Check for missing data & list them 
nas = pd.concat([train_data.isnull().sum(), test_data.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset']) 
print('Nan in the data sets')
print(nas[nas.sum(axis=1) > 0])

# [Extra] A boolean Condition to Check wheather our Dataset have any Missing Value.
#train_data.isnull().values.any()


# 
# 

# In[ ]:


# Fill Missing Values or Ignore Columns --> DONE
# Age                 
# Cabin            
# Embarked           
# Fare 
# ---------------- Age ----------------

# Fill NaN
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
 # convert from float to int
train_data['Age'] = train_data['Age'].astype(int)

# Fill NaN
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
 # convert from float to int
test_data['Age'] = test_data['Age'].astype(int)


# In[ ]:


g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Age', bins=10)


# In[ ]:


# ---------------- Cabin ----------------
# Drop the This not relevent 
train_data.drop(['Cabin','Ticket'], axis=1, inplace=True)
test_data.drop(['Cabin','Ticket'], axis=1, inplace=True)


# In[ ]:


# ---------------- Embarked ----------------
print(test_data['Embarked'].mode())
print(test_data['Embarked'].mode()[0])
#replacing the missing values in the Embarked feature with S
train_data = train_data.fillna({"Embarked": train_data['Embarked'].mode()})


# TestData doesn't contain Missing Values for Embarked


# In[ ]:


# ---------------- Fare ----------------
# Fill NaN
train_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)
 # convert from float to int
train_data['Fare'] = train_data['Fare'].astype(int)

# Fill NaN
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
 # convert from float to int
test_data['Fare'] = test_data['Fare'].astype(int)


# In[ ]:


g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Fare', bins=10)


# In[ ]:


print('_'*40)


# In[ ]:



# Check for missing data & list them 
nas = pd.concat([train_data.isnull().sum(), test_data.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset']) 
print('Nan in the data sets')
print(nas[nas.sum(axis=1) > 0])


# In[ ]:


print('train data: %s, test data %s' %(str(train_data.shape), str(test_data.shape)) )
train_data.info()


# #### D. Numerical Columns:
# #Pclass
# #SibSp
# #Parch
# #Fare
# #---------------- Age ----------------

# In[ ]:


train_data.shape


# In[ ]:


train_data.describe(include=['number'])


# #### E. Cateorical Columns:
# * Name: 
# #Sex - Male & Female
# * Embarked
# 
# 

# In[ ]:


train_data.describe(include=['object'])


# ### Univariate Analysis

# In[ ]:


fig, axes = plt.subplots(2, 4, figsize=(16, 10))
sns.countplot('Survived',data=train_data,ax=axes[0,0])
sns.countplot('Pclass',data=train_data,ax=axes[0,1])
sns.countplot('Sex',data=train_data,ax=axes[0,2])
sns.countplot('SibSp',data=train_data,ax=axes[0,3])
sns.countplot('Parch',data=train_data,ax=axes[1,0])
sns.countplot('Embarked',data=train_data,ax=axes[1,1])
sns.distplot(train_data['Fare'], kde=True,ax=axes[1,2])
sns.distplot(train_data['Age'].dropna(),kde=True,ax=axes[1,3])


# ## Bivariate EDA

# In[ ]:


import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))
corr = train_data.corr()
sns.heatmap(corr,
            mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# #### F. AlphaNumeric Columns:
# * Ticket
# * Cabin
# 

# #### G. dTypes MissMatch:
# * Age - It should be in int but available in float
# * Fare
# 

# #### H. Enrich Data:
# 1. SibSp & Parch relates to Family members there cab be a group of individuals or Families.
# -  Family = Parch + SibSp + Self
# - Individual = Find Alone Travellers.
# 2. Sex : We can find how many Male/ Female survivors are there.
# 3. Age: We can find who are Child / Adult/ Old age travellers
# 
# 3. Pclass & Fare : Will give us an idea about Financial Status of travellers 
# 4. Cabin & Ticket : Check if we can find some info from Ticket Numeber
# 5. Embarked : From which destination travellers borded in ship
# 

# In[ ]:


train_data.head()


# In[ ]:


# 4 Sex:
def personType(Gender):
    if Gender == "female":
        return 1
    elif Gender =="male":
        return 0
    
train_data['Sex'] = train_data['Sex'].apply(personType)
 
# test_data
test_data['Sex'] = test_data['Sex'].apply(personType)



# In[ ]:


# 5 Age:
def ageType(passAge):
    if passAge < 16:
        return str('Child')
    else :
        return str('adult')
    
# train_data['ageType'] = train_data['Age'].apply(ageType)

print('train dataset: %s, test dataset %s' %(str(train_data.shape), str(test_data.shape)) )


# In[ ]:



train_data.head()


# In[ ]:


test_data.head()


# ## C. Data Visualisation 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(train_data['Age'],bins=15,kde=False)
plt.ylabel('Count')
plt.title('Aget Distribution -AF')


# In[ ]:


from matplotlib import style
style.use('ggplot')
plt.figure(figsize=(12,4))
sns.boxplot(x='Age', data = train_data)


# **Conclusion**: There are few outliers. There were old guys above 57 age.

# In[ ]:


print('train dataset: %s, test dataset %s' %(str(train_data.shape), str(test_data.shape)) )


# ## Feature Engineering

# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_data = train_data.drop(['Name',  'Fare', 'Embarked'],axis=1)
test_data = test_data.drop(['Name', 'Fare', 'Embarked'],axis=1)


# In[ ]:



#X_train = train_dataset.drop("Survived",axis=1).as_matrix()
#Y_train = train_dataset["Survived"].as_matrix()
#X_test  = test_dataset.drop("PassengerId",axis=1).copy().as_matrix()


X_train = train_data.drop("Survived",axis=1)
Y_train = train_data["Survived"]


print(X_train.shape)
print(Y_train.shape)



# In[ ]:


X_test = test_data.copy()

X_test.shape
print(X_test.head())


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# In[ ]:


Y_train.head()


# In[ ]:


# machine learning
from sklearn.tree import DecisionTreeClassifier

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


print(Y_pred)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

# Classifier 
model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
# Fit Data
model.fit(X_train,Y_train)
print("oob_score : ", model.oob_score_)
y_oob = model.oob_prediction_
print("C-stat: ", roc_auc_score(Y_train,y_oob))


# In[ ]:


model.feature_importances_


# In[ ]:


feature_importances = pd.Series(model.feature_importances_,index=X_train.columns)
feature_importances.sort_values()
feature_importances.plot(kind="barh",figsize=(7,6));


# ## D. ML Model :

# ### Model Performance 

# ### Final Model

# ## E Test Data

# In[ ]:


my_submission  = pd.DataFrame({
    "PassengerId":X_test["PassengerId"],
    "Survived":Y_pred
})
my_submission.to_csv("afarane_titanic_kaggle.csv",index=False)


# In[ ]:


my_submission.head()


# In[ ]:


my_submission.tail()

