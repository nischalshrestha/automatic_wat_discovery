#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


# In[5]:


data = pd.read_csv('../input/train.csv')


# In[6]:


data.head()


# In[7]:


data.shape


# **PassengerId** , Numerical or Categorical ?
# 

# In[8]:


data.PassengerId.unique().shape


# **PassengerId** size match with train set size , It represent **unique** value to each row of dataset

# **Sex** is *Numerical* or *Categorical*** Variable?

# In[9]:


data.Sex.unique()


# Sex has  **two** values(Categorical) , visualise the value count .

# In[ ]:


fig = data['Sex'].value_counts().plot.bar()
fig.set_title('Sex')
fig.set_ylabel('No of passengars')


# **Pclass**: A proxy for socio-economic status (SES)
# 
# *1st* = Upper
# 
# *2nd* = Middle
# 
# *3rd* = Lower
# 
# 
# Hold three values .It's another categorical variable of type Ordinal categorical variables. 

# In[10]:


data.Pclass.unique()


# Categorial variables are visualised in Bar char their count values .

# In[11]:


fig = data.Pclass.value_counts().plot.bar()
fig.set_title('Pclass')
fig.set_ylabel('No of passengers')


# Passenger Fare, **Numerical** or **Categorical**?

# In[ ]:


data.Fare.unique()


# **Numerical variable** are numbers, Fare has numbers.
# **Fare** variable that may contain any value within some range, **Continuous**.
# Continous variable is visulaized using hist chart.

# In[12]:


fig = data.Fare.hist(bins=50)
fig.set_title('Fare')
fig.set_xlabel('Fare Amt')
fig.set_ylabel('No of Passengers')


# Age in years of Passengers , Continous

# In[13]:


data.Age.unique()


# In[ ]:


fig = data.Age.hist(bins=50)
fig.set_title('Age')
fig.set_xlabel('Age of Passengers')
fig.set_ylabel('No of Passengers')


# **sibsp** of siblings / spouses aboard the Titanic, a Discrete variable

# In[14]:


data.SibSp.unique() #Discrete Feature


# In[15]:


fig = data.SibSp.hist(bins=50)
fig.set_title('SibSp')
fig.set_xlabel('SibSp count')
fig.set_ylabel('No of Passengers')


# **parch** of parents / children aboard the Titanic , a Discrete varaible

# In[16]:


data.Parch.unique() # Discrete Feature


# In[17]:


fig = data.Parch.hist(bins=50)
fig.set_title('Parch')
fig.set_xlabel('Parch Count')
fig.set_ylabel('No Of Passengers')


# **embarked** - Port of Embarkation
# 
# C = Cherbourg,
# Q = Queenstown,
# S = Southampton
# 
# Has three values , Categorical variable

# In[18]:


data.Embarked.unique() # Discrete Feature


# In[19]:


fig = data.Embarked.value_counts().plot.bar()
fig.set_title('Embarked')
fig.set_ylabel('No of Passengers')


# **cabin** - Cabin number , Categorical variable 

# In[20]:


data.Cabin.unique()


# In[ ]:


fig = data.Cabin.value_counts().plot.bar()
fig.set_title('Cabin')
fig.set_ylabel('No of Passengers')


# Survived is the Class label , Categorical - Survived or not Survived.

# In[21]:


data.Survived.unique()


# In[22]:


fig = data.Survived.value_counts().plot.bar()
fig.set_title('Survived')
fig.set_ylabel('No of Passengers')


# **Visualize how many Passengers survived or not survived w.r.t each Feature variable values **
# 
# 
# **Discrete & Categorial variable:**

# In[23]:


survived_sex = data[data.Survived==1]['Sex'].value_counts()
not_survived_sex = data[data.Survived==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,not_survived_sex])
df.index =['survived_sex','not_survived_sex']
df.plot(kind='bar')


# In[24]:


survived_pclass = data[data.Survived==1]['Pclass'].value_counts()
not_survived_pclass = data[data.Survived==0]['Pclass'].value_counts()
df = pd.DataFrame([survived_pclass,not_survived_pclass])
df.index =['survived_pclass','not_survived_pclass']
df.plot(kind='bar')


# In[25]:


survived_sibsp = data[data.Survived==1]['SibSp'].value_counts()
not_survived_sibsp = data[data.Survived==0]['SibSp'].value_counts()
df = pd.DataFrame([survived_sibsp,not_survived_sibsp])
df.index =['survived_sibsp','survived_sibsp']
df.plot(kind='bar')


# In[26]:


survived_parch = data[data.Survived==1]['Parch'].value_counts()
not_survived_parch = data[data.Survived==0]['Parch'].value_counts()
df = pd.DataFrame([survived_parch,not_survived_parch])
df.index =['survived_parch','not_survived_parch']
df.plot(kind='bar')


# In[27]:


survived_embarked = data[data.Survived==1]['Embarked'].value_counts()
not_survived_embarked = data[data.Survived==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embarked,not_survived_embarked])
df.index =['survived_embarked','not_survived_embarked']
df.plot(kind='bar')


# **Numeric variable:**

# In[28]:


fig = plt.figure()
data1 = data.dropna() #error withou drop NA
plt.hist([data1[data1['Survived']==1]['Age'],data1[data1['Survived']==0]['Age']],bins=30,label=['Survived','Not_Survived'])
plt.xlabel('Age')
plt.ylabel('No of passengers')
plt.legend()


# In[29]:


ig = plt.figure()
data1 = data.dropna() #error withou drop NA
plt.hist([data1[data1['Survived']==1]['Fare'],data1[data1['Survived']==0]['Fare']],bins=30,label=['Survived','Not_Survived'])
plt.xlabel('Fare')
plt.ylabel('No of passengers')
plt.legend()


# **Feature Engineering **
# 
# Combine Training & Test data

# In[30]:


def combine_data():
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    
    y = train.Survived
    train.drop('Survived',axis = 1,inplace = True)
    train.drop('PassengerId',axis = 1,inplace = True)
    test.drop('PassengerId',axis = 1,inplace = True)# High cordinality , remove
    train.drop('Ticket',axis = 1,inplace = True)
    test.drop('Ticket',axis = 1,inplace = True)# High cordinality , remove
    combined = train.append(test)
    combined.reset_index(inplace = True)
    combined.drop('index',axis =1,inplace = True)
    return combined
    


# In[31]:


combined_data = combine_data()
print(combined_data.head())    


# In[32]:


def get_Name_title():
    global combined_data
    combined_data['Name_title'] = combined_data['Name'].map(lambda name : name.split(',')[1].split('.')[0].strip())
    combined_data.drop('Name',axis =1,inplace = True)
    return combined_data['Name_title']


# In[33]:


get_Name_title()


# In[34]:


combined_data.head()


# In[35]:


fig = combined_data.Name_title.value_counts().plot.bar()
fig.set_title('Name_title')
fig.set_ylabel('No of Passengers')


# In[36]:


combined_data.info()


# In[ ]:


combined_data.isnull().sum()


# In[37]:


combined_data.isnull().mean()


# In[38]:


combined_data.iloc[:891].Age.isnull().sum()


# In[39]:


combined_data.iloc[891:].Age.isnull().sum()


# In[40]:


combined_data.iloc[891:]['Age'].describe()


# In[41]:


combined_data['Age'].fillna(combined_data.iloc[:891]['Age'].mean(),inplace = True)
#combined_data.Age_fill.isnull().sum()


# In[42]:


combined_data.iloc[:891]['Age']


# In[43]:


combined_data[combined_data['Fare'].isnull()]


# In[44]:


combined_data['Fare'].fillna(combined_data.iloc[:891]['Fare'].mean(),inplace = True)
#combined_data[combined_data['Fare_fill'].isnull()]


# In[45]:


combined_data['Embarked'].fillna(combined_data.iloc[:891]['Embarked'].value_counts().index[0],inplace =True)
#combined_data[combined_data['Embarked_fill'].isnull()]


# In[46]:


def Cabin_fill():
    global combined_data
    combined_data['Cabin'] = combined_data['Cabin'][combined_data['Cabin'].notnull()].map(lambda c:c[0])
    combined_data['Cabin'].fillna(combined_data['Cabin'].value_counts().index[0],inplace = True)
    return combined_data


# In[47]:


combined_data = Cabin_fill()


# In[48]:


combined_data.head()


# In[49]:


name_dummies = pd.get_dummies(combined_data['Name_title'],prefix='Name')
combined_data = pd.concat([combined_data,name_dummies],axis = 1)
combined_data.drop('Name_title',axis=1,inplace=True)
combined_data.head()


# In[50]:


embarked_dummies = pd.get_dummies(combined_data['Embarked'],prefix='Embarked')
combined_data = pd.concat([combined_data,embarked_dummies],axis = 1)
combined_data.drop('Embarked',axis=1,inplace=True)
combined_data.head()


# In[51]:


cabin_dummies = pd.get_dummies(combined_data['Cabin'],prefix='Cabin')
combined_data = pd.concat([combined_data,cabin_dummies],axis = 1)
combined_data.drop('Cabin',axis=1,inplace=True)
combined_data.head()


# In[52]:


pclass_dummies = pd.get_dummies(combined_data['Pclass'],prefix='Pclass')
combined_data = pd.concat([combined_data,pclass_dummies],axis = 1)
combined_data.drop('Pclass',axis=1,inplace=True)
combined_data.head()


# In[53]:


sibSp_dummies = pd.get_dummies(combined_data['SibSp'],prefix='SibSp')
combined_data = pd.concat([combined_data,sibSp_dummies],axis = 1)
combined_data.drop('SibSp',axis=1,inplace=True)
combined_data.head()


# In[54]:


parch_dummies = pd.get_dummies(combined_data['Parch'],prefix='Parch')
combined_data = pd.concat([combined_data,parch_dummies],axis = 1)
combined_data.drop('Parch',axis=1,inplace=True)
combined_data.head()


# In[55]:


combined_data['Sex'] = combined_data['Sex'].map({'male':1,'female':0})
#combined_data['Sex'] = combined_data['Sex'].map(lambda s :if(s == 'male') 1 else 0)
#combined_data.head()


# In[56]:


combined_data.head()


# In[70]:


from  sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# In[64]:


def split_data():
    global combined_data
    targets = pd.read_csv('../input/train.csv', usecols=['Survived'])['Survived'].values
    train = combined_data.iloc[:891]
    test = combined_data.iloc[891:]
    return train, test, targets


# In[65]:


train, test, targets = split_data()


# In[71]:


def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)


# In[76]:


model = LogisticRegression()
score = compute_score(clf = model, X=train,y= targets,scoring = 'accuracy' )
print( 'CV score = {0}'.format(score))


# In[79]:


model.fit(train, targets)
output = model.predict(test).astype(int)


# In[83]:


test.head()


# In[85]:


df_output = pd.DataFrame()
orginal_test = pd.read_csv('../input/test.csv')
df_output['PassengerId'] = orginal_test['PassengerId']
df_output['Survived'] = output
#predictions_df.columns = ['PassengerId', 'Survived']



# In[ ]:


df_output.to_csv('submission.csv', index=False)

