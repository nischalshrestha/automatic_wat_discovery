#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing necessary modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# **********Loading dataset********

# In[ ]:



train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# **Checking **

# In[ ]:



print(train.shape)
print(test.shape)
print(train.columns.values)


# In[ ]:


print("Train")
print (train.info())
print ("*"*50)
print("Test")
print (test.info())


# **#Overall View of Data**

# In[ ]:



get_ipython().run_cell_magic(u'HTML', u'', u"<div class='tableauPlaceholder' id='viz1516349898238' style='position: relative'>\n<noscript><a href='#'>\n<img alt='An Overview of Titanic Training Dataset ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;Titanic_data_mining&#47;Dashboard1&#47;1_rss.png' style='border: none' />\n</a></noscript>\n<object class='tableauViz'  style='display:none;'>\n<param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> \n<param name='embed_code_version' value='3' /> <param name='site_root' value='' />\n<param name='name' value='Titanic_data_mining&#47;Dashboard1' /><param name='tabs' value='no' />\n<param name='toolbar' value='yes' />\n<param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;Titanic_data_mining&#47;Dashboard1&#47;1.png' /> \n<param name='animate_transition' value='yes' />\n<param name='display_static_image' value='yes' />\n<param name='display_spinner' value='yes' />\n<param name='display_overlay' value='yes' />\n<param name='display_count' value='yes' />\n<param name='filter' value='publish=yes' />\n</object></div>                \n<script type='text/javascript'>var divElement = document.getElementById('viz1516349898238');   \nvar vizElement = divElement.getElementsByTagName('object')[0];               \nvizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';  \nvar scriptElement = document.createElement('script');              \nscriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';    \nvizElement.parentNode.insertBefore(scriptElement, vizElement);       \n</script>")


# In[ ]:


print("Train :")
print (train.isnull().sum())
print (''.center(20, "*"))
print("Test :")
print (test.isnull().sum())


# **Pre-Processing Starts**

# In[ ]:


passengerid = test.PassengerId
train.drop(['PassengerId'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)


# In[ ]:


train.drop(['Embarked'],axis=1,inplace=True)
train.drop(['Name'],axis=1,inplace=True)
test.drop(['Embarked'],axis=1,inplace=True)
test.drop(['Name'],axis=1,inplace=True)


# In[ ]:


print(train.head(),"\n\n\n",train.shape)
print("*"*50)
print(test.head(),"\n\n\n",test.shape)


# In[ ]:


print(train.info())
print("*"*50)
print(test.info())


# **Feature  Cabin and Age has some missing values**
# 

# In[ ]:


train['sex_code'] = train['Sex'].astype('category').cat.codes
test['sex_code'] = test['Sex'].astype('category').cat.codes


# In[ ]:


del train['Sex']
del train['Ticket']
del test['Sex']
del test['Ticket']


# In[ ]:


del train['Cabin']
del test['Cabin']


# In[ ]:


print(train.head(3))
print("*"*50)
print(test.head(3))


# In[ ]:


train['family_size'] = train.SibSp + train.Parch+1
train['is_alone'] = [1 if i<2 else 0 for i in train.family_size]
train['calculated_fare'] = train.Fare/train.family_size

test['family_size'] = test.SibSp + test.Parch+1
test['is_alone'] = [1 if i<2 else 0 for i in test.family_size]
test['calculated_fare'] = test.Fare/test.family_size


# In[ ]:


print(train.columns.values,test.columns.values)


# In[ ]:


print(train.head())
print(test.head())


# In[ ]:


survived_summary = train.groupby("Survived")
print(survived_summary.mean())


# In[ ]:


survived_summary = train.groupby("sex_code")
print(survived_summary.mean())


# **Ploting different graghs**

# In[ ]:


plt.subplots(figsize = (15,12))
sns.heatmap(train.corr(), 
            annot=True,
            cmap = 'RdBu_r',
            linewidths=0.1, 
            linecolor='white',
            vmax = .9,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20);
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'] , color='r',shade=True,label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Age'] , color='g',shade=True, label='survived')
plt.title('Age Distribution - Surviver V.S. Non Survivors')
plt.xlabel("Age", fontsize = 15)
plt.ylabel('Frequency', fontsize = 15);
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15,8),)
## I have included to different ways to code a plot below, choose the one that suites you. 
ax=sns.kdeplot(train.Pclass[train.Survived == 0] , color='r',shade=True,label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Pclass'] , color='g',shade=True, label='survived')
plt.title('Passenger Class Distribution - Survived vs Non-Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Passenger Class", fontsize = 15)
## Converting xticks into words for better understanding
labels = ['First', 'Second', 'Third']
plt.xticks(sorted(train.Pclass.unique()), labels);
plt.show()


# In[ ]:


plt.subplots(figsize = (15,10))
sns.barplot(x = "Pclass", y = "Survived", data=train, edgecolor=(0,0,0), linewidth=2)
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)
plt.xlabel("Socio-Economic class", fontsize = 15);
plt.ylabel("% of Passenger Survived", fontsize = 15);
plt.show()


# In[ ]:


plt.subplots(figsize = (15,8))
sns.barplot(x = "sex_code", y = "Survived", data=train, linewidth=2)
plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 25)
plt.ylabel("% of passenger survived", fontsize = 15)
plt.xlabel("Gender",fontsize = 15)

labels = ['Female', 'Male']
plt.xticks(sorted(train.sex_code.unique()), labels);
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Fare'] , color='r',shade=True,label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Fare'] , color='g',shade=True, label='survived')
plt.title('Fare Distribution Survived vs Non Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Fare", fontsize = 15)
plt.show()


# In[ ]:


#print(train.Age.isnull().sum())
train['Age'].fillna(0,inplace=True)
features = train.drop(['Survived'],axis=1)
target = train.Survived


# In[ ]:


print(features.columns.values)
print(target[1:4])


# **Selecting different models**

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
train_x,test_x,train_y,test_y = train_test_split(features,target,random_state=30,test_size=0.3)
model_Log = LogisticRegression(random_state=34)
model_Log.fit(train_x,train_y)
print("Accuracy using LogisticRegression ",accuracy_score(test_y,model_Log.predict(test_x)))



# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model_RFC = RandomForestClassifier(random_state=45)
model_RFC.fit(train_x,train_y)
print("Accuracy using RandomForestClassifier ",accuracy_score(test_y,model_RFC.predict(test_x)))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model_DTC = DecisionTreeClassifier(random_state = 40)
model_DTC.fit(train_x,train_y)
print("Accuracy using DecisionTreeClassifier ",accuracy_score(test_y,model_DTC.predict(test_x)))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model_RFC = RandomForestClassifier(random_state=45)
model_RFC.fit(features,target)
test['Age'].fillna(test['Age'].mean(),inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace = True)
test['calculated_fare'].fillna(0,inplace = True)
pred = model_RFC.predict(test)
#print(test.info())
#print(train.info())
#print(test.info())
#print(train.shape,test.shape)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": passengerid,
        "Survived": pred
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:




