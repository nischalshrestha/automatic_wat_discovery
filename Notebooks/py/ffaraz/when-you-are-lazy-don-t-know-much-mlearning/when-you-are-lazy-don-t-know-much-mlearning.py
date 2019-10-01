#!/usr/bin/env python
# coding: utf-8

# # Titanic: When you are Lazy and don't know much about Machine Learning!

# Before starting this project, I have reviewed a few of the successful kernels. I am not sure if I can top the most popular ones and I am relatively a lazy coder so I try to use a simple approach to solve this problem.

# ### 1- Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
get_ipython().magic(u'matplotlib inline')


# ### 2- Importing the data

# In[ ]:


train=pd.read_csv('../input/train.csv')


# Just added familySize and filled the missing Age.

# In[ ]:


for item in [train]:
    item['FamilySize']=item['Parch']+item['SibSp']
    item['Age']=item['Age'].fillna(item['Age'].median())    
train.head()


# ### 3- PreAnalysis

# I am trying to find the independent variables that have a significant effect on the survival rate. Logically, we can argue that Fare rate, Pclass and having a cabin or not and having a title or not are correlated with each other; I just consider one of them, e.g. Pclass. Age, Sex, Family size, and Embark are independent with each other.
# Obviously, Sex is a very important factor since women have priority to get to lifeboats, so we categorize everything by Sex too.
# 

# In[ ]:


figure, [ax1,ax2,ax3,ax4]=plt.subplots(4,1,figsize=(12,12))

sb.violinplot(x='Pclass',y='Survived',hue='Sex',data=train,split=True,inner='point',ax=ax1)
sb.violinplot(x='FamilySize',y='Survived',hue='Sex',data=train,split=True,inner='point',ax=ax2)
sb.violinplot(x='Embarked',y='Survived',hue='Sex',data=train,split=True,inner='point',ax=ax3)
sb.violinplot(x='Survived',y='Age',hue='Sex',data=train,split=True,inner='point',ax=ax4)


# You can easily see that the 1st and 2nd class have very similar survival rate while the third class is very different.
# Family size and Embarked have some effect too. 
# The effect of Age is not very clear from this plot.

# In[ ]:


survivedAge=train.loc[(train['Survived']==1),'Age']
deadAge=train.loc[(train['Survived']==0),'Age'];
sb.distplot(survivedAge)
sb.distplot(deadAge,color='r')
plt.ylabel('Survival Probablity');plt.xlabel('Age');plt.ylim(0,0.08);plt.xlim(0,60)
plt.legend(['Survived','Dead'])
[survivedAge.mean(), deadAge.mean(),survivedAge.std(),deadAge.std()]


# The age groups are relatively similar; just there is a good amount of survived people below age of 10; so, we categorize them to adult(>10 yrs) and non-adults(<10 yrs)
# We also change the Sex to being Male or not and change the Embarked to 0,1,2 categories. Class status is true for 1st and 2nd and False for 3rd.
# Drop the rest.

# In[ ]:


for item in [train]:    
    item['Adult']=item['Age'].apply(lambda x:0 if x<10 else 1)
    item['Male']=item['Sex'].apply(lambda x:0 if x=='female' else 1)
    item['EmbarkedNum']=item['Embarked'].apply(lambda x:0 if x=='S' else (1 if x=='C' else 2 ))
    item['Class']=item['Pclass'].apply(lambda x:1 if (x==1 or x==2) else 0)
train=train.drop(['PassengerId','Pclass','Sex','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age'],axis=1)
train.head()


# ### 4-Model

# I just use logistic regression here. The score is accaptable.

# In[ ]:


import sklearn
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression().fit(train[['Male','Class','EmbarkedNum','FamilySize']],train['Survived'])
print(lg.score(train[['Male','Class','EmbarkedNum','FamilySize']],train['Survived']))


# ### 5- Testing

# In[ ]:


test=pd.read_csv('../input/test.csv')
for item in [test]:
    item['FamilySize']=item['Parch']+item['SibSp']
    item['Age']=item['Age'].fillna(item['Age'].median())  
    item['Adult']=item['Age'].apply(lambda x:0 if x<10 else 1)
    item['Male']=item['Sex'].apply(lambda x:0 if x=='female' else 1)
    item['EmbarkedNum']=item['Embarked'].apply(lambda x:0 if x=='S' else (1 if x=='C' else 2 ))
    item['Class']=item['Pclass'].apply(lambda x:1 if (x==1 or x==2) else 0)
test=test.drop(['Pclass','Sex','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age'],axis=1)


# In[ ]:


Answer=pd.DataFrame()
Answer['PassengerID']=test['PassengerId']
Answer['Survived']=lg.predict(test[['Male','Class','EmbarkedNum','FamilySize']])


# In[ ]:


Answer.to_csv('Answer.csv',index=False)


# With this model the score is 0.77511 which I beleive is good considering the simplicity.

# In[ ]:




