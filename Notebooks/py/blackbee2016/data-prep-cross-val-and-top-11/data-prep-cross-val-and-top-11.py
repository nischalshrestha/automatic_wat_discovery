#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# In[27]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[28]:


train.head(3)


# We see that we can easily extract the title ( Mrs, Mr, Miss) out of the Name variable to one hot encode the result.

# In[29]:


train['Name']=train['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0]


# In[30]:


train.Name[0:4]


# Let's one hot encode categorical variables.

# In[31]:


train = pd.get_dummies(train,columns=['Embarked','Sex','Name'])


# In[32]:


train.columns


# Lets not worry about those .. 

# In[33]:


del train['Cabin']
del train['Ticket']


# Here we fill nan with median as it is a robust estimator

# In[34]:


for i in train.columns:
    mediane = train[i].median()
    train[i].fillna(mediane,inplace=True)


# ### Applying treatment to the test dataset

# In[35]:


test['Name']=test['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0]
test = pd.get_dummies(test,columns=['Embarked','Sex','Name'])
del test['Cabin']
del test['Ticket']
for i in test.columns:
    mediane = test[i].median()
    test[i].fillna(mediane,inplace=True)


# In[36]:


train.columns.shape


# In[37]:


test.columns.shape


# We seem to have a mismatch in our columns... Probably in our name dummification...

# In[38]:


for i in train.columns:
    if i =='Survived':
        continue
    if i not in test.columns:
        del train[i]
        
for i in test.columns:
    if i not in train.columns:
        del test[i]


# Here we are going to tune our classifier and cross validate the result to make sure we are not overfitting

# In[39]:


from sklearn.model_selection import train_test_split


# In[40]:


score=[]
score_b = []
for i in range(10):
    rf = RandomForestClassifier(n_estimators=1000,max_depth=6,n_jobs=-1,criterion='entropy',max_features='sqrt',random_state=5)    
    TRAIN,TEST = train_test_split(train,test_size=0.2)
    y_TRAIN = TRAIN['Survived']
    del TRAIN['Survived']
    y_TEST = TEST['Survived']
    del TEST['Survived']
    rf.fit(TRAIN,y_TRAIN)

    a= sum(rf.predict(TEST)==y_TEST)/len(y_TEST)#Manually computing accuracy here
    b= sum(rf.predict(TRAIN)==y_TRAIN)/len(y_TRAIN) #Manually computing accuracy here
    score_b.append(b)
    score.append(a)


# In[41]:


import matplotlib.pyplot as plt
plt.plot(score_b)
plt.plot(score)
plt.legend(['Train Data','Test Data'])
plt.title('Cross validation accurracy accross results')


# In[42]:


y_train = train['Survived']
del train['Survived']


# In[43]:


rf = RandomForestClassifier(n_estimators=1000,max_depth=6,n_jobs=-1,criterion='entropy',max_features='sqrt',random_state = 5)
rf.fit(train,y_train)
Y = rf.predict(test)


# In[44]:


sub = pd.read_csv('../input/gender_submission.csv')
sub['Survived'] = Y


# In[45]:


sub.to_csv('titanic_prediction_file.csv',index = False)


# In[ ]:




