#!/usr/bin/env python
# coding: utf-8

# In[36]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[37]:


# Importing the  dataset
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")

print("Dimensions of train: {}".format(train.shape))
print("Dimensions of test: {}".format(test.shape))


# In[38]:


train.head()


# In[39]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

sex_pivot = train.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()


# In[40]:


class_pivot = train.pivot_table(index="Pclass",values="Survived")
class_pivot.plot.bar()
plt.show()


# In[41]:


train["Age"].describe()


# In[42]:


survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()


# In[43]:


def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)

pivot = train.pivot_table(index="Age_categories",values='Survived')
pivot.plot.bar()
plt.show()


# In[44]:


train["Pclass"].value_counts()


# In[45]:


def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

for column in ["Pclass","Sex","Age_categories"]:
    train = create_dummies(train,column)
    test = create_dummies(test,column)


# In[46]:


holdout = test


# In[47]:


columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior','Survived']


# In[48]:


#all_X = train[columns]
#all_y = train['Survived']


# In[49]:


import turicreate as tc


# In[50]:


get_ipython().system(u' mkdir temp')


# In[51]:


tc.config.set_runtime_config('TURI_CACHE_FILE_LOCATIONS', 'temp')


# In[52]:


all_data=train[columns]


# In[53]:


all_data.head()


# In[54]:


data =  tc.SFrame(all_data)


# In[55]:


# Splitting the data into training (80% of data) and testing (20% of data) sets.
train_data, test_data = data.random_split(0.8)


# In[56]:


train_data.explore


# In[57]:


model = tc.classifier.create(train_data, target='Survived',features = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male','Age_categories_Missing','Age_categories_Infant','Age_categories_Child', 'Age_categories_Teenager','Age_categories_Young Adult', 'Age_categories_Adult','Age_categories_Senior'], verbose =False)


# In[58]:


# Get predictions using the test data 
predictions = model.classify(test_data)


# In[59]:


predictions


# In[60]:


# obtain statistical results for the model by model.evaluate method 
results = model.evaluate(test_data)


# In[61]:



results


# In[62]:


data2= tc.SFrame(all_data)


# In[63]:


model = tc.classifier.create(data2, target='Survived',features = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male','Age_categories_Missing','Age_categories_Infant','Age_categories_Child', 'Age_categories_Teenager','Age_categories_Young Adult', 'Age_categories_Adult','Age_categories_Senior'], verbose = False)


# In[64]:


columns2 = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']


# In[65]:


data2= tc.SFrame(holdout[columns2])


# In[66]:


holdout_predictions = model.predict(data2)


# In[67]:


holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)


# In[68]:


submission.to_csv("submission.csv",index=False)


# In[ ]:





# In[ ]:




