#!/usr/bin/env python
# coding: utf-8

# ## Titanic Dataset 

# ### Importing libraries

# In[ ]:


import pandas as pd
import numpy as np


# ### Train and test Data

# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u"df_train=pd.read_csv('../input/train.csv')")


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u"df_test=pd.read_csv('../input/test.csv')")


# ### info describes the features

# In[ ]:


df_train.info()


# ### df.describe() tells the numeric values

# In[ ]:


df_train.describe() ##returns only numeric value


# In[ ]:


df_test.describe()


# ### df.describe() tells the numeric values and categorical values

# In[ ]:


df_train.describe(include='all')  ##returns all numeric and categorical features


# In[ ]:


df_train.head()


# ### checking null values and return the sum

# In[ ]:


df_train.isnull().sum()  ##age,cabin,embarked has null values


# ### df.head() gives first n rows

# In[ ]:


df_train.head()


# ### df.shape gives the rows x cols

# In[ ]:


print(df_train.shape)
print(df_test.shape)


# In[ ]:


df_test.head()


# ### uniques returns the unique values

# In[ ]:


df_train['Survived'].unique()


# ### df.columns returns the columns of a dataframe

# In[ ]:


df_train.columns


# ### dropping the columns

# In[ ]:


df_train=df_train.drop(['Ticket'],axis=1)


# In[ ]:


df_train=df_train.drop(['Name'],axis=1)


# In[ ]:


df_train=df_train.drop(['PassengerId'],axis=1)


# In[ ]:


df_train.columns


# In[ ]:


df_train.head()


# ### giving the last rows

# In[ ]:


df_train.tail()


# #### filling null values in Age column

# In[ ]:


mean_value=df_train['Age'].mean()
print(mean_value)


# In[ ]:


df_train['Age']=df_train['Age'].fillna(int(mean_value))
print(df_train.head())
print(df_train.sample(6))
print(df_train.tail())


# #### checking null values

# In[ ]:


df_train.isnull().sum()


# #### converting string columns to categorical columns

# In[ ]:


df_train['Embarked']=df_train['Embarked'].astype('category') 
df_train['Sex']=df_train['Sex'].astype('category')


# In[ ]:


df_train.dtypes


# In[ ]:


df_train=df_train.drop(['Cabin'],axis=1)


# In[ ]:


df_train.dtypes


# cat_columns=df.select_dtypes(['category']).columns 
# df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
# df

# In[ ]:


cat_columns=df_train.select_dtypes(['category']).columns
df_train[cat_columns]=df_train[cat_columns].apply(lambda x: x.cat.codes)


# In[ ]:


df_train.head()


# In[ ]:


df_train.dtypes


# #### contains two cols

# In[ ]:


df_train[cat_columns].sample(5)


# ### Test data

# In[ ]:


print(df_test.shape)


# In[ ]:


df_test.head()


# In[ ]:


df_test=df_test.drop(['Ticket'],axis=1)
df_test=df_test.drop(['Name'],axis=1)
df_test=df_test.drop(['Cabin'],axis=1)


# In[ ]:


df_test.head()


# In[ ]:


df_test.isnull().sum()


# In[ ]:


#del df_test


# In[ ]:


mean_val=df_test['Age'].mean()
print(mean_val)


# In[ ]:


df_test['Age']=df_test['Age'].fillna(int(mean_val))


# In[ ]:


print(df_test.head())
print(df_test.sample(6))
print(df_test.tail())


# In[ ]:


df_test.head()


# In[ ]:


df_test['Embarked'] = df_test['Embarked'].astype('category')
df_test['Sex'] = df_test['Sex'].astype('category')
 


# In[ ]:


df_test.dtypes


# In[ ]:


df_test.sample(4)


# In[ ]:


cat_col=df_test.select_dtypes(['category']).columns
df_test[cat_col]=df_test[cat_col].apply(lambda x: x.cat.codes)


# In[ ]:


df_test.head()


# In[ ]:


df_test.isnull().sum()


# In[ ]:


df_test.sample(10)


# In[ ]:


df_test=df_test.replace(np.nan,1)


# In[ ]:


df_test.isnull().sum()


# In[ ]:


df_train.head()


# In[ ]:


x_train=df_train.iloc[:,1:8]
print(x_train.head())
print(x_train.shape)


# In[ ]:


y_train=df_train.iloc[:,0:1]


# In[ ]:


y_train.head()


# In[ ]:


df_test.shape


# In[ ]:


df_test.head()


# In[ ]:


df=pd.DataFrame()


# In[ ]:


df=df_test.iloc[:,0:1]


# In[ ]:


df.head()


# In[ ]:


df_test=df_test.drop(['PassengerId'],axis=1)


# In[ ]:


df_test.shape


# In[ ]:


x_test=df_test.iloc[:,:]


# In[ ]:


x_test.head()


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'from sklearn.ensemble import RandomForestClassifier \nrf= RandomForestClassifier(random_state=0,n_estimators=10)\nrf')


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'rf.fit(x_train,y_train)')


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'print(rf.score(x_train,y_train))')


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'x_test_result=rf.predict(x_test)')


# In[ ]:


x_test_result


# In[ ]:


x_test_df=pd.DataFrame(x_test_result)


# In[ ]:


x_test_df=x_test_df.rename({0:"Survived"},axis=1)
# df.rename({1: 2, 2: 4}, axis='index')


# In[ ]:


x_test_df.sample(4)


# In[ ]:


df_test.columns


# In[ ]:


x_test_df1=df_test.iloc[:,0:1]


# In[ ]:


df_test.sample(2)


# In[ ]:


x_test_df1.sample(3)


# In[ ]:


x_test_final=pd.concat([x_test_df1,x_test_df],axis=1)


# In[ ]:


x_test_final.head()


# In[ ]:


x_test_final.to_csv('gender_submission.gz',index=False,compression='gzip')

