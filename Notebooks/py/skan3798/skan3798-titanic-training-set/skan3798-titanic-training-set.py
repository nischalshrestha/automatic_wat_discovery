#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().magic(u'pylab inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/train.csv')
df.columns


# In[ ]:


df.Survived.value_counts()


# In[ ]:


###PClass vs. Survived ###
df.Pclass.value_counts().plot(kind='bar')


# In[ ]:


df[df.Pclass.isnull()]


# In[ ]:


df.Pclass.hist()


# In[ ]:


#since unable to compare string to series values, convert Pclass to str
df['Pclass'] = df['Pclass'].astype(str)
#now compare Pclass 3 to survival rate
df[df.Pclass=="3"].Survived.value_counts()


# In[ ]:


119/(372+119)


# In[ ]:


#comparing Pclass 2 to survival rate
df[df.Pclass=="2"].Survived.value_counts()


# In[ ]:


87/(97+87)


# In[ ]:


#comparing Pclass 1 to survival rate
df[df.Pclass=="1"].Survived.value_counts()


# In[ ]:


136/(136+80)


# **The survival rate for Pclasses:
# * 1 = 62.96%
# * 2 = 47.28%
# * 3 = 24.24%
# **

# In[ ]:


###Sex vs. Survival###
df.Sex.value_counts().plot(kind='bar')


# In[ ]:


df[df.Sex.isnull()] #no missing values


# In[ ]:


#comparing Male to survival rate
df[df.Sex=="male"].Survived.value_counts()


# In[ ]:


109/(468+109)


# In[ ]:


#comparing Female to survival rate
df[df.Sex=="female"].Survived.value_counts()


# In[ ]:


233/(233+81)


# The survival rate for Sex:
# * male = 18.89%
# * female = 74.20

# In[ ]:


###Age vs. Survival ###
df.Age.value_counts().plot(kind='bar')


# In[ ]:


df[df.Age.isnull()] #there is missing data for age


# In[ ]:


df[df.Age.isnull()].Survived.value_counts()


# In[ ]:


125/(52+125)


# Of the individuals with no given age,  70.62% of individuals survived

# In[ ]:


#test, filling the null Ages with average age and checking survival rate impact
avgAge = df.Age.mean()
print(avgAge)


# In[ ]:


df.Age = df.Age.fillna(value=avgAge)
df[df.Age.isnull()] #check that there are now no null ages


# In[ ]:


df   #from looking at the data table, we see any null ages are changed to 	29.6991176471


# In[ ]:


df[df.Age <"15"].Survived.value_counts()


# In[ ]:


19/(19+11)


# In[ ]:


df[(df.Age >= "15") & (df.Age <"29.6991176471")].Survived.value_counts()


# In[ ]:


114/(114+202+114)


# In[ ]:


df[(df.Age <= "50") & (df.Age >"29.6991176471")].Survived.value_counts()


# In[ ]:


123/(123+153)


# In[ ]:


df[(df.Age > "50") & (df.Age <"70")].Survived.value_counts()


# In[ ]:


29/(44+29)


# In[ ]:


df[df.Age >= "70"].Survived.value_counts()


# In[ ]:


5/(5+14)


# In[ ]:


df[df.Age =="29.6991176471"].Survived.value_counts()


# In[ ]:


52/(125+52)


# Survival Rate for Age:
# * <15 = 63.33%
# * 15-Avg = 26.51%
# * Avg = 29.38%
# * Avg-50 = 44.57%
# * 50-70 = 39.73%
# * >70 = 26.32%

# In[ ]:


###Fare vs. Survival###


# In[ ]:


df.Fare.value_counts().plot(kind='barh')


# In[ ]:


df[df.Fare.isnull()] #no null entries for fare :)


# In[ ]:


df[df.Fare==0].Survived.value_counts()


# In[ ]:


1/15


# In[ ]:


df[(df.Fare >0)&(df.Fare <10)].Survived.value_counts()


# In[ ]:


66/(255+66)


# In[ ]:


df[(df.Fare >=10)&(df.Fare <20)].Survived.value_counts()


# In[ ]:


76/(103+76)


# In[ ]:


df[(df.Fare >=20)&(df.Fare <30)].Survived.value_counts()


# In[ ]:


58/(58+78)


# In[ ]:


df[(df.Fare >=30)&(df.Fare <40)].Survived.value_counts()


# In[ ]:


28/(36+28)


# In[ ]:


df[(df.Fare >=40)&(df.Fare <50)].Survived.value_counts()


# In[ ]:


4/15


# In[ ]:


df[(df.Fare >=50)&(df.Fare <60)].Survived.value_counts()


# In[ ]:


27/(27+12)


# In[ ]:


df[(df.Fare >=60)&(df.Fare <70)].Survived.value_counts()


# In[ ]:


6/17


# In[ ]:


df[(df.Fare >=70)].Survived.value_counts()


# In[ ]:


76/(76+29)


# Fare vs. Survival:
# * 0 = 6.67%
# * 0-10 = 20.56%
# * 10-20 = 42.46%
# * 20-30 = 43.75%
# * 30-40 = 43.75%
# * 40-50 = 26.67%
# * 50-60 = 69.23%
# * 60-70 = 35.29%
# >70 = 72.38%

# In[ ]:


df[(df.Fare >=60)&(df.Fare < 70)].Sex.value_counts()


# In[ ]:


df[(df.Age > "70")].Sex.value_counts()


# In[ ]:


df[(df.Age <= "50") & (df.Age >"29.6991176471")].Sex.value_counts()


# In[ ]:


df[df.Pclass=='1'].Sex.value_counts()


# In[ ]:


df[df.Pclass=='2'].Sex.value_counts()


# In[ ]:


df[(df.Pclass=='3')&(df.Sex=='female')].Survived.value_counts()


# In[ ]:


df[(df.Pclass=='2')&(df.Sex=='female')].Survived.value_counts()


# In[ ]:


df[(df.Pclass=='1')&(df.Sex=='female')].Survived.value_counts()


# In[ ]:


df[(df.Age >='50')&(df.Age <'70')&(df.Sex=='female')].Survived.value_counts()


# In[ ]:


df[(df.Age >'29.6991176471')&(df.Age <'50')&(df.Sex=='female')].Survived.value_counts()


# In[ ]:


df[(df.Age =='29.6991176471')&(df.Sex=='female')].Survived.value_counts()


# In[ ]:


df[(df.Age <'15')& (df.Sex=='female')].Survived.value_counts()


# In[ ]:


#predicting using TEST.CSV
df_test = pd.read_csv('../input/test.csv')
df_test.columns


# In[ ]:


df_test.insert(2,'Survived',0)

df_test.columns


# In[ ]:


df_test.loc[(df_test.Sex == 'female')&((df_test.Pclass == 1)|(df_test.Pclass == 2)|((df_test.Age >= 50)&(df_test.Age < 70)))]


# In[ ]:


df_test.loc[((df_test.Sex == 'female')&((df_test.Pclass == 1)|(df_test.Pclass == 2)|((df_test.Age >= 50)&(df_test.Age < 70)))),'Survived'] = 1


# In[ ]:


df_test.loc[(df_test.Sex == 'female')&((df_test.Pclass == 1)|(df_test.Pclass == 2)|((df_test.Age >= 50)&(df_test.Age < 70)))]


# In[1]:


submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": df_test["Survived"]
})
submission.to_csv('titanic_submission.csv',index=False)
print

