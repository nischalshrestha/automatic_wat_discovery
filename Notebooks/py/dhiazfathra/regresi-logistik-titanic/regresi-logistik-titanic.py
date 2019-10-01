#!/usr/bin/env python
# coding: utf-8

# # Regresi Logistik Titanic

# ### Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.api import Logit
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score


# ## Cek data

# In[ ]:


titanic = pd.read_csv("../input/train.csv")


# In[ ]:


titanic.head()


# ### Deskripsi variabel
# 
# Survived: Survived (1) or died (0);  this is the target variable  
# Pclass: Passenger's class (1st, 2nd or 3rd class)    
# Name: Passenger's name  
# Sex: Passenger's sex  
# Age: Passenger's age  
# SibSp: Number of siblings/spouses aboard  
# Parch: Number of parents/children aboard  
# Ticket: Ticket number  
# Fare: Fare  
# Cabin: Cabin  
# Embarked: Port of embarkation

# In[ ]:


titanic.describe()


# Tidak semua fitur berupa numerik.

# In[ ]:


titanic.info()


# ## Persiapan data
# Variabel kategori perlu diubah ke numerik.

# ### Tranformasi fitur embarkment port

# Terdapat tiga pelabuhan: C = Cherbourg, Q = Queenstown, S = Southampton

# In[ ]:


ports = pd.get_dummies(titanic.Embarked , prefix='Embarked')
ports.head()


# Fitur *Embarked* (kategori) ditranformasi ke 3 fitur biner,  (*Embarked_C = 0 not embarked in Cherbourg, 1 = embarked in Cherbourg*)

# In[ ]:


titanic = titanic.join(ports)
titanic.drop(['Embarked'], axis=1, inplace=True) # then drop the original column


# ### Transformasi fitur gender

# In[ ]:


titanic.Sex = titanic.Sex.map({'male':0, 'female':1})


# ### Buang fitur yang tidak perlu (data drop)

# In[ ]:


titanic.drop(['Cabin'], axis=1, inplace=True)
titanic.drop(['Ticket'], axis=1, inplace=True) 
titanic.drop(['Name'], axis=1, inplace=True) 
titanic.drop(['PassengerId'], axis=1, inplace=True)
titanic.info()


# ## Cek nilai yang kosong

# In[ ]:


titanic.isnull().values.any()


# Masih ada nilai kosong.

# In[ ]:


titanic.Age.fillna(titanic.Age.mean(), inplace=True)  # ganti NaN dengan umur rata-rata


# In[ ]:


titanic.isnull().values.any()


# Sudah tidak ada nilai kosong.

# ### Visualisasi

# In[ ]:


titanic['Survived'].value_counts(normalize=True)


# In[ ]:


sns.countplot(x=titanic['Survived'])


# ## Model

# ###  Regresi Logistik

# In[ ]:


titanic_ = add_constant(titanic)


# In[ ]:


model_ = Logit(titanic_['Survived'], titanic_.drop(['Survived'], axis=1))
result = model_.fit(); result.summary()


# In[ ]:


odd_ratio = np.exp(result.params); odd_ratio


# ### Ekstraksi variabel target
# Buat dataframe dengan X berupa masukan dan y berupa target (Survived)

# In[ ]:


y = titanic.Survived.copy() # copy “y” column values out
X = titanic.drop(['Survived'], axis=1) # then, drop y column


# ### Bagi data training dan validasi

# In[ ]:


from sklearn.model_selection import train_test_split
  # 80 % dijadikan untuk training test, 20% dijadikan untuk validation test
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)


# ### Training model

# In[ ]:


model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)


# ### Evaluasi model

# In[ ]:


model.score(X_train, y_train)


# In[ ]:


model.score(X_valid, y_valid)


# ## k-fold validation

# In[ ]:


crossVal = cross_val_score(model, X, y, cv=10); crossVal


# In[ ]:


np.mean(crossVal)

