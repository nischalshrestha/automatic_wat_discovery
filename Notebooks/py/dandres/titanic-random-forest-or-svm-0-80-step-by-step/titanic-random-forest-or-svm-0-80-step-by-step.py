#!/usr/bin/env python
# coding: utf-8

# # TITANIC - RANDOM FOREST OR SVM 0.80 (STEP BY STEP)

# ## LIBRARIES TO IMPORT

# In[ ]:


import numpy as np # NUMPY
import pandas as p # PANDAS

# DATA VIZUALIZATION LIBRARIES
from matplotlib import pyplot as plt
import seaborn as sns 

# METRICS TO MEASURE ACCURACY
from sklearn.model_selection import cross_validate
from sklearn import metrics 

from sklearn.linear_model import LogisticRegression #LOGISTIC REGRESSION
from sklearn import svm #SVM
from sklearn.ensemble import RandomForestClassifier # RANDOM FOREST


# ## READING DATAFRAMES

# In[ ]:


# SETTING PASSENGER ID IN INDEX COLUMN  
df1 = p.read_csv("../input/train.csv",index_col='PassengerId')
df2 = p.read_csv("../input/test.csv",index_col='PassengerId')


# ## FIST OVERVIEW IN DATA

# In[ ]:


df1.head(5) # DF1 HAS SURVIVED COLUMN


# In[ ]:


df2.head(5) # DF2 IS WITHOUT SURVIVED COLUMN


# In[ ]:


# CREATING A DATAFRAME ALL WITH WHOLE FEATURES DATA (TRAIN + TEST) 
df_features = p.concat([df1.iloc[:,1:],df2])


# In[ ]:


# 1309 ROWS AND 10 COLUMNS 
df_features.shape 


# In[ ]:


#NUMBER OF NAN IN EACH COLUMNS
df_features.isnull().sum() 


# In[ ]:


# GOOD GRAPH TO FIGURE OUT HOW NAN´s ARE DISTRUBUTED
sns.heatmap(df_features.isnull().astype(int))  


# In[ ]:


# GOOD GRAPH TO FIGURE OUT THE CORRELATIONS BETWEEN COLUMNS
sns.heatmap(df_features.corr(),annot=True,fmt=".2f")


# ## EXPLORATORY DATA

# ------------
# **-FEATURE ANALYSIS: **"Sex"

# In[ ]:


df1['Sex'].value_counts()


# In[ ]:


# CREATING A DICTIONARY OF SEX
dict_sex = {'male':1,'female':2}
dict_sex


# In[ ]:


sns.countplot(x='Survived',hue='Sex', data=df1, palette="coolwarm")


# **CONCLUSION 1: ** SURVIVOR RATE OF FEMALE IS HIGHER THAN MALE
# 
# -----------------

# -----------------
# **-FEATURE ANALYSIS: **"Pclass"

# In[ ]:


df1['Pclass'].value_counts()


# In[ ]:


sns.countplot(x='Survived',hue='Pclass', data=df1, palette="Blues")


# **CONCLUSION 2:** CLASS 3 HAS THE LOWEST SUVIVOR RATE
# 
# --------------------------------------

# -----------------------------
# **-FEATURE ANALYSIS: **"Age"

# In[ ]:


# COOL GRAPH TO FIGURE OUT THE AGE INFLUENCE IN SURVIVOR
sns.swarmplot(x='Survived',y='Age',hue='Sex', data=df1, palette="RdBu_r")


# **CONCLUSION 3: **
# 
#             SURVIVOR RATE OF [AGE < 10] IS ALMOST THE SAME FOR "MALE" AND "FEMALE"
#             SURVIVOR RATE OF [AGE > 10] IS HIGHER FOR "FEMALE" THAN "MALE"
#    -------------------------------------------------------

# -----------------------------
# **FEATURE ANALYSIS: ** "Embarked"

# In[ ]:


# CREATING A DICTIONARY OF EMBARKED
dict_embarked = {'S':1,'Q':2,'C':3}
dict_embarked


# In[ ]:


df1['Embarked_Numerical'] = df1['Embarked'].apply(lambda x: dict_embarked[x] if p.notnull(x) else x) 
sns.countplot(x='Survived',hue='Embarked_Numerical', data=df1, palette="Blues")
df1 = df1.drop(['Embarked_Numerical'], axis=1)


# **CONCLUSION 4:** EMBARKED='C' HAS THE HIGHEST SUVIVOR RATE
# 
# -----------------------------

# -------------------------------
# **-FEATURE ANALYSIS: **"Fare"

# In[ ]:


sns.swarmplot(x='Survived',y='Fare', data=df1, palette="RdBu")


# **CONCLUSION 5: ** NOTHING
# 
# ----------------------

# ---------------------------------------
# **-FEATURE ANALYSIS: **"Name"

# In[ ]:


# GETTING PERSONAL TITLES
df_features['Name'].head(5)


# In[ ]:


# PAY ATENTION IN THIS STRING PATTERN TO GET PERSONAL TITLE
S1 = df_features['Name'].apply(lambda x: x.split(',')[1].split(' ')[1])
S1.head(5)


# In[ ]:


# GETTING MOST USED PERSONAL TITLES
S2 = S1.value_counts()[S1.value_counts()>1]
S2.head(5)


# In[ ]:


# CREATING A DICTIONARY OF PERSONAL TITLES
dict_names = {S2.index[i]:i+1 for i in np.arange(0,S2.shape[0])}


# ----------------------------
# **-FEATURE ANALYSIS: **"Ticket"

# In[ ]:


# THERE ARE PEOPLE WITH THE SAME TICKET 
df_features['Ticket'].value_counts().head(5)


# In[ ]:


# CREATING A DICTIONARY OF TICKETS
S = df_features['Ticket'].value_counts()[df_features['Ticket'].value_counts() > 1]
dict_tickets = {S.index[i]:i+1 for i in np.arange(0,S.shape[0])}


# ---------------------------------
# **FEATURE ANALYSIS: **"Cabin"

# In[ ]:


# THERE ARE PEOPLE WITH THE SAME CABIN
df_features['Cabin'].value_counts().head(5)


# In[ ]:


# WE CAN GET ONLY THE FIRST LETTER OF CABIN
S1 = df_features['Cabin'].apply(lambda x: x[0] if p.notnull(x) else x).value_counts()
S1


# In[ ]:


# CREATING A DICTIONARY OF THE FIST LETTERS OF CABIN
dict_cabin1 = {S1.index[i]:i+1 for i in np.arange(0,S1.shape[0])}
dict_cabin1


# In[ ]:


# WE CAN GET ONLY CABIN TO GROUP
S2 = df_features['Cabin'].value_counts()[df_features['Cabin'].value_counts() > 1]
S2.head(5)


# In[ ]:


dict_cabin2 = {S2.index[i]:i+1 for i in np.arange(0,S2.shape[0])}


# -------------------------------
# 
# **FEATURE ANALYSIS: ** "SibSp" and "Parch"

# In[ ]:


sns.countplot(x='Survived',hue='SibSp', data=df1, palette="Blues")


# In[ ]:


sns.countplot(x='Survived',hue='Parch', data=df1, palette="Greens")


# ## DATA CLEAN

# IN THIS PART WE WILL TRANSFORM THE INPUT IN NUMERICAL VALUES AND COMPLETE NAN´s

# In[ ]:


# APPLYING SEX DICTIONARY IN "SEX"
df_features['Sex'] = df_features['Sex'].apply(lambda x: dict_sex[x] if x in dict_sex.keys() else 0)


# In[ ]:


# APPLYING CABIN DICTIONARIES IN "CABIN"
df_features['Cabin1'] = df_features['Cabin'].apply(lambda x: dict_cabin1[x[0]] if p.notnull(x) else 0)
df_features['Cabin2'] = df_features['Cabin'].apply(lambda x: dict_cabin2[x] if x in dict_cabin2.keys() else 0)
df_features = df_features.drop(['Cabin'], axis=1)


# In[ ]:


# APPLYING PERSONAL TITLE DICTIONARY IN "NAME"
df_features['Name'] = df_features['Name'].apply(lambda x: dict_names[x.split(',')[1].split(' ')[1]] if x.split(',')[1].split(' ')[1] in dict_names.keys() else 0)


# In[ ]:


# APPLYING TICKET DICTIONARY IN "TICKET"
df_features['Ticket'] = df_features['Ticket'].apply(lambda x: dict_tickets[x] if x in dict_tickets.keys() else 0)


# In[ ]:


# COMPLETING AGES WITH A GROUPBY MEDIAN OF SEX AND PERSONAL TITLE
df_features['Age'] = df_features.groupby(['Sex','Name'])['Age'].apply(lambda x: x.fillna(x.median()))


# In[ ]:


# COMPLETING FARE WITH A GROUPBY MEDIAN OF SEX AND PCLASS
df_features['Fare'] = df_features.groupby(['Sex','Pclass'])['Fare'].apply(lambda x: x.fillna(x.median()))


# In[ ]:


# APPLYING EMBARKED DICTIONARY IN "EMBARKED"
df_features['Embarked'] = df_features['Embarked'].apply(lambda x: dict_embarked[x] if x in dict_embarked.keys() else 0)
# COMPLETING EMBARKED WITH A GROUPBY MEDIAN OF SEX AND PCLASS
df_features['Embarked'] = df_features.groupby(['Sex','Pclass'])['Embarked'].apply(lambda x: x.fillna(x.median()))


# In[ ]:


# CREATING A NEW CATEGORY OF SEX FOR CHILDREN ('3') - SEE "CONCLUSION 3"
df_features['Sex'] = df_features[['Sex','Age']].apply(lambda row: row['Sex'] if row['Age']>=10 else 3 ,axis=1)


# In[ ]:


df_features.head(5)


# In[ ]:


#FILLING NAN WITH MEDIAN
df_features = df_features.fillna(df_features.median())

#NORMALIZING
df_features=(df_features - df_features.mean())/df_features.std() 

df_features.head(5)


# ## DATA MODELING

# In[ ]:


# SPLIT AGAIN IN DATA TRAINING AND DATA TESTING
df_train = p.concat([df1.iloc[:,0],df_features.iloc[:df1.shape[0],:]],axis=1)
df_test = df_features.iloc[df1.shape[0]:,:]


# In[ ]:


# DEFINING X_train, X_test, Y_train
X_train = df_train.iloc[:,1:]
Y_train = df_train.iloc[:,0]
X_test = df_test.iloc[:,:]


# In[ ]:


# RANDOM FOREST - YOU CAN DO A GRID SEARCH AND IMPROVE YOUR ACCURACY
mdl = RandomForestClassifier(max_depth=5, n_estimators=30,bootstrap=True)

# OTHER MODELS
#mdl = linear_model.LogisticRegression() # YOU CAN REACH 0.75 OF ACCURACY
#mdl = svm.SVC(kernel='rbf') # YOU CAN REACH 0.79 OF ACCURACY


# In[ ]:


# FIT
mdl.fit(X_train, Y_train)
# CALCULATING Y_predicted IN SAMPLE
Y_predicted = mdl.predict(X_train)


# In[ ]:


# ACCURACY IN SAMPLE
metrics.accuracy_score(Y_train, Y_predicted)


# In[ ]:


# COMPARING ACCURACY IN SAMPLE WITH AVERAGE ACCURACY IN CROSS VALIDATION (OVERFIT TEST)
scores = cross_validate(mdl, X_train, Y_train, scoring='accuracy', cv=10, return_train_score=False)
scores = scores['test_score']

print("Accuracy")
print("Min: %.3f" %scores.min()," Max: %.3f" %scores.max(), " Avg: %.3f" %scores.mean())
print(scores)


# In[ ]:


#YOU CAN REMOVE COLUMNS TO IMPROVE YOUR SOLUTION ("mdl.feature_importances_" IS FOR RANDOM FOREST)
dict_importance = {df_features.columns[i]:mdl.feature_importances_[i] for i in np.arange(0,df_features.columns.shape[0])}
sorted(dict_importance.items(), key=lambda x: x[1])


# In[ ]:


# CALCULATING Y_test OUT OF SAMPLE
Y_test = mdl.predict(X_test)


# In[ ]:


results = p.DataFrame(data={"PassengerId":df_test.index,"Survived":Y_test})
results.to_csv("output_RF.csv",index=False)


# In[ ]:




