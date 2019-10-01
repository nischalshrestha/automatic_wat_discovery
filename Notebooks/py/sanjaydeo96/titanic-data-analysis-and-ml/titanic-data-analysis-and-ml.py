#!/usr/bin/env python
# coding: utf-8

# <h1>Would You Survive the Titanic?</h1>

# 
# ## Data Analysis

# ![](https://blog.socialcops.com/wp-content/uploads/2016/07/OG-MachineLearning-Python-Titanic-Kaggle.png)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df1=pd.read_csv('../input/train.csv')
df2=pd.read_csv('../input/test.csv')


# ### 1.Variable Identification

# In[ ]:


df1.head()


# In[ ]:


df2.head()


# ###### VariableType:
# Target:Survived
# 
# Predictors: PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# ###### Variable Category: 
# Categorical: PassengerId,Survived,Pclass,Name,Sex,SibSp,Parch,Ticket,Cabin,Embarked
# 
# Continuous: Age,Fare
# ###### Data Type:
# String:Name,Sex,Cabin,Embarked,Ticket
# 
# Numeric:PassengerId,Survived,Pclass,Age,SibSp,Parch,Fare

# ### 2. Univariate Analysis

# In[ ]:


df1.describe()


# In[ ]:


df1.describe(include=['O'])


# In[ ]:


#Categorical Variable Analysis
df1['Survived'].value_counts().plot.bar(color='r')
plt.show()
df1['Pclass'].value_counts().plot.bar()
plt.show()
df1['Sex'].value_counts().plot.bar(color='g')
plt.show()
df1['SibSp'].value_counts().plot.bar()
plt.show()
plt.show()
df1['Embarked'].value_counts().plot.bar()
plt.show()


# In[ ]:


#Continous Variables Analysis
df1[['Age','Fare']].describe()
# df1['Age'].value_counts().plot.box()
df1['Age'].value_counts().plot.hist( grid=True,color='b',alpha=0.7)
plt.show()
df1['Fare'].value_counts().plot.hist(grid=True,color='r')
plt.show()


# ##### OR

# In[ ]:


df12=df1.fillna(df1.mean())
plt.hist(df12.Age, alpha=.3)
sns.rugplot(df12.Age);
plt.show()


# In[ ]:


plt.hist(df12.Fare)
sns.rugplot(df12.Fare, alpha=.3);
plt.show()


# ### 3.Bivariate Analysis

# ##### Continuous & Continuous

# In[ ]:


plt.figure()
df1.plot.scatter('Age', 'Fare')
plt.show()


# #### OR

# In[ ]:


i=df1[['Age','Fare','Survived']]
plt.style.use('seaborn-colorblind')
pd.tools.plotting.scatter_matrix(i);
plt.show()


# In[ ]:


#correlation matrix to know important features in prediction
corr=df1.corr()
sns.heatmap(corr[['Age','Fare']],annot=True,linewidth=0.1)
plt.show()


# #### Categorical & Categorical

# In[ ]:


freq_1=pd.crosstab(index=df1['Survived'],columns=df1['Pclass'])
freq_2=pd.crosstab(index=df1['Survived'],columns=df1['Sex'])
freq_3=pd.crosstab(index=df1['Survived'],columns=df1['SibSp'])
freq_4=pd.crosstab(index=df1['Survived'],columns=df1['Parch'])
freq_5=pd.crosstab(index=df1['Survived'],columns=df1['Pclass'])
freq_6=pd.crosstab(index=df1['Survived'],columns=df1['Embarked'])
l=[freq_1,freq_2,freq_3,freq_4,freq_5,freq_6]
for i in l:
    print(i)


# In[ ]:


for i in l:
    i.plot(kind='bar',stacked=True,figsize=(4,4))
    # freq_1.plot.bar()
    plt.show()


# In[ ]:



#Overall Correlation
sns.heatmap(df1.corr(),annot=True)
plt.show()


# ### 5.Missing values treatment

# In[ ]:


#counting nan in columns
count_nan = len(df1) - df1.count()
count_nan


# In[ ]:


df1=df1.fillna(df1.mean())
# count_null=len(df22) -df22.count()
# count_null
df2=df2.fillna(df2.mean())
count_nan = len(df1) - df1.count()
count_nan
#We have substituted Age nan value with mean.
#Cabin column doesn't seems to be important and it has highest proportion of nan  value i.e. 0.77 so we drop it


# # 6.Feature Selection and Transformation
# Columns 'Name' and 'Ticket'  are irrelevant as they are unique value which can be referred by PassengerId column. Also Cabin seems to very high number of nan value.So it is wise to drop these three columns.We can further see that Fare column has good correlation with Target variable but it has very strong correlation with Pclass.So we can dropped out Fare column.

# In[ ]:


train=df1.drop(['Ticket','Fare','Embarked','Cabin','Name'] , axis=1)
test=df2.drop(['Ticket','Fare','Embarked','Cabin','Name'] , axis=1)
train.head()


# 
# Conversion of String Values of Sex to Numeric.

# In[ ]:


# def conversion(x):
#     if x == 'male':
#         return 0
#     if x == 'female':
#         return 1
# train['Sex'] =train['Sex'].apply(conversion
#               OR
train['Sex'].replace(['male','female'],[0,1],inplace=True)


# In[ ]:


X=train.drop(['Survived'],axis=1)
y=pd.DataFrame(train['Survived'])
X.head()


# <p> We are going to use test set at  final stage of  testing as it does not have label. </p>

# # Machine Learning

# ### 1. Regression

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,np.ravel(y),test_size=0.3)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression().fit(X_train,y_train)
lr


# In[ ]:


acc_lr=lr.score(X_test,y_test)


# ### 2. KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier().fit(X_train,y_train)
knn


# In[ ]:


acc_knn=knn.score(X_test,y_test)


# ### 3.SVM

# In[ ]:


from sklearn.svm import SVC
svm=SVC().fit(X_train,y_train)
svm_predict=svm.predict(X_test)
svm


# In[ ]:


acc_svm=svm.score(X_test,y_test)


# ### 4.Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier().fit(X_train,y_train)
rf


# In[ ]:


acc_rf=rf.score(X_test,y_test)


# ### Model Evaluation

# In[ ]:


models = pd.DataFrame({
    'Model': ['LogisticRegression', 'KNeighborsClassifier','SVC' ,'RandomForestClassifier' ],
    'Score': [acc_lr, acc_knn, acc_svm, 
              acc_rf]})
models.sort_values(by='Score', ascending=False)

