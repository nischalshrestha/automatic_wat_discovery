#!/usr/bin/env python
# coding: utf-8

# In[115]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[116]:


df = pd.read_csv('../input/train.csv',encoding = "ISO-8859-1",low_memory=False)


# In[117]:


df.describe()


# In[118]:


df['Survived'].mean()


# In[119]:


df.groupby('Pclass').mean()


# In[120]:


class_sex_grouping = df.groupby(['Pclass','Sex']).mean()
class_sex_grouping


# In[121]:


class_sex_grouping['Survived'].plot.bar()


# In[122]:


group_by_age = pd.cut(df["Age"], np.arange(0, 90, 16))
age_grouping = df.groupby(group_by_age).mean()
age_grouping['Survived'].plot.bar()


# In[123]:


df['Sex'].replace(['female','male'],[1,0],inplace=True)


# In[124]:


avg_age = df["Age"].mean()
std_age = df["Age"].std()
count_nan_age = df["Age"].isnull().sum()
count_nan_age


# In[125]:


df['Title'] = None
for index,row in enumerate(df['Name']):
    title = row.split(', ')[1].split('. ')[0]
    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Mr','Rev', 'Sir']:
        df.loc[index, 'Title'] = 'Mr'
    elif title in [ 'Ms', 'Mlle', 'Mme', 'Mrs', 'the Countess','Lady']:
        df.loc[index, 'Title'] = 'Mrs'
    elif title in ['Master']:
        df.loc[index, 'Title'] = 'Master'
    elif title in ['Miss']:
        df.loc[index, 'Title'] = 'Ms'
    else:
        df.loc[index, 'Title'] = 'Other'


# In[126]:


df[['Title','Age']].groupby('Title').mean()


# In[127]:


#mean_age_master = 


# In[128]:


#random_age = np.random.randint(avg_age - std_age,avg_age + std_age,size=count_nan_age)
#random_age


# In[129]:


df[(df["Title"]=='Mr') & (np.isnan(df["Age"]))].count()


# In[130]:


df.loc[(df["Title"]=='Mr') & (np.isnan(df["Age"])),'Age'] = 32.9


# In[131]:


df.loc[(df["Title"]=='Master') & (np.isnan(df["Age"])),'Age'] = 4.6
df.loc[(df["Title"]=='Mrs') & (np.isnan(df["Age"])),'Age'] = 35.5
df.loc[(df["Title"]=='Miss') & (np.isnan(df["Age"])),'Age'] = 31.7
df.loc[(df["Title"]=='Other') & (np.isnan(df["Age"])),'Age'] = 42


# In[132]:


#df["Age"][np.isnan(df["Age"])] = random_age


# In[133]:


df.head()


# In[134]:


df['Fare'].plot()


# In[135]:


sns.distplot(df['Fare'])


# In[136]:


df['Fare_log'] = np.log(df['Fare'] + 1) 


# In[138]:


sns.distplot(df['Fare_log'])


# In[139]:


df['Fare_log'].skew()


# In[140]:


df['Fare_log'].kurt()


# In[107]:


#df['Fare'].fillna(0, inplace=True)


# In[141]:


df['Fare_log'].describe()


# In[147]:


#box plot overallqual/saleprice
var = 'Pclass'
data = pd.concat([df['Fare_log'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="Fare_log", data=data)
fig.axis(ymin=0, ymax=10);


# In[110]:


corr = df.corr()
corr.sort_values(['Survived'], ascending = False, inplace = True)
corr.Survived


# In[ ]:


df[(df['Pclass']==3)]['Fare'].describe()


# In[ ]:


df.loc[(df.Pclass == 3) &(df['Fare'] > 30)] 


# In[ ]:


df.loc[(df['Pclass']==3) & (df['Fare'] > 20),'Fare'] = df[df['Pclass']==3].mean()


# In[ ]:


df.loc[(df['Pclass']==2) & (df['Fare'] > 50),'Fare'] = df[df['Pclass']==2].mean()


# In[ ]:


df.loc[(df['Pclass']==1) & (df['Fare'] > 100),'Fare'] = df[df['Pclass']==1].mean()


# In[ ]:


#box plot overallqual/saleprice
var = 'Pclass'
data = pd.concat([df['Fare'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="Fare", data=data)
fig.axis(ymin=0, ymax=150);


# In[ ]:


df['Fare'].dropna()


# In[ ]:


#df.loc[(df['Pclass']==3) & (df['Fare'].isnull()),'Fare'] = df[df['Pclass']==3].mean()


# In[ ]:


#df.loc[(df['Pclass']==2) & (df['Fare'].isnull()),'Fare'] = df[df['Pclass']==2].mean()


# In[ ]:


df[df['Fare'].isnull()]


# In[ ]:


df = df.drop(df.loc[df['Fare'].isnull()].index)


# In[ ]:


df[df['Fare'].isnull()]


# In[ ]:


df['Fare'].isnull().sum()


# In[ ]:


sns.distplot(df['Fare'])


# In[ ]:


#correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[ ]:


bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
labels=[1,2,3,4,5,6,7,8,9]
df['Age_index']=pd.cut(df.Age, bins=bins,labels=labels)


# In[ ]:


df.head()


# In[ ]:


df[df['Cabin'].notnull()][['Survived','Cabin','Fare']].groupby(['Survived']).agg({'Fare':'mean','Cabin':'count'})


# In[ ]:


df['Cabin_available'] = df['Cabin'].isnull()
df['Cabin_available'].replace([True,False],[0,1],inplace=True)


# In[ ]:


df['Fare']=df['Fare'].round(1)


# In[ ]:


df['Fare']


# In[ ]:





# In[ ]:


x_train_all = df.as_matrix(columns=['Pclass','Sex','Fare','Age_index'])
y_train_all = df.as_matrix(columns=['Survived'])
x_train = x_train_all
y_train = y_train_all
y_train = y_train.reshape(-1)


# In[ ]:


x_train_lr = df.as_matrix(columns=['Sex','Pclass'])
y_train_lr = df.as_matrix(columns=['Survived'])


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=1)
lr.fit(x_train_lr,y_train_lr)
lr.score(x_train_lr,y_train_lr)


# In[ ]:


#nn_clf.score(x_train_lr,y_train_lr)


# In[ ]:


x_train[1]


# In[ ]:


import xgboost as xgb
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(x_train, y_train)
gbm.score(x_train,y_train)
stacked_models_df= pd.DataFrame()
stacked_models_df['gbm'] = gbm.predict(x_train)


# In[ ]:





# In[ ]:


from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(x_train, y_train)
stacked_models_df['svm'] = clf.predict(x_train)


# In[ ]:


clf.predict([[3.,1.,7.2833,38.]])


# In[ ]:





# In[ ]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
random_forest.score(x_train, y_train)
stacked_models_df['rf'] = clf.predict(x_train)


# In[ ]:





# In[ ]:


random_forest.predict([[1.,0.,71.2833,38.]])


# In[ ]:


#from sklearn.model_selection import cross_val_score
#cross_val_score_rf = cross_val_score(random_forest,x_train,y_train,cv=5)
#print(cross_val_score_rf.mean())


# In[ ]:


#cross_val_score_svm = cross_val_score(clf,x_train,y_train,cv=5)
#print(cross_val_score_svm.mean())


# In[ ]:


#from sklearn.linear_model import LogisticRegression
#lr = LogisticRegression(random_state=1)
#lr.fit(x_train,y_train)
#lr.score(x_train,y_train)
#stacked_models_df['lr'] = lr.predict(x_train)


# In[ ]:


stacked_models_df


# In[ ]:


#from sklearn.neural_network import MLPClassifier
#nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                    hidden_layer_sizes=(12,2),max_iter=500, random_state=1,activation='tanh')
#nn_clf.fit(x_train,y_train)


# In[ ]:


#nn_clf.score(x_train,y_train)


# In[ ]:


#cross_val_score(nn_clf,x_train,y_train,cv=10).mean()


# In[ ]:


x_train_stacked=stacked_models_df.values
x_train_stacked.shape


# In[ ]:


y_train.shape


# In[ ]:


from sklearn.neural_network import MLPClassifier
nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(12,2),max_iter=500, random_state=1,activation='tanh')
nn_clf.fit(x_train_stacked,y_train)


# In[ ]:


nn_clf.score(x_train_stacked,y_train)


# In[ ]:


df_test = pd.read_csv('../input/test.csv',encoding = "ISO-8859-1",low_memory=False)


# In[ ]:


df_test.head()


# In[ ]:


df_test['Sex'].replace(['female','male'],[0,1],inplace=True)
avg_age = df_test["Age"].mean()
std_age = df_test["Age"].std()
count_nan_age = df_test["Age"].isnull().sum()
random_age = np.random.randint(avg_age - std_age,avg_age + std_age,size=count_nan_age)
df_test["Age"][np.isnan(df_test["Age"])] = random_age


# In[ ]:


avg_fare = df_test["Fare"].mean()
std_fare = df_test["Fare"].std()
count_nan_fare = df_test["Fare"].isnull().sum()
random_fare = np.random.randint(avg_fare - std_fare,avg_fare + std_fare,size=count_nan_fare)
df_test["Fare"][np.isnan(df_test["Fare"])] = random_fare


# In[ ]:


df_test['Cabin_available'] = df_test['Cabin'].isnull()


# In[ ]:


df_test['Cabin_available'].replace([True,False],[2,3],inplace=True)
df_test[['Pclass','Sex','Parch','Fare','SibSp','Age','Cabin_available']].head()


# In[ ]:


nans = lambda df: df[df.isnull().any(axis=1)]
nans(df_test[['Pclass','Sex','Parch','Fare','SibSp','Age','Cabin_available']])


# In[ ]:


bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
labels=[1,2,3,4,5,6,7,8,9]
df_test['Age_index']=pd.cut(df_test.Age, bins=bins,labels=labels)


# In[ ]:


#x_test = df_test.as_matrix(columns=['Pclass','Sex','Fare','Age_index'])


# In[ ]:


x_test = df_test.as_matrix(columns=['Sex'])


# In[ ]:


x_test.shape


# In[ ]:


#box plot overallqual/saleprice
var = 'Pclass'
data = pd.concat([df_test['Fare'], df_test[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="Fare", data=data)
fig.axis(ymin=0, ymax=150);


# In[ ]:


stacked_model_pred=pd.DataFrame()


# In[ ]:


#stacked_model_pred['gbm']=gbm.predict(x_test)


# In[ ]:


#stacked_model_pred['clf']=clf.predict(x_test)


# In[ ]:


#stacked_model_pred['lr']=lr.predict(x_test)


# In[ ]:


#stacked_model_pred['rf']=random_forest.predict(x_test)


# In[ ]:


#stacked_model_pred.head()


# In[ ]:


#x_test_stacked = stacked_model_pred.values


# In[ ]:


test_predictions = lr.predict(x_test)


# In[ ]:


#test_predictions = gbm.predict(x_test)


# In[ ]:


lr.score(x_train_lr,y_train_lr)


# In[ ]:


test_predictions=test_predictions.astype(int)
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": test_predictions
    })
submission.head()


# In[ ]:


submission.to_csv("survival_submission_v9.csv", index=False)


# In[ ]:




