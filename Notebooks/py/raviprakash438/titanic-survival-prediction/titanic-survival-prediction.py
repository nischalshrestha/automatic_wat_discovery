#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix,classification_report
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df=pd.read_csv('../input/train.csv')
gen_df=pd.read_csv('../input/gender_submission.csv')
test_df=pd.read_csv('../input/test.csv')
#Putting train data and test in array so that data cleaning and new feature creation will be done in one go.
data_arr=[train_df,test_df]


# In[ ]:


train_df.info()
#print(test_df.head())
#test_df.info()


# In[ ]:


#Lets figure out which columns have null values.
colms=[col for col in train_df.columns if train_df[col].isnull().any()]
#colms=[col for col in test_df.columns if test_df[col].isnull().any()]


# In[ ]:


colms


# In[ ]:


#Lets visualize null values in columns.
#plt.subplots(figsize=(9,6))
#sns.heatmap(train_df.isnull())
#sns.heatmap(test_df.isnull())


# Logically we should drop the Cabin column since it has very less value but lets replace the null value with X and visualise the servial prediction.

# In[ ]:


train_df['Cabin']=pd.Series([i[0] if pd.notnull(i) else 'X' for i in train_df['Cabin'] ])
train_df['Cabin'].replace('T','X',inplace=True) # I am replacing cabin T with X since there is no T cabin in test data.
test_df['Cabin']=pd.Series([i[0] if pd.notnull(i) else 'X' for i in test_df['Cabin'] ])


# In[ ]:


print(train_df.Cabin.value_counts())
#print(test_df.Cabin.unique())


# In[ ]:


sns.factorplot(y='Survived',x='Cabin',data=train_df,kind='bar')
plt.show()


# As you can see from the above plot, the passangers who were in cabins are more likly to survived as compared to non cabin passangers(X).

# In[ ]:


#sns.heatmap(train_df.isnull())
for data in data_arr:
    data['Title']=data.Name.str.split(', ',expand=True)[1].str.split('. ',expand=True)[0]
    title_cnt=data.Title.value_counts()<10
    data.Title=data.Title.apply(lambda x: x if title_cnt[x]==False else 'Misc')
    


# In[ ]:


train_df.head()


# In[ ]:


#Lets update the null values in age.
med_age=pd.DataFrame()
def fill_age(cols):
    pclass=cols[0]
    sex=cols[1]
    age=cols[2]
    title=cols[3]
    if pd.isnull(age):
        return med_age[(med_age['Pclass']==pclass) & (med_age['Title']==title) & (med_age['Sex']==sex)]['Age']
    else:
        return age


# In[ ]:


for dataset in data_arr:
    med_age=dataset.groupby(['Pclass','Title','Sex'])['Age'].median().reset_index()
    dataset['Age']=dataset[['Pclass','Sex','Age','Title']].apply(fill_age,axis=1)
    #mode()[0] will get the most common value among the group
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    dataset['FamilySize']=dataset.SibSp+dataset.Parch+1
    dataset['IsAlone']=1
    dataset['IsAlone'].loc[dataset['FamilySize']>1]=0
    #distributing fare in almost equally in 4 group
    dataset['FareBin']=pd.qcut(dataset['Fare'],4,labels=[1,2,3,4])
    #Grouping age in 5 equal intervals
    dataset['AgeBin']=pd.cut(dataset['Age'],5,labels=[1,2,3,4,5])
    


#  **Lets do some data visialization**

# In[ ]:


# 1.How many passanger survived.
print(train_df.Survived.value_counts())
sns.countplot(x='Survived',data=train_df)
plt.show()


# In[ ]:


#How many male and female survived.
print(train_df.groupby('Sex')['Survived'].sum())
sns.countplot(x='Survived',hue='Sex',data=train_df)
plt.show()


# In[ ]:


#Find the survivers based on Class
print(train_df.groupby('Pclass')['Survived'].sum())
sns.countplot(x='Survived',hue='Pclass',data=train_df)
plt.show()


# In[ ]:


#Find the Survivers based on Title
print(train_df.groupby('Title')['Survived'].sum())
sns.countplot(x='Survived',hue='Title',data=train_df)
plt.show()


# In[ ]:


#Find the survivers who were alone
print(train_df.groupby('IsAlone')['Survived'].sum())
sns.countplot(x='Survived',hue='IsAlone',data=train_df)
plt.show()


# In[ ]:


#find the survivers based on family size
print(train_df.groupby('FamilySize')['Survived'].sum())
sns.countplot(x='Survived',hue='FamilySize',data=train_df)
plt.legend(loc=1) #moving the legned to the right
plt.show()


# In[ ]:


train_df.Age=train_df.Age.astype(int)
#Plot a age distribution of passangers who survived.
ageplt=sns.FacetGrid(train_df,hue='Survived',aspect=4)
ageplt.map(sns.kdeplot,'Age',shade=True)
ageplt.set(xlim=(0,train_df.Age.max()))
ageplt.add_legend()
plt.show()


# In[ ]:


#Plot a fare distribution of passangers who survived
fareplt=sns.FacetGrid(train_df,hue='Survived',aspect=5)
fareplt.map(sns.kdeplot,'Fare',shade=True)
fareplt.set(xlim=(0,train_df.Fare.max()))
fareplt.add_legend()
plt.show()


# In[ ]:


#print(test_df.head())
#print(train_df.info())
#Lets see the corelation matrix cofficient.
#train_df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(train_df.corr(),annot=True)
plt.show()


# In[ ]:


train_df.drop(['PassengerId','Ticket','Name','Fare','Age','SibSp','Parch'],axis=1,inplace=True)
test_df.drop(['Ticket','Name','Fare','Age','Parch','SibSp'],axis=1,inplace=True)


# In[ ]:


#train=train_df[['Survived','Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare']]
#test=test_df[['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare']]


# In[ ]:


#train_df.drop('FamilySize',axis=1,inplace=True)
#train.info()
#train_df.FamilySize.unique()
train_df.head()


# In[ ]:


#Lets incode the Sex and Embarked columns
train_df=pd.get_dummies(train_df,columns=['Sex','Embarked','Pclass','Title','AgeBin','FareBin','Cabin'],drop_first=True)
test_df=pd.get_dummies(test_df,columns=['Sex','Embarked','Pclass','Title','AgeBin','FareBin','Cabin'],drop_first=True)


# In[ ]:


train_df.corr()['Survived']


# In[ ]:


#test.Age=test.Age.astype(int)
train_df.info()


# In[ ]:


#Lets seperate input data and label data.
y=train_df['Survived']
X=train_df.iloc[:,1:] #taking all the columns except first.
PassengerId=test_df['PassengerId']
test_df.drop(labels=['PassengerId'],inplace=True,axis=1)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=1)


# **Predictions with Classification algorithms**

# *1. XGBoost*

# In[ ]:


from xgboost import XGBClassifier
modelxgb=XGBClassifier(n_estimators=300,learning_rate=0.001,max_depth=4,n_jobs=4,)
modelxgb.fit(X_train,y_train)
ypred=modelxgb.predict(X_test)
print(modelxgb.score(X_train,y_train))
print(confusion_matrix(y_test,ypred))
print(classification_report(y_test,ypred))


# *2. Logistic Regression*

# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(max_iter=100)
logmodel.fit(X_train,y_train)
ypred=logmodel.predict(X_test)
print(logmodel.score(X_train,y_train))
print(confusion_matrix(y_test,ypred))
print(classification_report(y_test,ypred))


# *4. SVM*

# In[ ]:


from sklearn.svm import SVC
modelsvc=SVC(probability=True,gamma='auto')
modelsvc.fit(X_train,y_train)
ypred=modelsvc.predict(X_test)
print(modelsvc.score(X_train,y_train))
print(confusion_matrix(y_test,ypred))
print(classification_report(y_test,ypred))


# *6. Decision Tree*

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dmodel=DecisionTreeClassifier()
dmodel.fit(X_train,y_train)
ypred=dmodel.predict(X_test)
print(dmodel.score(X_train,y_train))
print(confusion_matrix(y_test,ypred))
print(classification_report(y_test,ypred))


# *7. Random Forest*

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rmodel=RandomForestClassifier(n_estimators=50)
rmodel.fit(X_train,y_train)
ypred=rmodel.predict(X_test)
print(rmodel.score(X_train,y_train))
print(confusion_matrix(y_test,ypred))
print(classification_report(y_test,ypred))


# In[ ]:



amodel=AdaBoostClassifier(n_estimators=100)
amodel.fit(X_train,y_train)
ypred=amodel.predict(X_test)
print(amodel.score(X_train,y_train))
print(confusion_matrix(y_test,ypred))
print(classification_report(y_test,ypred))


# In[ ]:


gmodel=GradientBoostingClassifier(n_estimators=100)
gmodel.fit(X_train,y_train)
ypred=gmodel.predict(X_test)
print(gmodel.score(X_train,y_train))
print(confusion_matrix(y_test,ypred))
print(classification_report(y_test,ypred))


# **Voting Classifier**

# In[ ]:


#Here we are combining multiple estimators to get the voted prediction.
voting=VotingClassifier(estimators=[('logi',logmodel),('svc',modelsvc),('dtc',dmodel),('abc',amodel)],voting='soft',n_jobs=4)


# In[ ]:


voting=voting.fit(X_train,y_train)


# In[ ]:


pred1=voting.predict(test_df)
print(confusion_matrix(gen_df.Survived,pred1))
print(classification_report(gen_df.Survived,pred1))


# *Predicting value for test.csv file*

# In[ ]:


prediction=modelsvc.predict(test_df)


# In[ ]:


print(confusion_matrix(gen_df.Survived,prediction))
print(classification_report(gen_df.Survived,prediction))


# In[ ]:


#Generating Submission file.
sub=pd.DataFrame({'PassengerId':PassengerId,'Survived':prediction})


# In[ ]:


sub.to_csv('Submission.csv',index=False)

