#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Load Modules
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


#Read the train.csv
tit_df=pd.read_csv('../input/titanic/train.csv')


# In[ ]:


#check
tit_df.head(5)


# In[ ]:


#get a feel of the data
tit_df.describe()


# In[ ]:


tit_df.info()


# In[ ]:


#make a copy and do sanitising feature engineering on the copy
tit_df_san=tit_df.copy()
tit_df_san.info()


# In[ ]:


#1.Lot of values Missing For Cabin column . Hence , dropping it
tit_df_san.drop('Cabin',axis=1,inplace=True) 


# In[ ]:


tit_df_san.head(2)


# In[ ]:


#find correlation between 'Survived' and Other Columns  to see which columns can be useful
tit_df_san.corr().loc['Survived']


# In[ ]:


tit_df['Embarked'].unique()


# In[ ]:


#convert categorical to numerical 
tit_df_san['Embarked'].replace(['S','C','Q'],[1,2,3],inplace=True)


# In[ ]:


tit_df_san['Sex'].replace(['male','female'],[1,0],inplace=True)


# In[ ]:


tit_df_san.head(3)


# In[ ]:


#Dropping Name and Ticket columns 
tit_df_san.drop(['Name','Ticket'],axis=1,inplace=True)  


# In[ ]:


tit_df_san.head()


# In[ ]:


# check details about rows whose Embarked value is Nan
tit_df_san[tit_df_san['Embarked'].isnull()]


# In[ ]:


# try to find out the mean of all passengers whose Fare value is between 70 and 90 ,Pclass =1
tit_df_san[(tit_df_san['Fare']>70.0) & (tit_df_san['Fare']<90.0) & (tit_df_san['Pclass']==1) ].groupby('Embarked').mean()


# In[ ]:


#from above most probably the passengers embarked from 'C' Numerical value 2.0
tit_df_san['Embarked'].fillna(value=2.0,axis=0,inplace=True)
tit_df_san.loc[[61,829]]


# In[ ]:


tit_df_san.info()


# In[ ]:


# create a new column 'num of relatives' = 'Sibsp'+ 'Parch'
tit_df_san['num of relatives']=tit_df_san['SibSp']+tit_df_san['Parch']


# In[ ]:


# delete 'Sibsp' and 'parch' Column as 'num of relatives' column  should be enough
tit_df_san.drop(['SibSp','Parch'],axis=1,inplace=True)  


# In[ ]:


#graphical rep of corr()
figure=plt.figure(figsize=(10,5))

sb.heatmap(tit_df_san.corr(),annot=True)


# In[ ]:


tit_df_san.head(2)


# In[ ]:


# create a multi index data structure which will have the avg age of travellers based on Sex and Pclass
age_mapper=tit_df_san[tit_df_san['Age']>0.0].groupby(['Sex','Pclass']).mean()['Age']
age_mapper


# In[ ]:


age_mapper.loc[0].loc[3]


# In[ ]:


# Update age with Nan Values with average age based on age_mapper data structure
for i in tit_df_san[tit_df_san['Age'].isnull()].index:
    tit_df_san.loc[i, 'Age'] = age_mapper.loc[tit_df_san.loc[i,'Sex']].loc[tit_df_san.loc[i,'Pclass']]


    


# In[ ]:


tit_df_san.head(2)


# In[ ]:


tit_df_san.info()


# In[ ]:


##Model Training starts
from sklearn.model_selection import train_test_split


# In[ ]:


tit_df_san.columns


# In[ ]:


# scaling the features matrix
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X=tit_df_san[['Pclass', 'Sex', 'Age','Fare', 'Embarked', 'num of relatives']]
y=tit_df_san['Survived']           
std_scaler.fit(X)
scaled_X=std_scaler.transform(X)


# In[ ]:


final_X=pd.DataFrame(scaled_X,columns=['Pclass', 'Sex', 'Age','Fare', 'Embarked', 'num of relatives'])


# In[ ]:


final_X.head(2)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(final_X,y,test_size=0.30)


# In[ ]:


# try Support Vector Machine
from sklearn.svm import SVC


# In[ ]:


model = SVC()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


predictions = model.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


#using Logistic Regression
from sklearn.linear_model import LogisticRegression
lgrm=LogisticRegression()
lgrm.fit(X_train,y_train)


# In[ ]:


preds=lgrm.predict(X_test)


# In[ ]:


print(classification_report(y_test,preds))


# In[ ]:


#will choose SVM and train the model on whole test.csv as accuracy is 86%
model = SVC()
model.fit(final_X,y)


# In[ ]:


# Sanitising test set now
tit_df_test=pd.read_csv('../input/titanic/test.csv')


# In[ ]:


tit_df_test.head(2)


# In[ ]:


tit_df_test.describe()


# In[ ]:


tit_df_test.info()


# In[ ]:


tit_df_test.drop(['Cabin','Name','Ticket'],axis=1,inplace=True) #dropping


# In[ ]:


tit_df_test.head(2)


# In[ ]:


tit_df_test.info()


# In[ ]:


#convert categorical to numerical 
tit_df_test['Embarked'].replace(['S','C','Q'],[1,2,3],inplace=True)
tit_df_test['Sex'].replace(['male','female'],[1,0],inplace=True)


# In[ ]:


tit_df_test.head(2)


# In[ ]:


#check the row whose Fare details is missing
tit_df_test[tit_df_test['Fare'].isnull()]


# In[ ]:


# find out avg fare of passengers  Pclass=3 and Embarked=1 
tit_df_test[(tit_df_test['Pclass']==3) & (tit_df_test['Embarked']==1) ].mean()


# In[ ]:


#fill the missing value with the mean
tit_df_test['Fare'].fillna(value=13.91,axis=0,inplace=True)


# In[ ]:


tit_df_test.loc[152]


# In[ ]:


# Update age with Nan Values with average age based on age_mapper data structure
for i in tit_df_test[tit_df_test['Age'].isnull()].index:
    tit_df_test.loc[i, 'Age'] = age_mapper.loc[tit_df_test.loc[i,'Sex']].loc[tit_df_test.loc[i,'Pclass']]


# In[ ]:


# create a new column 'num of relatives' = 'Sibsp'+ 'Parch'
tit_df_test['num of relatives']=tit_df_test['SibSp']+tit_df_test['Parch']
# delete 'Sibsp' and 'parch' Column as 'num of relatives' column  should be enough
tit_df_test.drop(['SibSp','Parch'],axis=1,inplace=True) 


# In[ ]:


tit_df_test.info()


# In[ ]:


# scaling the features matrix

std_scaler = StandardScaler()
X=tit_df_test[['Pclass', 'Sex', 'Age','Fare', 'Embarked', 'num of relatives']]
std_scaler.fit(X)
scaled_X=std_scaler.transform(X)


# In[ ]:


final_test_X=pd.DataFrame(scaled_X,columns=['Pclass', 'Sex', 'Age','Fare', 'Embarked', 'num of relatives'])


# In[ ]:


final_test_X.head(2)


# In[ ]:


predictions = model.predict(final_test_X)


# In[ ]:


len(predictions)


# In[ ]:


preds_test_whole_df=pd.DataFrame(data=predictions,columns=['Survived'])


# In[ ]:


result_df=pd.DataFrame()
result_df['PassengerId']=tit_df_test['PassengerId']
result_df['Survived']=preds_test_whole_df['Survived']
result_df.head()


# In[ ]:


result_df.to_csv('predictions.csv',index=False)


# In[ ]:


result_df['Survived'].value_counts()


# In[ ]:




