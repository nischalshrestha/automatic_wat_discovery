#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# In[ ]:


train_data.head()


# In[ ]:


data = train_data
tdata = test_data


# In[ ]:


#dropping unimportant feautres
data = data.drop(labels=['Name' ,'Ticket','Cabin'],axis=1)
tdata = tdata.drop(labels=['Name' ,'Ticket','Cabin'],axis=1)


# In[ ]:


data.shape


# In[ ]:


#checking null Values
data.isnull().sum()


# In[ ]:


#filling null values
data = data.fillna(method='bfill')
tdata = tdata.fillna(method='bfill')


# In[ ]:


tdata['Age'] = tdata.Age.fillna(method='ffill')
tdata.isnull().sum()


# In[ ]:


#handelling catagorical data
data['Sex'] = data.Sex.replace({'male':1,'female':0})
tdata['Sex'] = tdata.Sex.replace({'male':1,'female':0})


# In[ ]:


#creating dummy variables
Embarked_dummy=pd.get_dummies(data.Embarked,prefix='Embarked')
Embarked_dummy_test=pd.get_dummies(tdata.Embarked,prefix='Embarked')


# In[ ]:


data = pd.concat([data , Embarked_dummy] , axis=1, sort = False)
tdata = pd.concat([tdata , Embarked_dummy_test] , axis=1, sort = False)


# In[ ]:


data = data.drop('Embarked' , axis=1)
tdata = tdata.drop('Embarked' , axis=1)


# In[ ]:


tdata.head()


# In[ ]:


#normalizing train data
from sklearn.preprocessing import MinMaxScaler
min_max_scalar = MinMaxScaler()
scaled_data = min_max_scalar.fit_transform(data)
#normalizing test data
from sklearn.preprocessing import MinMaxScaler
min_max_scalar_1 = MinMaxScaler()
scaled_data_test = min_max_scalar.fit_transform(tdata) 


# In[ ]:


normalized_data = pd.DataFrame(scaled_data)
normalized_data_test = pd.DataFrame(scaled_data_test)


# In[ ]:


normalized_data.columns = data.columns
normalized_data_test.columns = tdata.columns

normalized_data.head()


# In[ ]:


#selecting dependent and independent variables
X = normalized_data.drop(['Survived'] , axis=1)
y = normalized_data['Survived']


# In[ ]:


#splitting data train , test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[ ]:


#classification using logistic regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train , y_train)
y_pred_log_reg = log_reg.predict(X_test)


# In[ ]:


#checking accuracy using test data
from sklearn.metrics import accuracy_score , confusion_matrix
acc_logistic_regression  =accuracy_score(y_test , y_pred_log_reg)
print(acc_logistic_regression)
print("Confusion matrix\n",confusion_matrix(y_test , y_pred_log_reg))


# In[ ]:


#classification using random forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train , y_train)
random_forest_predicted = random_forest.predict(X_test)


# In[ ]:


#accuracy
acc_random_forest = accuracy_score(y_true=y_test , y_pred=random_forest_predicted)
print(acc_random_forest)
print("Confusion matrix\n",confusion_matrix(y_test , random_forest_predicted))


# In[ ]:


#feature importance 
importance = random_forest.feature_importances_
plt.barh(range(len(importance)), importance)
plt.yticks(range(len(X.columns)), X.columns,fontsize=15,color='blue')
plt.xlabel("relative importance" , fontsize=15)
plt.ylabel("feature importnace" , fontsize=15)
plt.show()


# In[ ]:


#support vector machine classifier
from sklearn.svm import SVC
svc = SVC(C=1.0,kernel='rbf')
svc.fit(X_train , y_train)
svc_predicted = svc.predict(X_test)


# In[ ]:


#accuracy
acc_svm = accuracy_score(y_test , svc_predicted)
print(acc_svm)
print("Confusion matrix\n",confusion_matrix(y_test , svc_predicted))


# In[ ]:


#model seleection
models = pd.DataFrame({'model':['logistic regression','random forest','SVM'],
                      'accuracy':[acc_logistic_regression,acc_random_forest,acc_svm]})


# In[ ]:


#we choose model with highest accuracy
models


# In[ ]:




