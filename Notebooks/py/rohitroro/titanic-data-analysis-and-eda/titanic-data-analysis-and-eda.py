#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from sklearn import model_selection 
import statsmodels.formula.api as sm
from sklearn import metrics
from sklearn.model_selection import cross_val_score


# In[ ]:


data_train=pd.read_csv("../input/train.csv")
data_test=pd.read_csv("../input/test.csv")
data_train.head()
data_train.columns


# In[ ]:


data_train.info()


# In[ ]:


data_train['Age']=data_train['Age'].fillna(data_train['Age'].median())
data_test["Age"]=data_test['Age'].fillna(data_test['Age'].median())
data_test['Fare']=data_test['Fare'].fillna(data_test['Fare'].median())


# In[ ]:


data_train=data_train.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
data_test=data_test.drop(['Name','Ticket','Cabin','Embarked'],axis=1)


# In[ ]:


data_train.replace(to_replace='male',value=1,inplace=True)
data_train.replace(to_replace='female',value=0,inplace=True)
data_test.replace(to_replace='male',value=1,inplace=True)
data_test.replace(to_replace='female',value=0,inplace=True)


X=data_train[data_train.columns.difference(['Survived'])]
Y=data_train['Survived']


# In[ ]:


logreg=LogisticRegression()
logit_model=sm.Logit(Y,X)
result=logit_model.fit()
print(result.summary2())


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


# In[ ]:


logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print(logreg.score(X_test,y_test))


# In[ ]:


kfold=model_selection.KFold(n_splits=10, random_state=7)
modelCV=LogisticRegression()
scoring='accuracy'
results=model_selection.cross_val_score(modelCV,X_train,y_train,cv=kfold,scoring=scoring)
print("10 fold cross validation average accuracy: %3f" % (results.mean()))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[ ]:


fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred)
roc_auc=metrics.auc(fpr,tpr)
plt.plot(fpr,tpr,label="auc="+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc=4)
plt.show()


# ###### ROC curve (area=0.77)

# # EDA

# In[ ]:


data_train_eda=pd.read_csv("../input/train.csv")
fig,axes=plt.subplots(figsize=(10,4))
sns.barplot(x="Survived", y="Age", data=data_train_eda)


# In[ ]:


sns.barplot(y="Survived", x="Sex", data=data_train_eda)


# ###### More number of men couldn't make it out of the crash. Number of females who survived is higher

# In[ ]:


sns.barplot(x="Pclass", y="Fare", data=data_train)


# In[ ]:


sns.barplot(y="Survived", x="Embarked", data=data_train_eda)


# ###### People who boarded the ship at Cherbourg have maximum survival rate

# In[ ]:


sns.barplot(y="Survived", x="Pclass", data=data_train_eda)


# ###### As we can see the lower class cabins had lesser number of people who survived
