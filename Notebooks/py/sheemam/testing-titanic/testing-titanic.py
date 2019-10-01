#!/usr/bin/env python
# coding: utf-8

# Testing Titanic Data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
train_df.head()


# In[ ]:


train_df['SX'],S=train_df['Sex'].factorize()
test_df['SX']=test_df['Sex'].map(pd.Series(data=range(S.size),index=S))
train_df['SX'].value_counts().plot.bar()


# In[ ]:


import re
train_df['Title']=train_df['Name'].map(lambda x: re.search('.*, ([^\.]*).*',x).group(1))
train_df['Title_num'],T=train_df['Title'].factorize()
T


# In[ ]:


train_df['Title'].value_counts().plot.bar()


# In[ ]:


TITLES=pd.Series(data=range(T.size),index=T)
TITLES


# In[ ]:


test_df['Title']=test_df['Name'].map(lambda x: re.search('.*, ([^\.]*).*',x).group(1))
test_df['Title_num']=test_df['Title'].map(TITLES)
test_df['Title_num'].isnull().any()


# In[ ]:


train_df.groupby('SX')['Title_num'].value_counts().plot.bar()


# In[ ]:


grouped=train_df.groupby('SX')['Title_num'].value_counts().reset_index(name='count')
#Freq_Title=grouped.loc(grouped.groupby('SX'))


# In[ ]:


Freq_title=grouped.loc[grouped.groupby('SX')['count'].idxmax()][['SX','Title_num']]
Freq_title


# In[ ]:


Freq_title.set_index('SX',inplace=True)
Freq_title


# In[ ]:


Freq_title_Series=Freq_title.T.squeeze()
#test_df['SX'].map(Freq_title)
#test_df['Title_num'].fillna(test_df['SX'].map(Freq_title))


# In[ ]:


test_df['SX'].map(Freq_title_Series)


# In[ ]:


test_df['Title_num'].fillna(test_df['SX'].map(Freq_title_Series),inplace=True)


# In[ ]:


test_df['Title_num'].isnull().any()


# In[ ]:


familysize=lambda x: 1+x['Parch']+x['SibSp']
train_df['FamilySize']=familysize(train_df)
test_df['FamilySize']=familysize(test_df)


# In[ ]:


train_df['FamilySize'].describe()


# In[ ]:


train_df['Age'].isnull().any()


# In[ ]:


#fill up missing age values
Median_age=train_df.groupby('Title_num')['Age'].median()
Median_age


# In[ ]:


Median=train_df.groupby('Title_num')['Age'].transform('median')
Median


# In[ ]:


train_df['Title_num'].map(Median_age)


# In[ ]:


train_df['Age'].fillna(train_df['Title_num'].map(Median_age),inplace=True)
test_df['Age'].fillna(test_df['Title_num'].map(Median_age),inplace=True)


# In[ ]:


train_df['Age'].isnull().any()


# In[ ]:


test_df['Age'].isnull().any()


# In[ ]:


features=['Age','SX','Pclass','Title_num','Fare','FamilySize']


# In[ ]:


train_df[features].isnull().any()


# In[ ]:


test_df[features].isnull().any()


# In[ ]:


Median_Fare=train_df.groupby('Pclass')['Fare'].median()


# In[ ]:


Median_Fare


# In[ ]:


test_df['Fare'].fillna(test_df['Pclass'].map(Median_Fare),inplace=True)


# In[ ]:


test_df[features].isnull().any()


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_train=scaler.fit_transform(train_df[features])
y_train=train_df['Survived']


# In[ ]:


scaler.scale_


# In[ ]:


scaler.mean_


# In[ ]:


X_train[0:5]


# In[ ]:


#y_train=train_df['Survived']
X_test=scaler.transform(test_df[features])


# In[ ]:


X_test[0:5]


# In[ ]:


type(X_train)


# In[ ]:


from sklearn import svm,model_selection

classifier=svm.SVC(kernel='rbf')
C=np.exp2(np.arange(1,15,2))
g=np.exp2(np.arange(0,-15,-2))
grid_search=model_selection.GridSearchCV(classifier,{'C':C,'gamma':g},cv=5,refit=False)
grid_search.fit(X_train,y_train)


# In[ ]:


grid_search.best_params_


# In[ ]:


Cm=np.log2(grid_search.best_params_['C'])
gm=np.log2(grid_search.best_params_['gamma'])
C=np.exp2(np.arange(Cm-1,Cm+1,0.1))
g=np.exp2(np.arange(gm+1,gm-1,-0.1))
final_model=model_selection.GridSearchCV(classifier,{'C':C,'gamma':g},cv=5)
final_model.fit(X_train,y_train)


# In[ ]:


final_model.best_params_


# In[ ]:


final_model.best_score_


# In[ ]:


final_model.predict(X_train[0:10])


# In[ ]:


y_train.head(10)


# In[ ]:


from sklearn.externals import joblib
joblib.dump(final_model,'Titanic_SVM.model')


# In[ ]:


#final_model=joblib.load('Titanic_SVM.model')
prediction=final_model.predict(X_test)
submission=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':prediction})
submission.to_csv('submission.csv',index=False)


# In[ ]:


print(check_output(["ls", "../working"]).decode("utf8"))

