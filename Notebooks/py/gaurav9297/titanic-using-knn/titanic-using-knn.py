#!/usr/bin/env python
# coding: utf-8

# In[ ]:





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


dataset=pd.read_csv('../input/train.csv')
testset=pd.read_csv('../input/test.csv')
label=dataset.iloc[0:890,1]
data=dataset.iloc[0:890,[2,4,5]]
testdat=testset.iloc[0:418,[1,3,4]]
x=[data,testdat]

for change in x:
    change['Sex']=change['Sex'].map({'female':0,'male':1}).astype(int)
    

data=(data.fillna(0)) #filling NA values
testdat=testdat.fillna(0)
print(testdat)


# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split

train_data,test_data,train_labels,test_labels=train_test_split(data,label,random_state=7,train_size=0.7)


# In[ ]:





# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier()
clf.fit(data,label)


# In[ ]:


predictions=clf.predict(test_data)


# In[ ]:


from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels,predictions))


# In[ ]:


result=clf.predict(testdat)


# In[ ]:


print(result)


# In[ ]:


index=[testset['PassengerId']]
df=pd.DataFrame(data=result,index=testset['PassengerId'],columns=['Survived'])
df.to_csv('gender_submission.csv',header=True)
print('gender_submission.csv')

