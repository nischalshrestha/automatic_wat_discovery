#!/usr/bin/env python
# coding: utf-8

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


train = pd.read_csv("../input/train.csv",index_col = 0)
test = pd.read_csv("../input/test.csv",index_col = 0)


# In[ ]:


train.head()


# In[ ]:


train = train.drop(['Name','Ticket','Cabin'],axis=1)
test = test.drop(['Name','Ticket','Cabin'],axis=1)


# In[ ]:


train.head()


# In[ ]:


sex_map = {}
sex_map['male'] = 0
sex_map['female'] = 1
train.Sex = train.Sex.map(sex_map)



# In[ ]:


train.head()


# In[ ]:


test.Sex = test.Sex.map(sex_map)



# In[ ]:


train.Embarked.unique()


# In[ ]:


em_map = {}
em_map['S'] = 0
em_map['C'] = 1
em_map['Q'] = 2
train.Embarked = train.Embarked.map(em_map)
test.Embarked = test.Embarked.map(em_map)


# In[ ]:


train.head()


# In[ ]:


for c in train.columns:
    print (c,train[c].unique())


# In[ ]:


train['Sex']


# In[ ]:


Y = train.Survived.tolist()
train = train.drop(['Survived'],axis=1)


# In[ ]:


for c in train.columns:
    train[c] = train[c].fillna(train[c].median())
    test[c] = test[c].fillna(train[c].median())    


# In[ ]:


train.head()


# In[ ]:


from sklearn.manifold import TSNE


# In[ ]:


data = pd.concat([train,test])
data.head()


# In[ ]:


TSNE_features = TSNE(n_components = 3).fit_transform(data)


# In[ ]:


new_features = pd.DataFrame(TSNE_features)


# In[ ]:


new_features.head()


# In[ ]:


train_rows = train.shape[0]
test_rows = test.shape[0]


# In[ ]:


train['TSNE1'] = new_features.iloc[0:(train_rows-1),0]
train['TSNE2'] = new_features.iloc[0:(train_rows-1),1]
train['TSNE3'] = new_features.iloc[0:(train_rows-1),2]


# In[ ]:


print (train.shape)
print (test.shape)


# In[ ]:


new_features.shape


# In[ ]:


test['TSNE1'] = new_features.iloc[train_rows:(train_rows+test_rows-1),0]
test['TSNE2'] = new_features.iloc[train_rows:(train_rows+test_rows-1),1]
test['TSNE3'] = new_features.iloc[train_rows:(train_rows+test_rows-1),2]


# In[ ]:


train.head()


# In[ ]:


from mlxtend.classifier import StackingCVClassifier as SCVC


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import RidgeClassifier as RC


# In[ ]:


clf1 = kNN()
clf2 = SVC(probability=True)
clf3 = RFC()
meta_clf = RC()


# In[ ]:


stacker = SCVC(classifiers = [clf1,clf2,clf3,clf1],meta_classifier = meta_clf,use_probas=True, use_features_in_secondary=True)


# In[ ]:


for c in train.columns:
    train[c] = train[c].fillna(train[c].median())
    test[c] = test[c].fillna(train[c].median())    
stacker.fit(train.values,np.array(Y))


# In[ ]:


my_prediction = stacker.predict(test.values)


# In[ ]:


# PassengerId,Survived
submission = pd.DataFrame()
submission['PassengerId'] = test.index.tolist()
submission['Survived'] = my_prediction


# In[ ]:


submission.to_csv("submission.csv",index=False)


# In[ ]:




