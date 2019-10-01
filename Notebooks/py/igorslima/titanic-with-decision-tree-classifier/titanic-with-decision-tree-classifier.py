#!/usr/bin/env python
# coding: utf-8

# #Titanic: solving with a Decision Tree Classifier
# 
# 
# ----------
# 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score


# In[ ]:


train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")


# In[ ]:


def preprocess_data(data):
    new_dataFrame = pd.DataFrame()

    new_dataFrame['Age'] = data.Age.fillna(data.Age.mean())
    new_dataFrame['Sex'] = pd.Series([1 if s == 'male' else 0 for s in data.Sex], name = 'Sex')

    return new_dataFrame


# In[ ]:


train_data = preprocess_data(train)
train_labels = train.Survived


# In[ ]:


classifier = tree.DecisionTreeClassifier()


# In[ ]:


classifier.fit(train_data, train_labels)


# In[ ]:


test_data = preprocess_data(test)


# In[ ]:


predicao = classifier.predict(test_data)


# In[ ]:


submission = pd.DataFrame()
submission['PassengerId'] = test.PassengerId
submission['Survived'] = pd.Series(predicao)
submission.to_csv("kaggle.csv", index=False)


# In[ ]:


print('Score: {}'.format(classifier.score(train_data, train_labels)))

