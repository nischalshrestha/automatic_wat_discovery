#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def refineDataSets(data, isTest):
    
    import numpy as np
    import pandas as pd
    import math 
    
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare'], axis=1)

    for i in range(0, data.shape[0]):
    
        if(data['Sex'][i] == 'female'):
            data['Sex'][i] = 1
        
        else:
            data['Sex'][i] = 0
    
        if(data['Embarked'][i] == 'S'):
            data['Embarked'][i] = 0
    
        elif(data['Embarked'][i] == 'C'):
            data['Embarked'][i] = 1
    
        else:
            data['Embarked'][i] = 2
            
        if(math.isnan(data['Age'][i])):
            data['Age'][i] = data['Age'].mean()
            
        if(math.isnan(data['Pclass'][i])):
            data['Pclass'][i] = data['Pclass'].mean()
            
        if(math.isnan(data['SibSp'][i])):
            data['SibSp'][i] = data['SibSp'].mean()
    
    
    data['Embarked'][0] = 0
    
    x_data = data.values
    x_data = np.float32(np.transpose(x_data))

    if isTest:
        return x_data
    
    p_data = np.array([data['Survived'].values])
    p_data = np.float32(np.transpose(p_data))
    return x_data, p_data


# In[ ]:


import numpy as np


# In[ ]:


import pandas as pd

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


import math
from math import isnan

ct = 0

for i in range(0, len(train_data)):
    for j in range(0, 11):
        if j != 3 and j != 4 and j != 8 and j != 10 and j != 11:
            if isnan(train_data.values[i][j]):
                ct = ct + 1
            
print(ct)


# In[ ]:


x_train_data, p_train_data = refineDataSets(train_data, False)
x_test_data = refineDataSets(test_data, True)


# In[ ]:


x_train_data = np.transpose(x_train_data)[:, [0, 1, 3, 4, 5, 6]]


# In[ ]:


x_train_data


# In[ ]:


x_test_data = np.transpose(x_test_data)


# In[ ]:


x_test_data


# In[ ]:


import sklearn
from sklearn.svm import SVC

classifier = SVC(kernel='rbf', gamma='auto', degree=3, C=0.2) # linear, 1
classifier.fit(x_train_data, np.reshape(p_train_data, (len(x_train_data))))


# In[ ]:


predictions = list(np.reshape(list(classifier.predict(x_test_data)), (len(test_data))))

for i in range(0, len(predictions)):
    predictions[i] = int(predictions[i])


# In[ ]:


ids = list(test_data['PassengerId'].values[0:len(test_data)])

for i in range(0, len(ids)):
    ids[i] = int(ids[i])


# In[ ]:


predictions


# In[ ]:


submission = pd.DataFrame(np.transpose([ids, predictions]))


# In[ ]:


submission


# In[ ]:


submission.columns = ['PassengerId', 'Survived']


# In[ ]:


submission


# In[ ]:


submission.to_csv('My_Last_Hope.csv', index=False)

