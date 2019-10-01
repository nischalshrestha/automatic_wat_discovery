#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Machine learning classification libraries
from sklearn.svm import SVC
from sklearn.metrics import scorer
from sklearn.metrics import accuracy_score
 
# For data manipulation
import pandas as pd
import numpy as np
 
# To plot
import matplotlib.pyplot as plt
import seaborn


# In[ ]:


fields = ['DATPRE', 'PREABE', 'PREMAX', 'PREMIN', 'PREULT']

dataset = pd.read_csv('../input/b3-stock-quotes/COTAHIST_A2009_to_A2018P.csv', usecols=fields, index_col='DATPRE', parse_dates=['DATPRE'])


# In[ ]:


dataset = dataset['2017-09-01':'2018']
dataset.head()


# In[ ]:


Df= dataset.dropna()
Df['PREULT'].plot(figsize=(10,5))
plt.ylabel("S&P500 Price")
plt.show()


# In[ ]:


y = np.where(Df['PREULT'].shift(-1) > Df['PREULT'],1,-1)


# In[ ]:


Df['Open-Close'] = Df['PREABE'] - Df['PREULT']
Df['High-Low'] = Df['PREMAX'] - Df['PREMIN']
 
X=Df[['Open-Close','High-Low']]


# In[ ]:


split_percentage = 0.8
split = int(split_percentage*len(Df))
 
# Train data set
X_train = X[:split]
y_train = y[:split]
 
# Test data set
X_test = X[split:]
y_test = y[split:]
cls = SVC().fit(X_train, y_train)
accuracy_train = accuracy_score(y_train, cls.predict(X_train))
 
accuracy_test = accuracy_score(y_test, cls.predict(X_test))
print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))
print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))




# In[ ]:





# In[ ]:





# In[ ]:


Df['Predicted_Signal'] = cls.predict(X)
 
# Calculate log returns
Df['Return'] = np.log(Df['PREMIN'].shift(-1) / Df['PREMIN'])*100
Df['Strategy_Return'] = Df.Return * Df.Predicted_Signal
Df.Strategy_Return.iloc[split:].cumsum().plot(figsize=(10,5))
plt.ylabel("Strategy Returns (%)")
plt.show()

