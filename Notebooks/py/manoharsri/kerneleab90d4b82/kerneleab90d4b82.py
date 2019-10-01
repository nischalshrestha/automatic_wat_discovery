#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[17]:


# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Figures inline and set visualization style
get_ipython().magic(u'matplotlib inline')
sns.set()

# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[18]:


# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])


# In[19]:


data.info()


# In[20]:


# Impute missing numerical variables
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

# Check out info of data
data.info()


# In[21]:


data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data.head()


# In[22]:


# Select columns and view head
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
data.head()


# In[23]:


data.info()


# In[24]:


data_train = data.iloc[:891]
data_test = data.iloc[891:]


# In[25]:


X = data_train.values

test = data_test.values

y = survived_train.values


# In[26]:


# Instantiate model and fit to data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)


# In[27]:


# Make predictions and store in 'Survived' column of df_test
Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred


# In[28]:


#df_test[['PassengerId', 'Survived']].to_csv('Downloads/titanic/1st_dec_tree.csv', index=False)


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y)


# In[30]:


# Setup arrays to store train and test accuracies
dep = np.arange(1, 9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

# Loop over different values of k
for i, k in enumerate(dep):
    # Setup a Decision Tree Classifier
    clf = tree.DecisionTreeClassifier(max_depth=k)

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = clf.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = clf.score(X_test, y_test)

# Generate plot
plt.title('clf: Varying depth of tree')
plt.plot(dep, test_accuracy, label = 'Testing Accuracy')
plt.plot(dep, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:




