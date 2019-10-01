#!/usr/bin/env python
# coding: utf-8

# #**Data preparation: **
# 1. Removed columns "Name", "Age", "SibSp", "Ticket", "Cabin",  "Parch", and "Embarked".
# 2. Convert objects to numbers with pandas.get_dummies.
# 3. Filled nulls with a value of 0.0.
# 4. Transformed data with MinMaxScaler() method.
# 5. Randomly splited training set into train and validation subsets.
# 
# #**Training Gradient Boosting classifier:**
# 1. Computed the accuracy scores on train and validation sets when training with different learning rates. When learning rate was 0.5, the accuracy scores on training and validation subsets were 0.829 and 0.830, respectively.
# 2. Trained Gradient Boosting classifier on training subset with parameters of criterion="mse", n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 0.  The average precision, recall, and  f1-scores on validation subsets were 0.83, 0.83, and 0.82, respectively. The area under ROC (AUC) was 0.88.

# In[ ]:


get_ipython().magic(u'matplotlib notebook')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


# load data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.info(), test.info()


# In[ ]:


# set "PassengerId" variable as index
train.set_index("PassengerId", inplace=True)
test.set_index("PassengerId", inplace=True)


# In[ ]:


# generate training target set (y_train)
y_train = train["Survived"]


# In[ ]:


# delete column "Survived" from train set
train.drop(labels="Survived", axis=1, inplace=True)


# In[ ]:


# shapes of train and test sets
train.shape, test.shape


# In[ ]:


# join train and test sets to form a new train_test set
train_test =  train.append(test)


# In[ ]:


# delete columns that are not used as features for training and prediction
columns_to_drop = ["Name", "Age", "SibSp", "Ticket", "Cabin", "Parch", "Embarked"]
train_test.drop(labels=columns_to_drop, axis=1, inplace=True)


# In[ ]:


# convert objects to numbers by pandas.get_dummies
train_test_dummies = pd.get_dummies(train_test, columns=["Sex"])


# In[ ]:


# check the dimension
train_test_dummies.shape


# In[ ]:


# replace nulls with 0.0
train_test_dummies.fillna(value=0.0, inplace=True)


# In[ ]:


# generate feature sets (X)
X_train = train_test_dummies.values[0:891]
X_test = train_test_dummies.values[891:]


# In[ ]:


X_train.shape, X_test.shape


# In[ ]:


# transform data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)


# In[ ]:


# split training feature and target sets into training and validation subsets
from sklearn.model_selection import train_test_split

X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = train_test_split(X_train_scale, y_train, random_state=0)


# In[ ]:


# import machine learning algorithms
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


# In[ ]:


# train with Gradient Boosting algorithm
# compute the accuracy scores on train and validation sets when training with different learning rates

learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(X_train_sub, y_train_sub)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train_sub, y_train_sub)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_validation_sub, y_validation_sub)))
    print()


# In[ ]:


# Output confusion matrix and classification report of Gradient Boosting algorithm on validation set

gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 0)
gb.fit(X_train_sub, y_train_sub)
predictions = gb.predict(X_validation_sub)

print("Confusion Matrix:")
print(confusion_matrix(y_validation_sub, predictions))
print()
print("Classification Report")
print(classification_report(y_validation_sub, predictions))


# In[ ]:


# ROC curve and Area-Under-Curve (AUC)

y_scores_gb = gb.decision_function(X_validation_sub)
fpr_gb, tpr_gb, _ = roc_curve(y_validation_sub, y_scores_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))


# In[ ]:




