#!/usr/bin/env python
# coding: utf-8

# First the data is loaded into Pandas data frames

# In[ ]:


import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Read the input datasets
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# Fill missing numeric values with median for that column
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

print(train_data.info())
print(test_data.info())


# Next select a subset of our train_data to use for training the model

# In[ ]:


# Encode sex as int 0=female, 1=male
train_data['Sex'] = train_data['Sex'].apply(lambda x: int(x == 'male'))

# Extract the features we want to use
X = train_data[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']].as_matrix()
print(np.shape(X))

# Extract survival target
y = train_data[['Survived']].values.ravel()
print(np.shape(y))


# Now train the SVM classifier and get validation accuracy using K-Folds cross validation

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

# Build the classifier
kf = KFold(n_splits=3)
model = SVC(kernel='rbf', C=300)

scores = []
for train, test in kf.split(X):
    # Normalize training and test data using train data norm parameters
    normalizer = MinMaxScaler().fit(X[train])
    X_train = normalizer.transform(X[train])
    X_test = normalizer.transform(X[test])
    
    scores.append(model.fit(X_train, y[train]).score(X_test, y[test]))
    
print("Mean 3-fold cross validation accuracy: %s" % np.mean(scores))


# Make predictions on the test data and output the results

# In[ ]:


# Create model with all training data
normalizer = MinMaxScaler().fit(X)
X = normalizer.transform(X)
classifier = model.fit(X, y)

# Encode sex as int 0=female, 1=male
test_data['Sex'] = test_data['Sex'].apply(lambda x: int(x == 'male'))

# Extract desired features
X_ = test_data[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']].as_matrix()
X_ = normalizer.transform(X_)

# Predict if passengers survived using model
y_ = classifier.predict(X_)

# Append the survived attribute to the test data
test_data['Survived'] = y_
predictions = test_data[['PassengerId', 'Survived']]
print(predictions)

# Save the output for submission
predictions.to_csv('submission.csv', index=False)

