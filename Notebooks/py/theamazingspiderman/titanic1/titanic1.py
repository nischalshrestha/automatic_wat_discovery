#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sb
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')

sb.set(style="whitegrid")

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


training = pd.read_csv('../input/train.csv')
testing = pd.read_csv('../input/test.csv')


# In[ ]:


training.head()


# In[ ]:


training.describe()


# In[ ]:


def null_table(training, testing):
    print("Training Data Frame")
    print(pd.isnull(training).sum()) 
    print(" ")
    print("Testing Data Frame")
    print(pd.isnull(testing).sum())

null_table(training, testing)


# In[ ]:


training.drop(labels = ['Cabin', 'Ticket'], axis = 1, inplace = True)
testing.drop(labels = ['Cabin', 'Ticket'], axis = 1, inplace = True)

null_table(training, testing)


# In[ ]:


copy = training.copy()
copy.dropna(inplace = True)
sb.distplot(copy["Age"])


# In[ ]:


#the median will be an acceptable value to place in the NaN cells
training['Age'].fillna(training['Age'].median(), inplace = True)
testing["Age"].fillna(testing["Age"].median(), inplace = True) 
training["Embarked"].fillna("S", inplace = True)
testing["Fare"].fillna(testing["Fare"].median(), inplace = True)

null_table(training, testing)


# In[ ]:


training.loc[training["Sex"] == "male", "Sex"] = 0
training.loc[training["Sex"] == "female", "Sex"] = 1

training.loc[training["Embarked"] == "S", "Embarked"] = 0
training.loc[training["Embarked"] == "C", "Embarked"] = 1
training.loc[training["Embarked"] == "Q", "Embarked"] = 2

testing.loc[testing["Sex"] == "male", "Sex"] = 0
testing.loc[testing["Sex"] == "female", "Sex"] = 1

testing.loc[testing["Embarked"] == "S", "Embarked"] = 0
testing.loc[testing["Embarked"] == "C", "Embarked"] = 1
testing.loc[testing["Embarked"] == "Q", "Embarked"] = 2


# In[ ]:


training.sample(5)


# In[ ]:


training["FamSize"] = training["SibSp"] + training["Parch"] + 1
testing["FamSize"] = testing["SibSp"] + testing["Parch"] + 1

training["IsAlone"] = training.FamSize.apply(lambda x: 1 if x == 1 else 0)
testing["IsAlone"] = testing.FamSize.apply(lambda x: 1 if x == 1 else 0)


# In[ ]:


training.sample(5)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


features = ["Pclass", "Sex", "Age", "Embarked", "Fare", "FamSize", "IsAlone"]
X_train = training[features] #define training features set
y_train = training["Survived"] #define training label set
X_test = testing[features] #define testing features set
#we don't have y_test, that is what we're trying to predict with our model


# In[ ]:


from sklearn.model_selection import train_test_split #to create validation data set

X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2,
                                                            random_state=0)
#X_valid and y_valid are the validation sets


# In[ ]:


rf_clf = RandomForestClassifier()
rf_clf.fit(X_training, y_training)
pred_rf = rf_clf.predict(X_valid)
acc_rf = accuracy_score(y_valid, pred_rf)

print(acc_rf)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam


# In[ ]:


# Initialising the NN
model = Sequential()

# layers
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
model.fit(X_training, y_training, batch_size = 32, epochs = 2000)


# In[ ]:





# **submission
# ******

# In[ ]:


submission_predictions = rf_clf.predict(X_test)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": testing["PassengerId"],
        "Survived": submission_predictions
    })

submission.to_csv("titanic.csv", index=False)
print(submission.shape)


# In[ ]:


submission


# In[ ]:


import csv


# In[ ]:


write.csv(submission,"output.csv")


# In[ ]:





# In[ ]:




