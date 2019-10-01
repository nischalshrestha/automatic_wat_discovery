#!/usr/bin/env python
# coding: utf-8

# In[146]:


import pandas as pd
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('../input/train.csv')
data = shuffle(data)


# In[171]:


def probs2bin(probs):
    preds = []
    for index, probability in enumerate(probs):
        if probability < 0.5:
            preds.append(0)
        else:
            preds.append(1)
    return preds


# In[172]:


from sklearn.impute import SimpleImputer
predictors = ["Sex", "Pclass", "Age", "Fare"]
#X = pd.get_dummies(data[predictors], columns = predictors)
dataCopy = pd.get_dummies(data[predictors])

cols_with_missing = [col for col in dataCopy.columns 
                                 if dataCopy[col].isnull().any()]
for col in cols_with_missing:
    dataCopy[col + '_was_missing'] = dataCopy[col].isnull()
    
print(dataCopy.head())
    
X = dataCopy

imputer = SimpleImputer()
X = pd.DataFrame(imputer.fit_transform(X))
X.columns = dataCopy.columns
print(X.head())

for age in X.Age:
    if age <= 16:
        X["isChild"] = 1
    else:
        X["isChild"] = 0
        
import numpy as np

dataCopy = np.array(X)
        
for i in range(X.shape[0]):
    dataCopy[:,1][i] = dataCopy[:,1][i] * dataCopy[:,0][i]
    
dataCopy = pd.DataFrame(dataCopy)
dataCopy.columns = X.columns
X = dataCopy

print(X.info())

y = data["Survived"]


# In[173]:


from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2, random_state = 0)    


# In[205]:


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


a = pd.DataFrame(train_X[['Age', 'Age_was_missing']])
modelA = XGBRegressor(n_estimators=2, learning_rate = 1, objective = "binary:logistic",
                      booster = "gbtree", tree_method = "exact", max_depth = 1)

modelA.fit(a, train_y, early_stopping_rounds=2,
             eval_set=[(val_X[["Age", "Age_was_missing"]], val_y)], verbose=False)
predsA = modelA.predict(pd.DataFrame(val_X[["Age", "Age_was_missing"]]))


acc = mean_squared_error(val_y, predsA)
print(acc)


# In[213]:


b = pd.DataFrame(train_X[['Sex_female', 'isChild']])
modelB = XGBRegressor(n_estimators=3, learning_rate = 1, objective = "binary:logistic",
                      booster = "gbtree", tree_method = "exact", max_depth = 2)

modelB.fit(b, train_y, early_stopping_rounds=1,
             eval_set=[(val_X[['Sex_female','isChild']], val_y)], verbose=False)
predsB = modelB.predict(pd.DataFrame(val_X[['Sex_female', 'isChild']]))


acc = mean_squared_error(val_y, predsB)
print(acc)


# In[214]:


c = pd.DataFrame(train_X[['Pclass', 'Fare']])
modelC = XGBRegressor(n_estimators=5, learning_rate = .4, objective = "binary:logistic",
                      booster = "gbtree", tree_method = "exact", max_depth = 4)

modelC.fit(c, train_y, early_stopping_rounds=3,
             eval_set=[(val_X[['Pclass', 'Fare']], val_y)], verbose=False)
predsC = modelC.predict(pd.DataFrame(val_X[['Pclass','Fare']]))


acc = mean_squared_error(val_y, predsC)
print(acc)


# In[215]:


train_preds = pd.concat([pd.DataFrame(modelA.predict(a)),
                         pd.DataFrame(modelB.predict(b)),
                         pd.DataFrame(modelC.predict(c))], axis=1)
val_preds = pd.concat([pd.DataFrame(predsA), pd.DataFrame(predsB), pd.DataFrame(predsC)], axis=1)
sc = StandardScaler()
train_preds = sc.fit_transform(train_preds)
val_preds = sc.transform(val_preds)


# In[222]:


model = Sequential()
model.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid', input_dim = 3))
model.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(train_preds, train_y, batch_size = 5, epochs = 20)


# In[223]:


ann_preds = probs2bin(model.predict(val_preds))
acc = accuracy_score(val_y, ann_preds)
print(acc)


# In[224]:


test = pd.read_csv('../input/test.csv')
testCopy = pd.get_dummies(test[predictors])

for col in cols_with_missing:
    testCopy[col + '_was_missing'] = testCopy[col].isnull()
    
imputedTest = imputer.transform(testCopy)

#X = sc.fit_transform(imputedData)
imputedTest = pd.DataFrame(imputedTest)
imputedTest.columns = testCopy.columns

for age in imputedTest.Age:
    if age <= 16:
        imputedTest["isChild"] = 1
    else:
        imputedTest["isChild"] = 0
        
testCopy = imputedTest
imputedTest = np.array(imputedTest)
        
for i in range(imputedTest.shape[0]):
    imputedTest[:,1][i] = imputedTest[:,1][i] * imputedTest[:,0][i]
    
imputedTest = pd.DataFrame(imputedTest)
imputedTest.columns = testCopy.columns

print(imputedTest.head())


# In[225]:


testA = pd.DataFrame(imputedTest[['Age', 'Age_was_missing']])
testB = pd.DataFrame(imputedTest[['Sex_female', 'isChild']])
testC = pd.DataFrame(imputedTest[['Pclass', 'Fare']])
test_preds = pd.concat([pd.DataFrame(modelA.predict(testA)),
                         pd.DataFrame(modelB.predict(testB)),
                         pd.DataFrame(modelC.predict(testC))], axis=1)
test_preds = sc.transform(test_preds)
finalPreds = model.predict(test_preds)


# In[226]:


binPreds = probs2bin(finalPreds)
submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': binPreds})
pd.to_numeric(submission.PassengerId)
submission.to_csv('xgbStacked.csv', index=False)

