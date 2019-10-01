#!/usr/bin/env python
# coding: utf-8

# 
# ### Dataset
# Titanic: Machine Learning from Disaster (https://www.kaggle.com/c/titanic/data)
# 
# #### Variable Notes
# 
# pclass: A proxy for socio-economic status (SES)
# * 1st = Upper
# * 2nd = Middle
# * 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...
# * Sibling = brother, sister, stepbrother, stepsister
# * Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# * Parent = mother, father
# * Child = daughter, son, stepdaughter, stepson
# * Some children travelled only with a nanny, therefore parch=0 for them.
# 
# ### Practice 3 models 
# * Linear
# * Tree
# * Neural Network
# 
# 
# ### Necessary process
# * Split train, validation, test data
# * Handle categorical data
# * Cross feature
# * Train/prediction
# * Visualize (at least 1 graph)
# 
# 
# 

# In[ ]:


#import pkg_resources

#for dist in pkg_resources.working_set:
#    print(dist.project_name, dist.version)


# In[ ]:


import os

# for a local xgboost, the path is required to be added.
#mingw_path = 'W:\\langs\MinGW64\\mingw64\\bin;'
#os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']


# In[ ]:


import xgboost as xgb

# linear algebra
import numpy as np 

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.cross_validation import *
from sklearn.linear_model import *
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

import matplotlib as mpl
from matplotlib import pyplot as plt

import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense,Dropout

import warnings
warnings.filterwarnings('ignore')


# ### 1. Read the data sets first

# In[ ]:


#print(os.listdir("../input"))

csv_train = pd.read_csv('../input/train.csv')
csv_final_test = pd.read_csv('../input/test.csv')
csv_gender_submit = pd.read_csv('../input/gender_submission.csv')

#The score of gender_submission.csv is 0.76555


# In[ ]:


csv_train.info()
csv_train_null_values = csv_train.isnull().sum()
csv_train_null_values


# ### In training set, there are empty data in 'Age','Cabin' and 'Embarked' fields.

# In[ ]:


csv_train_null_values.plot.bar()


# In[ ]:


csv_final_test.info()
csv_final_test_null_values = csv_final_test.isnull().sum()
csv_final_test_null_values


# ### In test set, there are empty data in 'Age','Fare' and 'Cabin' fields.

# In[ ]:


csv_final_test_null_values.plot.bar()


# In[ ]:


csv_train.describe()


# In[ ]:


csv_train['Sex'].describe()


# In[ ]:


csv_train['Age'].describe()


# In[ ]:


sns.countplot (csv_train['Sex'], hue=csv_train['Survived'])
display(csv_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().round(3))


# ### 2. Feature engineering

# #### Remap the 'Sex' data to 0 and 1

# In[ ]:


csv_train['Sex']


# In[ ]:


csv_train['Sex'] = csv_train['Sex'].map({'female':0, 'male':1}).astype(int)
csv_train['Sex']


# In[ ]:


csv_train.describe()
#The percentage of the male is 64%, the mean Survived 38%.


# In[ ]:


csv_final_test['Sex'] = csv_final_test['Sex'].map({'female':0, 'male':1}).astype(int)
csv_final_test['Sex']


# #### fill the NULL values with the mean values

# In[ ]:


age_mean = csv_train['Age'].mean()
print ("age_mean={}".format(age_mean))
csv_train['Age'] = csv_train['Age'].fillna(age_mean)
csv_train.describe()


# In[ ]:


csv_final_test['Age'] = csv_final_test['Age'].fillna(age_mean)
csv_final_test.describe()


# #### The OneHot encoding for 'Embarked' field

# In[ ]:


csv_train_OneHot = pd.get_dummies(data=csv_train,columns=['Embarked'])
csv_train_OneHot[:5]


# In[ ]:


csv_final_test_OneHot = pd.get_dummies(data=csv_final_test,columns=['Embarked'])
csv_final_test_OneHot[:5]


# #### Fill NA with 0

# In[ ]:


csv_train_OneHot = csv_train_OneHot.fillna(0)
csv_train_OneHot[:5]


# In[ ]:


csv_final_test_OneHot = csv_final_test_OneHot.fillna(0)
csv_final_test_OneHot[:5]


# #### The relationships between the data fields

# In[ ]:


plt.figure(figsize = (10, 10))
sns.heatmap(csv_train_OneHot.corr(),  vmax=0.9, square=True);


# In[ ]:


csv_train_corr_Survived = csv_train_OneHot.corr()['Survived']
csv_train_corr_Survived


# ### 3. The data sets
# 

# In[ ]:


train_data_X = csv_train_OneHot[['Sex','Pclass','Fare','Age','SibSp','Parch','Embarked_C','Embarked_Q','Embarked_S']]
train_data_X[:5] 


# In[ ]:


train_data_Y = csv_train_OneHot[['Survived']]
train_data_Y[:5] 


# In[ ]:


train_data,validation_data,train_labels,validation_labels=train_test_split(train_data_X,train_data_Y,random_state=7,train_size=0.8)
train_data.describe()


# In[ ]:


train_data[:5]


# In[ ]:


train_labels[:5]


# In[ ]:


validation_data.describe()


# In[ ]:


validation_data[:5]


# In[ ]:


validation_labels[:5]


# In[ ]:


test_data = csv_final_test_OneHot[['Sex','Pclass','Fare','Age','SibSp','Parch','Embarked_C','Embarked_Q','Embarked_S']]
test_data[:5]


# ### 4. The Linear Regression Model

# In[ ]:


LR = LinearRegression()
predictions = []

#train_predictors = (titanic[predictors].iloc[train, :])  # the features for training (x1, x2...xn)
#train_target = titanic["Survived"].iloc[train]  # the predictive target (y)

LR.fit(train_data, train_labels)  # finding the best fit for the target 
validation_predictions = LR.predict(validation_data)  # predict based on the best fit produced by alg.fit
predictions.append(validation_predictions)
predictions = np.concatenate(predictions)
predictions[predictions >= 0.5] = 1
predictions[predictions < 0.5] = 0
#print (predictions)

accuracy = np.count_nonzero(validation_labels == predictions)/validation_labels.count()
print ('accuracy: {}'.format(accuracy['Survived']))


# In[ ]:


predictions = []
#print (predictions)
test_predictions = LR.predict(test_data)  # predict based on the best fit produced by alg.fit
predictions.append(test_predictions)
#predictions = test_predictions
predictions = np.concatenate(predictions)
predictions[predictions >= 0.5] = 1
predictions[predictions < 0.5] = 0
#csv_gender_submit["PassengerId"]
#print (predictions)
#predictions_list = predictions.tolist()
#print (predictions_list)
submission = pd.DataFrame()
#submission = pd.DataFrame({"PassengerId": csv_gender_submit["PassengerId"],"Survived": predictions})
submission['PassengerId'] = csv_gender_submit["PassengerId"]
submission['Survived'] = predictions.astype(int)
#submission.info()
#submission

submission.to_csv("lr_submission.csv", index=False)

# the score of "lr_submission.csv" is 0.76076


# ### 5. The Decision Tree Model

# In[ ]:


DT = tree.DecisionTreeClassifier()
predictions = []

DT.fit(train_data, train_labels) 
validation_predictions = DT.predict(validation_data)  # predict based on the best fit produced by alg.fit
predictions.append(validation_predictions)
#predictions = np.concatenate(predictions)
#predictions[predictions > 0.5] = 1
#predictions[predictions < 0.5] = 0
#print (predictions)
#validation_labels
accuracy = np.count_nonzero(validation_labels == predictions)/validation_labels.count()
print ('accuracy: {}'.format(accuracy['Survived']))

print('DT.score: {}'.format(DT.score(train_data, train_labels)))


# In[ ]:


predictions = []
#print (predictions)
test_predictions = DT.predict(test_data)  # predict based on the best fit produced by alg.fit
#predictions.append(test_predictions)
predictions = test_predictions
#predictions = np.concatenate(predictions)
#predictions[predictions > 0.5] = 1
#predictions[predictions < 0.5] = 0

#print (predictions)
#predictions_list = predictions.tolist()
#print (predictions_list)
submission = pd.DataFrame()
#submission = pd.DataFrame({"PassengerId": csv_gender_submit["PassengerId"],"Survived": predictions})
submission['PassengerId'] = csv_gender_submit["PassengerId"]
submission['Survived'] = predictions
#submission.info()
#print (submission)

submission.to_csv("dt_submission.csv", index=False)

# the score of "dt_submission.csv" is 0.72248


# ### 6. The NN Model

# #### Convert data sets to arrays

# In[ ]:


train_data[:5]


# In[ ]:


ndarray = train_data.values
ndarray.shape
train_data_array = ndarray
ndarray[:2]


# In[ ]:


ndarray = train_labels.values
ndarray.shape
train_labels_array = ndarray
ndarray[:5]


# In[ ]:


ndarray = validation_data.values
ndarray.shape
validation_data_array = ndarray
ndarray[:2]


# In[ ]:


ndarray = validation_labels.values
ndarray.shape
validation_labels_array = ndarray
ndarray[:5]


# In[ ]:


ndarray = test_data.values
ndarray.shape
test_data_array = ndarray
ndarray[:2]


# #### Data set scaling

# In[ ]:


minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))


# In[ ]:


train_data_array_scaled = minmax_scale.fit_transform(train_data_array)
validation_data_array_scaled = minmax_scale.fit_transform(validation_data_array)
test_data_array_scaled = minmax_scale.fit_transform(test_data_array)


# In[ ]:


train_data_array_scaled[:5]


# In[ ]:


validation_data_array_scaled[:5]


# In[ ]:


test_data_array_scaled[:5]


# #### Train Model

# In[ ]:


model = Sequential()

# input layer and hidden layer 1
model.add(Dense(units=40, input_dim=9, 
                kernel_initializer='uniform', 
                activation='relu'))

# hidden layer 2
model.add(Dense(units=30, 
                kernel_initializer='uniform', 
                activation='relu'))

# output layer
model.add(Dense(units=1, 
                kernel_initializer='uniform',
                activation='sigmoid'))


# In[ ]:


model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy'])


# In[ ]:


train_history =model.fit(x=train_data_array_scaled, 
                         y=train_labels_array, 
                         validation_split=0.1, 
                         epochs=30, 
                         batch_size=30,verbose=2)


# In[ ]:


def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[ ]:


show_train_history(train_history,'acc','val_acc')


# In[ ]:


show_train_history(train_history,'loss','val_loss')


# In[ ]:


scores = model.evaluate(x=validation_data_array_scaled, 
                        y=validation_labels_array)
scores[1]


# In[ ]:


test_probability=model.predict(test_data_array_scaled)
test_probability[test_probability >= 0.5] = 1
test_probability[test_probability < 0.5] = 0
test_probability[:10]


# In[ ]:


submission = pd.DataFrame()
#submission = pd.DataFrame({"PassengerId": csv_gender_submit["PassengerId"],"Survived": predictions})
submission['PassengerId'] = csv_gender_submit["PassengerId"]
submission['Survived'] = test_probability.astype(int)
#submission.info()
print (submission)

submission.to_csv("nn_submission.csv", index=False)

# the score of "nn_submission.csv" is 0.77033


# ### 7 The scores
# 
# #### nn_submission.csv
# * 0.77033
# 
# #### dt_submission.csv
# * 0.72248
# 
# #### lr_submission.csv
# * 0.76076
# 
# #### gender_submission.csv
# * 0.7655
# 
# 

# In[ ]:





# ### References
# * https://www.kaggle.com/gaurav9297/fork-of-titanic-using-linear-regression
# * https://www.kaggle.com/igorslima/titanic-with-decision-tree-classifier
# * book: TensorFlow+Keras深度學習人工智慧實務應用 by 林大貴
# 

# In[ ]:




