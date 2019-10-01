#!/usr/bin/env python
# coding: utf-8

# # Titanic prediction using Keras
# 
# 

#  I have used in my work the following references:
# 
# 1. [Kernel](http://https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy) by ldfreeman3 
# 2. Udacity's course:  [Deep Learning A-Z™: Hands-On Artificial Neural Networks](https://www.udemy.com/deeplearning/)
# 

# In[ ]:


# Imports
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV


# In[ ]:


# Load the dataset
data_train = pd.read_csv('../input/train.csv')
data_test  = pd.read_csv('../input/test.csv')

# Drop unnecessary columns
drop_column = ['PassengerId','Cabin', 'Ticket', 'Embarked']
data_train.drop(drop_column, axis=1, inplace = True)

data_cleaner = [data_train, data_test]

# Fill missing values
for dataset in data_cleaner:    
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    


# In[ ]:


# Create data
for dataset in data_cleaner:    
    #Discrete variables
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    #quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


    #Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
    #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)


# In[ ]:


stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/

for dataset in data_cleaner:    
    title_names = (dataset['Title'].value_counts() < stat_min) #this will create a true false series with title name as index
    #apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)


#code categorical data
label = LabelEncoder()
for dataset in data_cleaner:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])


# In[ ]:


# Y traget
Target = ['Survived']
# X cols
data_x_cols = ['Sex','Pclass', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts

# Convert categorical variable into dummy variables
data_x_train = pd.get_dummies(data_train[data_x_cols])
data_y_test = pd.get_dummies(data_test[data_x_cols])

data_x_cols_list = data_x_train.columns.tolist()

# Split data    
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data_x_train[data_x_cols_list], data_train[Target], random_state = 0)


# ## Method for building the model.
#  Needed in GridSearch

# In[ ]:


def build_classifier(optimizer):
    classifier = keras.Sequential()
    classifier.add(keras.layers.Dense(units = 8, kernel_initializer = 'lecun_uniform', activation = 'relu', input_dim = 14))
    classifier.add(keras.layers.Dropout(rate = 0.1))
    classifier.add(keras.layers.Dense(units = 8, kernel_initializer = 'lecun_uniform', activation = 'relu'))
    classifier.add(keras.layers.Dense(units = 1, kernel_initializer = 'lecun_uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


# ## Tuning the hyper-parameters of an estimator 
# 
# Using GridSearchCV
# 
# *I have commented this out because it takes too long to run in this notebook.
# You can run this locally in your own computer and try to tune the parameters.*
# 
# ```
# # Tuning the ANN
# classifier = keras.wrappers.scikit_learn.KerasClassifier(build_fn = build_classifier)
# 
# parameters = {'batch_size': [25, 32],
#               'epochs': [100, 200],
#               'optimizer': ['adam', 'rmsprop']}
# grid_search = GridSearchCV(estimator = classifier,
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10)
# grid_search = grid_search.fit(test_x, test_y)
# best_parameters = grid_search.best_params_
# best_accuracy = grid_search.best_score_
# 
# print(best_parameters)
# print(best_accuracy)
# ```

# ## Fitting the model using the best parameters from the part above

# In[ ]:


classifier = build_classifier(keras.optimizers.RMSprop(lr=0.03))
classifier.fit(train1_x, train1_y, batch_size = 15, nb_epoch = 150, validation_data=(test1_x, test1_y))

predictions = classifier.predict_classes(data_y_test)
ids = data_test.PassengerId.copy()
output = ids.to_frame()
output["Survived"]=predictions
output.columns = ['PassengerId', 'Survived']


# In[ ]:


# Save output to file for the competition
output.to_csv("my_submission.csv",index=False)

