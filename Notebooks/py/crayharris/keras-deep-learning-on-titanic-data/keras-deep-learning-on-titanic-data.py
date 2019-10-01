#!/usr/bin/env python
# coding: utf-8

# ## Keras Deep Learning on Titanic data
# 
# 
# ---
# With this notebook, I am trying to do two things in parallel:
# 
# 1. Applying my basic learnings from the book [Deep Learning With Python by Jason Brownlee](https://machinelearningmastery.com/deep-learning-with-python/) that I recently studied. 
# 2. Publish my first kaggle notebook for my own practice to become a Kaggler :-) 
# 
# 
# There are several good Kaggel notebooks showing in-depth visualisations, data exploration and wrangling on the Titanic data:
# - [In-Depth Visualisations - Simple Methods](https://www.kaggle.com/jkokatjuhha/in-depth-visualisations-simple-methods)
# - [Titanic Data Exploration Starter](https://www.kaggle.com/neviadomski/titanic-data-exploration-starter)
# - [Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook)
# 
# I won't repeat any of the above work here and jump more or less directlty into the coding with Keras.
# 
# @Jason, I would highly appreciate your feedback. Especially because the scores of the Keras model arn't that great (0.76-0.78).
# 
# ---
# 
# ### Overview:
# 1. Simple Keras model with minimal cleansed data
# 2. Predict the missing data in the Age feature (using Keras ;-)
# 3. Wrangle, prepare, cleanse the titanic data manually. Source: [Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook)
# 4. Predict 'Survived' with Kears based on wrangled input data
# 5. Summary
# ---
# ** My playground:**
# - conda 4.3.27
# - Python 3.5.4
# - Keras 2.0.8 using TensorFlow backend
# 

# ### Part 1 - Simple Keras model with minimal cleansed data
# 
# 
# As usual, import modules, read data and display some data

# In[ ]:


# data processing
import numpy as np
import pandas as pd 

# machine learning
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# utils
import time
from datetime import timedelta

# some configuratin flags and variables
verbose=0 # Use in classifier

# Input files
file_train='../input/train.csv'
file_test='../input/test.csv'

# defeine random seed for reproducibility
seed = 69
np.random.seed(seed)

# read training data
train_df = pd.read_csv(file_train,index_col='PassengerId')



# In[ ]:


# Show the columns
train_df.columns.values


# In[ ]:


# Show the shape
train_df.shape


# In[ ]:


# preview the training dara
train_df.head()


# In[ ]:


# Show that there is NaN data (Age,Fare Embarked), that needs to be handled during data cleansing
train_df.isnull().sum()


# #### A function for simple data cleansing
# - Drop unwanted features ['Name', 'Ticket', 'Cabin']
# - Fill missing data: Age and Fare with the mean, Embarked with most frequent value
# - Convert categorical features into numeric
# - Convert Embarked to one-hot

# In[ ]:


def prep_data(df):
    # Drop unwanted features
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    
    # Fill missing data: Age and Fare with the mean, Embarked with most frequent value
    df[['Age']] = df[['Age']].fillna(value=df[['Age']].mean())
    df[['Fare']] = df[['Fare']].fillna(value=df[['Fare']].mean())
    df[['Embarked']] = df[['Embarked']].fillna(value=df['Embarked'].value_counts().idxmax())
    
    # Convert categorical  features into numeric
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
      
    # Convert Embarked to one-hot
    enbarked_one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = df.drop('Embarked', axis=1)
    df = df.join(enbarked_one_hot)

    return df


# Prepare training data and show that there isn't any null data

# In[ ]:


train_df = prep_data(train_df)
train_df.isnull().sum()


# Split training data into input X and output Y

# In[ ]:


# X contains all columns except 'Survived'  
X = train_df.drop(['Survived'], axis=1).values.astype(float)

# It is almost always a good idea to perform some scaling of input values when using neural network models (jb).

scale = StandardScaler()
X = scale.fit_transform(X)

# Y is just the 'Survived' column
Y = train_df['Survived'].values


# #### Simple Network using Keras
# - Input lauer with 16 neuron (units/outputs).
# - Two hidden layers.
# - Output layer with a single neuron and sigmoid activation function to output a value between 0 and 1.
# - Optimizer and intit will be searched with GridSearch.
# 

# In[ ]:


def create_model(optimizer='adam', init='uniform'):
    # create model
    if verbose: print("**Create model with optimizer: %s; init: %s" % (optimizer, init) )
    model = Sequential()
    model.add(Dense(16, input_dim=X.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# #### Run GridSearch, optionally
# GridSearch is very time consuming. 
# Set `run_gridsearch = True ` in case you really want to run it.
# Or simply tune optimizers, inits, epochs and batches. 
# 

# In[ ]:


run_gridsearch = True

if run_gridsearch:
    
    start_time = time.time()
    if verbose: print (time.strftime( "%H:%M:%S " + "GridSearch started ... " ) )
    optimizers = ['rmsprop', 'adam']
    inits = ['glorot_uniform', 'normal', 'uniform']
    epochs = [50, 100, 200, 400]
    batches = [5, 10, 20]
    
    model = KerasClassifier(build_fn=create_model, verbose=verbose)
    
    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X, Y)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    if verbose: 
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        elapsed_time = time.time() - start_time  
        print ("Time elapsed: ",timedelta(seconds=elapsed_time))
        
    best_epochs = grid_result.best_params_['epochs']
    best_batch_size = grid_result.best_params_['batch_size']
    best_init = grid_result.best_params_['init']
    best_optimizer = grid_result.best_params_['optimizer']
    
else:
    # pre-selected paramters
    best_epochs = 200
    best_batch_size = 5
    best_init = 'glorot_uniform'
    best_optimizer = 'rmsprop'


# #### Build model and predit
# - Create a classifier with best parameters
# - Fit model 
# - Predict 'Survived'

# In[ ]:


# Create a classifier with best parameters
model_pred = KerasClassifier(build_fn=create_model, optimizer=best_optimizer, init=best_init, epochs=best_epochs, batch_size=best_batch_size, verbose=verbose)
model_pred.fit(X, Y)

# Read test data
test_df = pd.read_csv(file_test,index_col='PassengerId')
# Prep and clean data
test_df = prep_data(test_df)
# Create X_test
X_test = test_df.values.astype(float)
# Scaling
X_test = scale.transform(X_test)

# Predict 'Survived'
prediction = model_pred.predict(X_test)



# Save predictions

# In[ ]:


submission = pd.DataFrame({
    'PassengerId': test_df.index,
    'Survived': prediction[:,0],
})

submission.sort_values('PassengerId', inplace=True)    
submission.to_csv('submission-simple-cleansing.csv', index=False)


# #### Result part 1 - Simple Keras model with minimal cleansed data
# Submitting the prediction csv will have scroe of ~78%. Not bad and not that good.

# ### Part 2 - Predict the missing data in the Age feature 
# 
# The Age feature has a lot missing data. Let's try to predict the age where needed.
# 
# #### Overview
# - Read the data (again) and put all data into one data frame (dfa)
# - Clean-up and prep data
# - Preview a few rows
# - Split data in to training set (Age not null) and 'to-be-predicted' set
# - Predict age

# In[ ]:


# Read the data
file_train='../input/train.csv'
file_test='../input/test.csv'
    
df_train = pd.read_csv(file_train,index_col='PassengerId')
df_test = pd.read_csv(file_test,index_col='PassengerId')  
l = len(df_train.index)
    

## All data train and test in one dataframe 
dfa = df_train.append(df_test)

# Drop unwanted features
dfa = dfa.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    
# Fill missing data: Fare with mean, Embarked with most frequent value
dfa[['Fare']] = dfa[['Fare']].fillna(value=dfa[['Fare']].mean())
dfa[['Embarked']] = dfa[['Embarked']].fillna(value=dfa['Embarked'].value_counts().idxmax())
    
# Convert categorical features into numeric
dfa['Sex'] = dfa['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# Convert 'Embarked' to one-hot
enbarked_one_hot = pd.get_dummies(dfa['Embarked'], prefix='Embarked')
dfa = dfa.drop('Embarked', axis=1)
dfa = dfa.join(enbarked_one_hot)


# In[ ]:


dfa.head()


# Split data in to training set (Age not null) and 'to-be-predicted' set (Age in nan)

# In[ ]:


# Split data in to training set (Age not null) and 'to-be-predicted' set (Age in nan)
df_age_train = dfa[dfa.Age.notnull()]
df_age_nan = dfa[dfa.Age.isnull()]


# #### Predict age
# - Create input X, output Y and X_test
# - The model for the regression. The regression problem may have a single output neuron and the neuron may have no activation function (jb).
# - Create a pipeline, do cross-validation and predict
# - Generate new input files for the training and test data but with predicted age
# 

# In[ ]:


# split data into input X and output Y
X = df_age_train.drop(['Age', 'Survived'], axis=1).values.astype(float)
Y = df_age_train['Age'].values.astype(float)

X_test = df_age_nan.drop(['Age', 'Survived'], axis=1).values.astype(float)

def age_model():
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# **Create a pipeline**

# In[ ]:


estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=age_model, epochs=100, batch_size=5, verbose=verbose)))
pipeline = Pipeline(estimators)


# **Cross-validation**

# In[ ]:


kfold = KFold(n_splits=2, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Result: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# Ups, the mean square erroe is very poor.
# 
# **Predict**

# In[ ]:


pipeline.fit(X, Y)
prediction_train = pipeline.predict(X)
prediction_test = pipeline.predict(X_test)


# ** Generate new input files for the training and test data with predicted age **

# In[ ]:


# Create a data frame with PassengerId and predicted age
df_age_pred = pd.DataFrame({
    'PassengerId': df_age_nan.index,
    'Age_pred': prediction_test.astype(int)
})
df_age_pred.set_index('PassengerId', inplace=True)
   

# Add column with predicted age to the dataframe with all data (dfa)
dfa2 = df_train.append(df_test) 
dfa_pred = pd.concat([dfa2, df_age_pred], axis=1)   

# Update Age column with prediction where nan and remove Age_pred
dfa_pred['Age'] = np.where(pd.isnull(dfa_pred['Age']), dfa_pred['Age_pred'] , dfa_pred['Age'])
dfa_pred = dfa_pred.drop(['Age_pred'], axis=1)

# Create new files
l = len(df_train)
df_train2 = dfa_pred[0:l] 
df_test2 = dfa_pred[l:] 
df_test2 = df_test2.drop(['Survived'], axis=1)

df_train2.to_csv('train-age-predicted.csv')
df_test2.to_csv('test-age-predicted.csv')


# The two file above can be used instead of the original input data. Give it a try with the part 1 with the new file names. The result is still not that great (scroe of ~78%).

# ### Part 3 - Wrangle, prepare, cleanse the titanic data manually
# Source: [Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook) by Manav Sehgal
# 
# Neither the simple cleaning nor the age prediction generated any score above 80%. Eventually manual data prep will help.
# 
# 
# **Re-read data, drop the Cabin and Ticket features**

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]


# **Creating new feature extracting from existing ...**

# **New title featue**

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


# convert the categorical titles to ordinal.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Drop Name feature
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]


# **Converting a categorical feature**

# In[ ]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# **Completing a numerical continuous feature**
# Guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender, and Pclass. Guess Age values using [median](https://en.wikipedia.org/wiki/Median) values for Age across sets of Pclass and Gender feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on..

# In[ ]:


train_df.head(10)


# In[ ]:


guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)


# **Create age bands and save ordinals based on these bands**

# In[ ]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()


# **Create new feature for FamilySize which combines Parch and SibSp**

# In[ ]:


combine = [train_df, test_df]
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# **Create another feature called IsAlone**

# In[ ]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# **Drop Parch, SibSp, and FamilySize features in favor of IsAlone**

# In[ ]:


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]


# **Create an artificial feature combining Pclass and Age**

# In[ ]:


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# **Completing a categorical feature**

# In[ ]:


freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)


# **Fare feature to ordinal values based on the FareBand**
# 

# In[ ]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

combine = [train_df, test_df]


# **Save wrangled training and test data to files**

# In[ ]:


train_df.to_csv('train-wrangled.csv',index=False)
test_df.to_csv('test-wrangled.csv',index=False)


# ### Part 4 - Predict 'Survived' with Kears based on wrangled input data
# This part is more or less like part 1. Just with other input data.

# In[ ]:


# Input files
file_train='train-wrangled.csv'
file_test='test-wrangled.csv'

# read training data
train_df = pd.read_csv(file_train,index_col='PassengerId')


# In[ ]:


# Show the columns
train_df.columns.values


# In[ ]:


# Show the shape
train_df.shape


# In[ ]:


# preview the training dara
train_df.head()


# In[ ]:


# Show that there isn't any NaN data 
train_df.isnull().sum()


# Split training data into input X and output Y

# In[ ]:


# X contains all columns except 'Survived'  
X = train_df.drop(['Survived'], axis=1).values.astype(float)

# It is almost always a good idea to perform some scaling of input values when using neural network models (jb).

scale = StandardScaler()
X = scale.fit_transform(X)

# Y is just the 'Survived' column
Y = train_df['Survived'].values


# #### Run GridSearch, optionally
# GridSearch is very time consuming. 
# Set `run_gridsearch = True ` in case you really want to run it.
# Or simply tune optimizers, inits, epochs and batches. 
# 

# In[ ]:


run_gridsearch = False

if run_gridsearch:
    
    start_time = time.time()
    if verbose: print (time.strftime( "%H:%M:%S " + "GridSearch started ... " ) )
    optimizers = ['rmsprop', 'adam']
    inits = ['glorot_uniform', 'normal', 'uniform']
    epochs = [50, 100, 200, 400]
    batches = [5, 10, 20]
    
    model = KerasClassifier(build_fn=create_model, verbose=verbose)
    
    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X, Y)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    if verbose: 
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        elapsed_time = time.time() - start_time  
        print ("Time elapsed: ",timedelta(seconds=elapsed_time))
        
    best_epochs = grid_result.best_params_['epochs']
    best_batch_size = grid_result.best_params_['batch_size']
    best_init = grid_result.best_params_['init']
    best_optimizer = grid_result.best_params_['optimizer']
    
else:
    # pre-selected paramters
    best_epochs = 200
    best_batch_size = 5
    best_init = 'glorot_uniform'
    best_optimizer = 'rmsprop'


# #### Build model and predit
# - Create a classifier with best parameters
# - Fit model 
# - Predict 'Survived'

# In[ ]:


# Create a classifier with best parameters
model_pred = KerasClassifier(build_fn=create_model, optimizer=best_optimizer, init=best_init, epochs=best_epochs, batch_size=best_batch_size, verbose=verbose)
model_pred.fit(X, Y)

# Read test data
test_df = pd.read_csv(file_test,index_col='PassengerId')

# Create X_test
X_test = test_df.values.astype(float)
# Scaling
X_test = scale.transform(X_test)

# Predict 'Survived'
prediction = model_pred.predict(X_test)



# Save predictions

# In[ ]:


submission = pd.DataFrame({
    'PassengerId': test_df.index,
    'Survived': prediction[:,0],
})

submission.sort_values('PassengerId', inplace=True)    
submission.to_csv('submission-manual-cleansing.csv', index=False)


# #### Result part 4 - Predict 'Survived' with Kears based on wrangled input data
# Submitting the prediction csv will have scroe of ~78%. Not a good result :-(

# ## Summary
# The results are not really great. Either the simple multilayer perceptrons in this notebook are too simple for solving the prediction problem or the problem is not suited for deep learning and therefore other other algorithms (SVM, decision tree) are just better. At least it was fun playing with Keras.
# 
# Your feedback and suggestions are very welcome.

# In[ ]:




