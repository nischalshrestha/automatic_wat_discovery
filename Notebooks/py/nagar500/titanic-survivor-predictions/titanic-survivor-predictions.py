#!/usr/bin/env python
# coding: utf-8

# # Steps followed for problem solving - 
# 1. Loading the data
# 2. Exploratory Data Analysis (EDA)
# 3. Missing value treatment
# 4. Scaling the features
# 5. One-hot encoding the features
# 6. Building the model
# 
# 

# **1. Problem Statement - **
# This is a binary classification problem.  In the problem, we need to predict of the passenger will survive or not. 
# **
# 2. Hypothesis Generation - **
# 
# It involves finding/thinking of the features which might affect the outcome/prediction. 
# 
# Here are some of the factors which I think might affect the survical rate - 
# 1. Gender : Gender plays an important role in the prediction. 
# 2. Fare : It might be a factor worth considering. As the passengers who paid higher fare might have a better chance for survival. 
# 3. PClass - This will also play a role in survival rate. 
# 4. Age - Small children and senior citizens will have less chance of survival. So it will definitely affect the target variable. 
# 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from time import time
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **1. Reading training and testing files.**

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
# keeping copy of original datasets 
# 1. Train data  -  Has all the features along with target variable. 
# 2. Test data - Similar to training data , but it does not have target variable. 
train_original = train_df.copy()
test_original = test_df.copy()

train_df.head()



# In[ ]:


train_df['Cabin'].value_counts().head()


# In[ ]:


test_df['Cabin'].value_counts().head()


# In[ ]:


# ckecking structure of training dataset

train_df.columns


# In[ ]:


# checking structure of testing dataset 

test_df.columns


# **Target Variable **
# We will analyze the target variable. As it is a class variable with two outupts 1 - Survived and 0 - Not survived. We will plot a bar chart for this variable 
# 

# In[ ]:


train_df['Survived'].value_counts()


# In[ ]:


# we will now normalize this variable 
train_df['Survived'].value_counts(normalize = True)


# **Categorical Variable : **
# 
# 1. Categorical Variables  -  Sex , Cabin, Embarked, Ticket,PClass  

# In[ ]:


plt.figure(1)
plt.subplot(221)
train_df['Sex'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender')

plt.subplot(222)
train_df['Embarked'].value_counts(normalize=True).plot.bar(title= 'Embarked')

plt.subplot(223)
train_df['Pclass'].value_counts(normalize=True).plot.bar(title= 'Passenger_Class')

plt.subplot(224)
train_df['SibSp'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Sibling_Spouse')



plt.show()


# **Inference from the plots  - **
# 1. 60% passengers in the data set are Male. 
# 2. 70% passengers have embarked from 'S' port. 
# 3. More than 50% passengers belong to the PClass = 3
# 4. More than 60% of passengers came without their spouse or children. 
# 

# **Target Variable and it's relation with the categorical variables**

# In[ ]:


Gender=pd.crosstab(train_df['Sex'],train_df['Survived'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# From the above graph it is observed that female survivals were more as compared to the male survivors
# 
# Now let us Visualize the remaining variables

# In[ ]:


p_class=pd.crosstab(train_df['Pclass'],train_df['Survived'])
Dependents=pd.crosstab(train_df['SibSp'],train_df['Survived'])
Embarked=pd.crosstab(train_df['Embarked'],train_df['Survived'])


p_class.div(p_class.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show()

Embarked.div(Embarked.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()



# A passenger had greater chance of survival if he belongs to class = 1 and has embarked on 'C' port. 

# In[ ]:


#Now let us look at the correlation between the variables by heatmap 
matrix = train_df.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");


# **Correlation **
# We see that most correlated variables are - (SibSp - Parch ) and  (Survived - FAre)

# **Filling Missing Values *
# 
# We will need to fill the misisng values . SO we will do the following - 
# 1. For categorical values - filling with the most occured values(mode)
# 2. For numerical values - filling with mean/median

# In[ ]:


# printing information of all the columns in training and test datasets.
train_df.info()
print("-----------------Test Info------------------")
test_df.info()


# In[ ]:


# to see categorical data
train_df.describe(include=['O'])


# In[ ]:





# In[ ]:



def delete_features(df):
    return df.drop(['PassengerId','Ticket','Cabin'], axis=1)

def fill_value(df):
    df.Embarked = df.Embarked.fillna("S")
    #df['Age'] = df.Age.fillna(df.Age.median())
    df['Age'] = df.groupby(['Sex'],sort=False)['Age'].apply(lambda x: x.fillna(x.median()))
    df['Fare'] = df.groupby(['Pclass','Embarked'],sort=False)['Fare'].apply(lambda x : x.fillna(x.median()))
    return df

def fill_cabin(df):
    df.Cabin = df['Cabin'].fillna("N")
    '''Keep only the 1st character where Cabin is alphanumerical.'''
    df.Cabin = df['Cabin'].apply(lambda c : c[0])
    return df

def extract_name_from_title(df):
    df['Name'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    return df
    
    
def family_members(df):
    # to find if passenger has any family member or not in the ship.
    df['Family'] = df['SibSp'] + df['Parch'] + 1
   
    return df

def family_size_bin(df):
    """ Creating buckets as per the family size - Individual, Small , Medium and large family"""
    df.loc[ df['Family'] == 1, 'Family'] = "Individual"
    #df.loc[(df['Family'] > int(1)) & (df['Family'] <= int(2)), 'Family'] = "Small"
    #df.loc[(df['Family'] > 2) & (df['Family'] <= 5), 'Family'] = "Medium"
    #df.loc[(df['Family'] > 5) , 'Family'] ="Large"
    df['Family'].replace(to_replace = [2,3,4], value = 'small', inplace = True)
    df['Family'].replace(to_replace = [5,6], value = 'medium', inplace = True)
    df['Family'].replace(to_replace = [7,8,9, 10,11], value = 'large', inplace = True)
    return df
    
    
def transform_feature(df):
    df = delete_features(df)
    df = fill_value(df)
   # df = fill_cabin(df)
    df = extract_name_from_title(df)
    df = family_members(df)
    df = family_size_bin(df)
    return df

train_df = transform_feature(train_df)
test_df = transform_feature(test_df)

train_df.head()


# In[ ]:


display(train_df['Name'].value_counts())


# In[ ]:


""" Now we will bin the titles and also try to place  the same titles together """

def replace_title(df):
    df['Name'] = df['Name'].replace(['Lady','Countess','Capt', 'Col','Don', 'Major','Rev','Sir','Jonkheer','Dona'], 'Special')
    df['Name'] = df["Name"].replace(['Mlle','Ms','Miss'],'Miss')
    df['Name'] = df['Name'].replace(['Mrs','Mme'],'Mrs')
    return df

train_df = replace_title(train_df)
test_df = replace_title(test_df)
train_df.head()
    


# In[ ]:


display(train_df['Name'].value_counts())


# **3. Scaling Features:**
# We should perform scaling on numerical features. Normalization ensures that each feature is treated equally while applyingalgorithms.
# 
# We will be performing min max scaler in numerical features - Age, Fare, SibSp

# In[ ]:



from sklearn.preprocessing import MinMaxScaler

def encoder(df):
    scaler = MinMaxScaler()
    numerical = ['Age', 'Fare', 'SibSp','Parch']
    features_transform = pd.DataFrame(data= df)
    features_transform[numerical] = scaler.fit_transform(df[numerical])
    display(features_transform.head(n = 5))
    return df

train_df = encoder(train_df)
test_df = encoder(test_df)


# **4. One-hot encoding**
# There are several categorical features in the dataset. As the algorithms mainly work on numerical data, we will convert features - sex to numerical values.

# In[ ]:


def convert_numerical(df):
    #categorical = df.select_dtypes(exclude=["number"])
           
        
    #df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    #df = pd.get_dummies(df, columns=['Embarked'], drop_first=False)
    #df = pd.get_dummies(df, columns=['Name'], drop_first=False)
    #df = pd.get_dummies(df, columns=['Family'], drop_first=False)
    
    df = pd.get_dummies(df)
    
    encoded = list(df.columns)
    print("{} total features after one-hot encoding.".format(len(encoded)))
    print(encoded)
    return df
    

train_df_final = convert_numerical(train_df)
test_df_final = convert_numerical(test_df)


#print(test_df_final.Cabin_N)


# 
# We will drop the target variable form the training data set and will store in in another variable. 
# 

# In[ ]:


# splitting the data into training and testing data set 
from sklearn.model_selection import train_test_split
ytest  = train_df_final['Survived']
xtrain = train_df_final.drop(['Survived'], axis = 1)


X_train, X_test, y_train, y_test = train_test_split(xtrain, ytest, test_size=.25, random_state=1)


# **Implementing algorithm**
# We will be using RandomForestClassifier. 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogiscticRegression
from sklearn.metrics import make_scorer, accuracy_score,fbeta_score
from sklearn.grid_search import GridSearchCV


clf = RandomForestClassifier(random_state = 1)

#creating parameters to fit into algortihm 
parameters = {'n_estimators' : [10, 20, 30,50, 100] , 'max_features' : [0.6, 0.2, 0.3], 'min_samples_leaf' :[1,2,3], 
              'min_samples_split':[2,3,4,6]}

#parameters = {'penalty':['l1', 'l2'],'C': np.logspace(0, 4, 10)}

# calculating accuracy score
acc_scorer = make_scorer(accuracy_score)

# Running grid search 
grid_obj = GridSearchCV(clf, parameters,  scoring=acc_scorer, cv = 5)

# Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
best_clf = grid_obj.best_estimator_

# Fit the best parameter to the data. 
best_clf.fit(X_train, y_train)

#making predictions 
best_predictions = best_clf.predict(X_test)

#printing fbeta score and accuracy score of the optimized model . 
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))


# **Predicting the test data**

# In[ ]:


xtrain.head()


# In[ ]:


test_df_final.head()


# In[ ]:


pred_test = best_clf.predict(test_df_final)

submission = pd.read_csv('../input/gender_submission.csv')
submission['Survived']=pred_test
submission['PassengerId']=test_original['PassengerId']

#submission = pd.DataFrame({
#"PassengerId": test["PassengerId"],
#        "Survived": y_pred_rf_tunned})""""""
#submission.to_csv('submission_rf.csv', index = False)"''"
#converting to csv

pd.DataFrame(submission, columns=['PassengerId','Survived']).to_csv('randomforest.csv', index = False)
print(submission.head())

