#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''Titanic Survivors Prediction using Python
    A classification project by Vikas Zingade.
    Submitted to Kaggle competition.
'''


# In[ ]:


#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

print('All Good!')


# In[ ]:


#Read and write datasets
#Re-usable utility functions

#import pandas as pd

def read_data(train = 'train', test = 'test'):
    '''Inputs: "Train" CSV filename, "Test" CSV filename
        To read "train" and "test" data from CSV
        Outputs: "Train" and "Test" DataFrames
        
    Don't add ".csv" at the end of the filanmes
    '''
    
    train = pd.read_csv("../input/" + train + ".csv")
    test  = pd.read_csv("../input/" + test + ".csv")
    
    return train, test

#To write a DataFrame to CSV
def write_df(df, filename):
    '''Inputs: DataFRame to be written, target CSV filename
        To write DataFrame to CSV
        Outputs: None
        
    Don't add ".csv" at the end of the filanmes
    '''
    
    df.to_csv(filename + '.csv', index = False)
    print(filename, 'written to csv.')
    
print('All Good!')


# In[ ]:


#Read data: train and test
train, test = read_data()

print('All Good!')


# In[ ]:


print('Train data:', train.columns.values)
print('\n')
print('Test data:', test.columns.values)


# **Data Preprocessing**

# In[ ]:


#Combine train and test datasets for preprocessing: Missing Value Analysis and Outlier Analysis
predictor_cols = test.columns.values

#Combine train and test
data = pd.concat([train[predictor_cols], test])
data.is_copy = False

print('All Good!')


# In[ ]:


data.head()


# 1. PassengerId is a primary key. We will drop it before fitting the model.
# 2. SibSp and Parch relate to traveling with family (Kaggle data dictionary). To avoid the effects of multicollinearity, we will combine these two variables into one categorical variable: travelling alone or not.
# 3. Ticket will dropped.

# In[ ]:


data['Travelling_alone'] = data['SibSp'] + data['Parch']
data['Travelling_alone'] = np.where(data['Travelling_alone'] > 0, 0, 1)

data.drop(['SibSp', 'Parch'], axis=1, inplace=True)

print('All Good!')


# In[ ]:


data.drop(['PassengerId', 'Ticket', 'Name'], axis=1, inplace=True)

print('All Good!')


# In[ ]:


data.head()


#  **Missing Value Analysis**

# In[ ]:


data.info()


# In[ ]:


#Missing values count
#Re-usable utility function to get the columns with missing values

def na_count(data):
    '''Inputs: "data" DataFrame, "id" primary-key column
        To count the number of NAs in every column of the DataFrame.
        Outputs: A list of count of NAs in every column of teh DataFrame
    '''

    na_cols = []
    for i in data.columns:
        if data[i].isnull().sum() != 0:
            na_cols.append([i, data[i].isnull().sum(), round( data[i].isnull().sum() / len(data[i]) * 100, 4)])
    
    return na_cols

print('All Good!')


# In[ ]:


na_cols = na_count(data)

print('Missing Values:')
for i in na_cols:
    print(i)


# **Missing Value Analysis: Age**
# 
# 20% of the values are missing. We will impute the missing values. Let's plot the histogram to decide wheteher to use mean or median for imputation..

# In[ ]:


plt.figure()
_ = data["Age"].hist()


# The 'Age' variable is right-skewed. Imputing the missing values with mean will be biased. So we will use median.

# In[ ]:


data['Age'].fillna(data['Age'].median(skipna=True), inplace=True)

print('All Good!')


# **Missing Value Analysis: Fare**
# 
# Less than 1% missing values. Let's look at the Fare variable.

# In[ ]:


data['Fare'].head()


# In[ ]:


data['Fare'].describe()


# In[ ]:


plt.figure()
_ = data['Fare'].hist()


# We can see that there is an outlier here! Apart from that, 'Fare' is right-skewed too. We can use median for imputation.

# In[ ]:


data['Fare'].fillna(data['Fare'].median(skipna=True), inplace=True)

print('All Good!')


# **Missing Value Analysis: Cabin**
# 
# 77% of the values misisng. We will drop this column.

# In[ ]:


data.drop('Cabin', axis=1, inplace = True)

print('All Good!')


# **Missing Value Analysis: Embarked**
# 
# 

# In[ ]:


data['Embarked'].head()


# In[ ]:


data['Embarked'].describe()


# Embarked is a categorical variable. So impute the missing values with the most frequent value. Use countplot to find out.

# In[ ]:


plt.figure()
_ = sbn.countplot(x='Embarked',data=data)


# Fill the 'Embarked missing values with 'S'

# In[ ]:


data['Embarked'].fillna('S', inplace = True)

print('All Good!')


# In[ ]:


na_cols = na_count(data)

print('Missing Values:')
for i in na_cols:
    print(i)


# Missing Value Analysis is done.

# **Outlier Analysis**

# In[ ]:


data.info()


# Outlier Analysis for: Pclass, Age, Fare, 

# **Outlier Analysis: Pclass**

# In[ ]:


plt.figure(figsize=(15,3))
_ = data.boxplot(column='Pclass', vert=False)


# Pclass looks good.

# **Outlier Analysis: Age**

# In[ ]:


plt.figure(figsize=(15,3))
_ = data.boxplot(column='Age', vert=False)


# Age looks good.

# **Outlier Analysis: Fare**

# In[ ]:


plt.figure(figsize=(15,3))
_ = data.boxplot(column='Fare', vert=False)


# Ooh! Looks there is a potential outlier! 

# In[ ]:


data[data['Fare'] == max(data['Fare'])]


# There are 4 observations with this value. So this cannot be an outlier!

# In[ ]:


#If decided to go ahead with the outlier for 'Fare'
#Fare_max = max(data['Fare'])
#data.loc[data['Fare'] == Fare_max] = -999
#data.loc[data['Fare'] == -999] = data['Fare'].median()


# Outlier Analysis is done.

# **Exploratory Data Analysis**

# **Feature Engineering** 

# In[ ]:


data.info()


# In[ ]:


data['Pclass'].head()


# In[ ]:


data['Pclass'].unique()


# We will pandas get_dummies to convert 'Pclass' into boolean categorical variable.

# In[ ]:


data = pd.get_dummies(data, columns=['Pclass'])

print('All Good!')


# In[ ]:


data.drop('Pclass_3', axis=1, inplace=True)

print('All Good!')


# In[ ]:


data.head()


# In[ ]:


data['Sex'].head()


# In[ ]:


data['Sex'].unique()


# 'Sex' is nominal categorical variable. We will use get_dummies method to create separate boolean columns for 'male and 'female'

# In[ ]:


data = pd.get_dummies(data, columns=['Sex'])

print('All Good!')


# In[ ]:


data.head()


# In[ ]:


data.drop('Sex_female', axis=1, inplace=True)

print('All Good!')


# In[ ]:


data['Embarked'].head()


# In[ ]:


data['Embarked'].unique()


# In[ ]:


data = pd.get_dummies(data, columns=['Embarked'])

print('All Good!')


# In[ ]:


data.drop('Embarked_Q', axis=1, inplace=True)

print('All Good!')


# In[ ]:


data.head()


# From EDA we learnt that being minor increased the chanes of survival. Let's add another variable 'is_minor' if the age is below 16 years.

# In[ ]:


data['is_minor'] = np.where(data['Age'] <= 16, 1, 0)


# In[ ]:


data.head()


# For now, this looks fine. We will now build different Machine Learning models to predict the survivors. We will also try different Feature Engineering and Selection methods and compare their respective models.

# **Dirty Models**
# 
# It looks like 'Sex'=female had the high rate of survival. So let's assume all females survived and make a submission.

# In[ ]:


train_df = data.iloc[:891]
test_df = data.iloc[891:]

print('All Good!')


# **Survivors Prediction with Machine Learning models**
# 
# We will try out Decison Tree classifier, Random Forest, KNN classifier, and Logistic classier for now.
# We will also employ train-test split startegy to calculate the error metrics in house. Later we will chose the best model and submit to the Kaggle competition.

# In[ ]:


test_df.head()


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train_df, train['Survived'], test_size = 0.33, random_state = 0)

print('All Good!')


# **Decision Tree Classifier**
# 
# Decision Tree Classifier takes only arrays as input. We will now build a Decision Tree classifier model.

# In[ ]:


from sklearn import tree

DTC3 = tree.DecisionTreeClassifier(max_depth = 3)
DTC3.fit(X_train, Y_train)
Y_pred_DTC3 = DTC3.predict(X_test)

print('All Good!')


# In[ ]:


from sklearn.metrics import mean_absolute_error

DTC3_mae = mean_absolute_error(Y_test, Y_pred_DTC3)
print('Decision Tree Classifier with max_depth = 3:\nMean Absolute Error: ', DTC3_mae)


# In[ ]:


import graphviz 

DTC3_view = tree.export_graphviz(DTC3, out_file=None, feature_names = X_train.columns.values, rotate=True) 
DTC3viz = graphviz.Source(DTC3_view)
DTC3viz


# In[ ]:


DTC3_pred = DTC3.predict(test_df)

print('All Good!')


# In[ ]:


DTC3_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':DTC3_pred})

DTC3_submission.to_csv('DTC3_submission.csv', index=False)

print('All Good!')


# **Random Forest**
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators = 100)
RFC.fit(X_train, Y_train)
Y_pred_RFC = RFC.predict(X_test)

print('All Good!')


# In[ ]:


RFC_mae = mean_absolute_error(Y_test, Y_pred_RFC)

print('Random Forest Classifier with max_depth = 3:\nMean Absolute Error: ', RFC_mae)


# In[ ]:


RFC_pred = RFC.predict(test_df)

print('All Good!')


# In[ ]:


RFC_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':RFC_pred})

RFC_submission.to_csv('RFC_submission.csv', index=False)

print('All Good!')


# **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression()
LogReg.fit(X_train, Y_train)
Y_pred_LogReg = LogReg.predict(X_test)

print('All Good!')


# In[ ]:


LogReg_mae = mean_absolute_error(Y_test, Y_pred_LogReg)

print('Random Forest Classifier with max_depth = 3:\nMean Absolute Error: ', LogReg_mae)


# In[ ]:


LogReg_pred = LogReg.predict(test_df)

print('All Good!')


# In[ ]:


LogReg_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':LogReg_pred})

LogReg_submission.to_csv('LogReg_submission.csv', index=False)

print('All Good!')


# 
