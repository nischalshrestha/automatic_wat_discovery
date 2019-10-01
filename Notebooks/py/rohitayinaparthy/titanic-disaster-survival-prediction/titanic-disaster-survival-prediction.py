#!/usr/bin/env python
# coding: utf-8

# #                        Titanic Disaster Survival Prediction

# In[ ]:


### Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##from plotline import *


# In[ ]:


# Read the data set

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# In[ ]:





# ### Exploring the Titanic dataset

# In[ ]:


train_data.tail()


# In[ ]:


### Checking the dimension of data
train_data.shape


# In[ ]:


train_data.dtypes


# In[ ]:


## Checking the passenger class counts
train_data.Pclass.value_counts()


# In[ ]:


## chekking the count of males and females
train_data.Sex.value_counts()


# In[ ]:


## chekking the count of males and females
train_data.Survived.value_counts()


# In[ ]:


train_data.columns


# In[ ]:


train_data.Age.describe()


# In[ ]:





# In[ ]:





# 
# ### Feature Engineering

# In[ ]:


## Extracting surnames
train_data["Title"] = train_data.Name.str.split(",").str.get(1).str.split(".").str.get(0).str.strip()


# In[ ]:


## Checking all the possible values in surname field
train_data["Title"].value_counts()


# In[ ]:


## Titles with low cell values are clubbed and assigned to "Rare title"


# In[ ]:


rare_title =  ['Dr',
 'Rev',
 'Col',
 'Major',
 'Capt',
 'Jonkheer',
 'Sir',
 'Lady',
 'Don',
 'the Countess']


# In[ ]:


train_data["Title"][train_data["Title"].str.contains("Mlle|Ms")] = "Miss" 
train_data["Title"][train_data["Title"].str.contains("Mme")] = "Mrs" 


# In[ ]:


train_data["Title"][train_data["Title"].str.contains('Dr|Rev|Col|Major|Capt|Jonkheer|Sir|Lady|Don|the Countess')] = "Rare Title" 


# In[ ]:


train_data.Title.value_counts()


# In[ ]:


### Identifying titles to test data well


# In[ ]:





# In[ ]:


## Extracting surnames
test_data["Title"] = test_data.Name.str.split(",").str.get(1).str.split(".").str.get(0).str.strip()


# In[ ]:


## Checking all the possible values in surname field
test_data["Title"].value_counts()


# In[ ]:


## Titles with low cell values are clubbed and assigned to "Rare title"


# In[ ]:


test_data["Title"].value_counts().index


# In[ ]:


rare_title_test =  ['Col', 'Rev', 'Dona', 'Dr']


# In[ ]:


test_data["Title"][test_data["Title"].str.contains("Ms")] = "Miss" 


# In[ ]:


test_data["Title"][test_data["Title"].str.contains('Col|Rev|Dona|Dr')] = "Rare Title" 


# In[ ]:


test_data.Title.value_counts()


# ### Creating family size

# In[ ]:


train_data['Family_Size']=train_data['SibSp']+train_data['Parch']+1


# In[ ]:


test_data['Family_Size']=test_data['SibSp']+test_data['Parch']+1


# In[ ]:


train_data.Family_Size.describe()


# In[ ]:


## Descritize family variable
train_data["FsizeD"] = "NaN"
train_data["FsizeD"][train_data["Family_Size"] == 1]  = "singleton"


# In[ ]:


mask = (train_data["Family_Size"] < 5) & (train_data["Family_Size"] > 1)
train_data.loc[mask,"FsizeD"] = "small"
train_data.loc[(train_data.Family_Size > 4),"FsizeD"] = "large"


# In[ ]:


train_data.FsizeD.value_counts()


# In[ ]:





# In[ ]:


#### Similar operations for test data as well.


# In[ ]:


test_data.shape


# In[ ]:


test_data['Family_Size']=test_data['SibSp']+test_data['Parch']+1


# In[ ]:


test_data.Family_Size.describe()


# In[ ]:


## Descritize family variable
test_data["FsizeD"] = "NaN"
test_data["FsizeD"][test_data["Family_Size"] == 1]  = "singleton"


# In[ ]:


mask = (test_data["Family_Size"] < 5) & (test_data["Family_Size"] > 1)
test_data.loc[mask,"FsizeD"] = "small"
test_data.loc[(test_data.Family_Size > 4),"FsizeD"] = "large"


# In[ ]:


test_data.FsizeD.value_counts()


# In[ ]:





# #### Removing irrelevant columns not required for model building
# 

# In[ ]:


### Removing cols from train data
train_data_1 = train_data.drop(axis=1,columns=["PassengerId","Name","Ticket","Cabin","Family_Size"])
train_data_1.head()


# In[ ]:


### Removing cols from train data
test_data_1 = test_data.drop(axis=1,columns=["PassengerId","Name","Ticket","Cabin","Family_Size"])
test_data_1.head()


# In[ ]:





# In[ ]:





# ### Distinguish predictors and target variables 

# In[ ]:


## Storing target variable in y
y = train_data_1["Survived"]


# In[ ]:


## Storing predictors in X
X = train_data_1[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked','Title','FsizeD']]
X.head(1)


# ### Creating dummy variables

# In[ ]:


### Creating dummy variable in training data set
X = pd.get_dummies(data = X,drop_first=True)
X.head(1)


# In[ ]:


### Creating dummy variable in test data set
test_data_1 = pd.get_dummies(data = test_data_1,drop_first=True)
test_data_1.head(1)


# In[ ]:





# In[ ]:


### Split Training data into train and validation data


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_validation, y_train,y_validation = train_test_split(X,y,test_size = 0.2,random_state = 3454)


# In[ ]:





# ### Handling Null values

# In[ ]:


### Check all the null values we have in all formats of data


# In[ ]:


X_train.isnull().any()


# In[ ]:


## So Only Age value has null values in X_Train so lets check X_validation as well
X_validation.isnull().any()


# In[ ]:


### So both X_train and X_validation has Age null values


# In[ ]:





# #### How about null values in test data..-lets check what fields have null values in test data

# In[ ]:


test_data_1.isnull().any()


# In[ ]:


## Check the null fare values. check its corresponding categorical values and impute accordingly


# In[ ]:


test_data_1[test_data_1.Fare.isnull()]


# In[ ]:


## The above person belongs to class 3 and lets take median of class 3 and embarkment = S people and impute median value


# In[ ]:


fare_null_filter = ((test_data_1.Pclass == 3) & (test_data_1.Embarked_S == 1))
fare_null_df = test_data_1.loc[fare_null_filter,]
fare_null_df["Fare"].median()


# In[ ]:


test_data_1.loc[test_data_1.Fare.isnull(),"Fare"] = fare_null_df["Fare"].median()


# In[ ]:


### Check again the null values of test data


# In[ ]:


test_data_1.isnull().any()  ### Now we have null values in only in Age


# In[ ]:





# In[ ]:


### Imputing mean age to all the null values


# In[ ]:


####  from fancyimpute import MICE   - Donot find this package but can consider install later


# In[ ]:





# In[ ]:


from sklearn.preprocessing import Imputer
imputer = Imputer(axis=0,missing_values="NaN",strategy="mean")
X_train[:] = imputer.fit_transform(X_train)
X_validation[:] = imputer.transform(X_validation)


# In[ ]:


### Transform the mean value of training set data to test data set
test_data_1[:] = imputer.transform(test_data_1)


# In[ ]:





# #### Check the null values in all the datasets again

# In[ ]:


X_train.isnull().any()


# In[ ]:


X_validation.isnull().any()


# In[ ]:


test_data_1.isnull().any()


# Since there are no null values, Now check we create additional featured by using Age

# In[ ]:



# Now that we know everyone's age, we can create age dependent variables child and mother


# In[ ]:


# Train data
X_train["Child_col"] = "NaN"
X_train.loc[X_train.Age < 18,"Child_col"] = "Child"
X_train.loc[X_train.Age >= 18,"Child_col"] = "Adult"


# Validation data
X_validation["Child_col"] = "NaN"
X_validation.loc[X_validation.Age < 18,"Child_col"] = "Child"
X_validation.loc[X_validation.Age >= 18,"Child_col"] = "Adult"


### Test data
test_data_1["Child_col"] = "NaN"
test_data_1.loc[test_data_1.Age < 18,"Child_col"] = "Child"
test_data_1.loc[test_data_1.Age >= 18,"Child_col"] = "Adult"



# In[ ]:





# In[ ]:





# 
# ## Creating Mother variable

# In[ ]:


## Train data
X_train["Mother_col"] = "Not Mother"
X_train.loc[((X_train.Parch > 0) & (X_train.Title_Miss == 0) & (X_train.Sex_male == 0) & (X_train.Child_col == "Adult")),"Mother_col"] = "Mother"

## Validation data
X_validation["Mother_col"] = "Not Mother"
X_validation.loc[((X_validation.Parch > 0) & (X_validation.Title_Miss == 0) & (X_validation.Sex_male == 0) & (X_validation.Child_col == "Adult")),"Mother_col"] = "Mother"

## Test data
test_data_1["Mother_col"] = "Not Mother"
test_data_1.loc[((test_data_1.Parch > 0) & (test_data_1.Title_Miss == 0) & (test_data_1.Sex_male == 0) & (test_data_1.Child_col == "Adult")),"Mother_col"] = "Mother"



# In[ ]:





# In[ ]:


### Create dummy variables for the newly created cols in all the 3 data sets- Train, Validation and Test


# In[ ]:


X_train = pd.get_dummies(X_train,drop_first=True)


# In[ ]:


X_validation = pd.get_dummies(X_validation,drop_first=True)


# In[ ]:


test_data_1 = pd.get_dummies(test_data_1,drop_first=True)


# In[ ]:





# In[ ]:



### Check the columns of all the datasets are in same order---- All the cols are in the sames order as we can check below
X_train.columns


# In[ ]:


X_validation.columns


# In[ ]:


test_data_1.columns


# In[ ]:





# In[ ]:





# ### Calculating Inflation factor - to check multiple dependency
# 
# #### Note: that VIF is not required as the multicolinearity will not effect RandomForrest model

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


from statsmodels.tools.tools import add_constant


# In[ ]:


### Adding contanst(intercept) to train data 
#X_train = add_constant(X_train)


# In[ ]:


### Lets try to create new columns and calculate vif

pd.Series([variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])],index = X_train.columns)


# In[ ]:





# ## Model building -  Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[ ]:


rf_Classifier = RandomForestClassifier(random_state = 1)
rf_Classifier.fit(X_train,y_train)


# In[ ]:


y_pred = rf_Classifier.predict(X_validation)


# In[ ]:


mean_absolute_error(y_validation,y_pred)


# In[ ]:


validation_check_df = pd.DataFrame({"val":y_validation,"pred":y_pred})


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:



confusion_matrix(y_validation,y_pred)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_validation,y_pred)


# In[ ]:


print("The Accuracy of model using Random Forest is : " + str(accuracy_score(y_validation,y_pred)))


# In[ ]:





# ## Model  using Cross validation
#     

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline


# In[ ]:


my_pipeline = make_pipeline(RandomForestClassifier())


# In[ ]:


scores = cross_val_score(my_pipeline,X_train,y_train,scoring='accuracy')


# In[ ]:


scores.mean()


# In[ ]:


print("The Accuracy of model using Random Forest and Cross validation with pipelines is : " + str(scores.mean()))


# In[ ]:





# ## Model - XGBoost

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


model_3 = XGBClassifier(n_estimators=2000,learning_rate=0.05)


# In[ ]:


model_3.fit(X_train,y_train,verbose= False,early_stopping_rounds=10,eval_set=[(X_validation,y_validation)])


# In[ ]:


y_pred_3 = model_3.predict(X_validation)


# In[ ]:


mean_absolute_error(y_validation,y_pred_3)


# In[ ]:


confusion_matrix(y_validation,y_pred_3)


# In[ ]:


accuracy_score(y_validation,y_pred_3)


# In[ ]:


print("The Accuracy of model using XGBoost is : " + str(accuracy_score(y_validation,y_pred_3)))


# In[ ]:





# ## Model - Using cross validation and XGBoost 

# In[ ]:


my_pipeline_2 = make_pipeline(XGBClassifier(n_estimators=2000,learning_rate=0.05))

scores = cross_val_score(my_pipeline_2,X_train,y_train,scoring='accuracy')

print("The Accuracy of model using XGBoost and Cross validation with pipelines is : " + str(scores.mean()))


# In[ ]:





# ### Comparing the below models and its accuracies, we choose model built from 
# ### XGBoost  with accuracy 83.24%
# 
# 1. Random Forest - 81%
# 2. Cross validaiton using Random Forest - 80.6%
# 3. XGBoost - 83.24%
# 4. Crossvalition using XGBoost - 80.5%
# 
# 

# In[ ]:





# ### Predicting Survival in test data- Writing output to an Excel data file
# 
# ##### The final model deduced is "model_3"

# In[ ]:


## MAKE PREDICTIONS
y_test_pred = model_3.predict(test_data_1)


# In[ ]:


## Create output file with only passenger id's and predicted survival values


# In[ ]:


output = pd.DataFrame({"PassengerId":test_data.PassengerId, "Survived":y_test_pred})


# In[ ]:


output.to_csv("Titanic_Survival_submission.csv",index= False)


# In[ ]:





# In[ ]:




