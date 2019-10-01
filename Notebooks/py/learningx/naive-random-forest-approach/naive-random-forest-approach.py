#!/usr/bin/env python
# coding: utf-8

# 
# 
# A run-through of Mike's channel: https://www.youtube.com/watch?v=0GrciaGYzV0&t=766s
# ------------------------------------

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
x = pd.read_csv('../input/train.csv')
y = x.pop("Survived")


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# **1. Input Examination**

# In[ ]:


x.describe()


# **All the above is only for the numerical input, however, there are still categorical columns have not shown up, such as Embarked, Cabin... My confusion is why they did not show up in describe. Should describe function give all summary of columns regardless their data type?** 

# **2. Remove NAs  
# From output[4], we found out there are some NAs in Age group. In this case, I choose to use the mean age to replace them.**

# In[ ]:


x["Age"].fillna(x.Age.mean(), inplace = True)
x.describe()


# **3.Model Prediction**

# **We cannot regress the whole data set due to data type error I guess thus we may try numerical data first**

# In[ ]:


numerical_variables = list(x.dtypes[x.dtypes != object].index)
x[numerical_variables].head()


# In[ ]:


model = RandomForestRegressor(n_estimators = 100, oob_score = True, random_state = 42 )
model.fit(x[numerical_variables],y)


# **Let's take a look at this model. 
# Out-of-bag(OOB) description :http://scikitlearn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html**

# In[ ]:


model.oob_score_


# In[ ]:


y_oob = model.oob_prediction_
print( "c-stat:",roc_auc_score(y,y_oob))


# **roc_auc_score is approximately 74%, which sets up a basic benchmark. In order to enhance the model,  we need to consider about categorical variables.**

# In[ ]:


def describe_categorical(x):   
    from IPython.display import display, HTML
    display(HTML(x[x.columns[x.dtypes=="object"]].describe().to_html()))


# In[ ]:


describe_categorical(x)


# **There are missing values in Cabin and Embarked
# Name is irrelevant so we can ignore it
# Cabin we only keep the first letter which indicates the class of ticket**

# In[ ]:


x.drop(["Name","Ticket","PassengerId"], axis = 1, inplace = True)


# In[ ]:



def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"
    
x["Cabin"] = x.Cabin.apply(clean_cabin)


# In[ ]:


describe_categorical(x)


# In[ ]:


categorical_variables = ["Sex","Cabin","Embarked"]
for variable in categorical_variables:
    x[variable].fillna("Missing", inplace = True)
    dummies = pd.get_dummies(x[variable], prefix = variable)
    x = pd.concat([x,dummies],axis=1)
    x.drop([variable],axis = 1, inplace = True)


# In[ ]:


model = RandomForestRegressor(100,oob_score = True,n_jobs = -1,random_state = 42)
model.fit(x,y)
print ("c-stat",roc_auc_score(y,model.oob_prediction_))


# **Optimize Parameters**
# 

# In[ ]:


#To find which parameters are important
feature_importance = pd.Series(model.feature_importances_, index = x.columns )
feature_importance.sort()
feature_importance.plot(kind = "barh",figsize =(7,6) )


# In[ ]:


#To find n_estimators
n_estimators_options = [30, 50 ,100 ,200 ,1000, 2000 ]
results = []
for tree in n_estimators_options:
    model = RandomForestRegressor(tree,oob_score = True,n_jobs = -1,random_state = 42)
    model.fit(x,y)
    print(tree, "trees")
    roc = roc_auc_score(y, model.oob_prediction_)
    print("C-stat",  roc)
    results.append(roc)
    print ("")
    
pd.Series(results, n_estimators_options).plot()
    
    


# **From above we know we will choose  n_estimator = 1000**

# In[ ]:


#To find max_features
max_features_options = ["auto", None , "sqrt","log2" ,0.9, 0.2 ]
results = []
for max_features in max_features_options:
    model = RandomForestRegressor(1000,oob_score = True,n_jobs = -1,max_features = max_features,random_state = 42)
    model.fit(x,y)
    print(max_features, "max_features")
    roc = roc_auc_score(y, model.oob_prediction_)
    print("C-stat",  roc)
    results.append(roc)
    print ("")
    
pd.Series(results, max_features_options).plot()


# In[ ]:


#To find min_samples_leaf
min_sample_leaf_options = np.asarray(range(1,10))
results = []
for min_sample_leaf in min_sample_leaf_options:
    model = RandomForestRegressor(1000,oob_score = True,
                                  n_jobs = -1,max_features = "auto",
                                  random_state = 42, 
                                  min_samples_leaf =min_sample_leaf )
    model.fit(x,y)
    print(min_sample_leaf, "min_sample_leaf")
    roc = roc_auc_score(y, model.oob_prediction_)
    print("C-stat",  roc)
    results.append(roc)
    print ("")
    
pd.Series(results, min_sample_leaf_options).plot()


# In[ ]:


x.describe()


# In[ ]:


#Thus the final model will be
final_model = RandomForestRegressor(1000,oob_score = True,
                                  n_jobs = -1,max_features = "auto",
                                  random_state = 42, 
                                  min_samples_leaf =5)
final_model.fit(x,y)
roc = roc_auc_score(y, final_model.oob_prediction_)
print("C-stat", roc)


# In[ ]:


#Use the final model to predict survivals 
test_data = pd.read_csv("../input/test.csv")
passengerId = test_data["PassengerId"]
test_data.drop(["Name","Ticket","PassengerId"], axis = 1, inplace = True)
test_data["Cabin"] = test_data.Cabin.apply(clean_cabin)
test_data["Age"].fillna(test_data.Age.mean(), inplace = True)
test_data["Fare"].fillna(test_data.Fare.mean(),inplace = True)
for variable in categorical_variables:
    test_data[variable].fillna("Missing", inplace = True)
    dummies = pd.get_dummies(test_data[variable], prefix = variable)
    test_data = pd.concat([test_data,dummies],axis=1)
    test_data.drop([variable],axis = 1, inplace = True)

length = len(test_data["Cabin_G"])
test_data["Cabin_T"] = pd.Series(np.zeros(length), index = test_data.index)
test_data["Embarked_Missing"] = pd.Series(np.zeros(length), index = test_data.index)
test_data.describe()


# In[ ]:


results= final_model.predict(test_data)
for i in range(0,len(results)):
    if results[i] > 0.4:
        results[i] = 1
    else: 
        results[i] = 0
results= results.astype(int)
submission = pd.DataFrame({
        "PassengerId": passengerId,
        "Survived": results
    })
submission.head()
submission.to_csv('submission.csv', index=False)

