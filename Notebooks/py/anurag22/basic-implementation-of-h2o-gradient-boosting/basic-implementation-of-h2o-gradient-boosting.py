#!/usr/bin/env python
# coding: utf-8

# # Basic Implementation of H2O Gradient Boosting for Classification on Titatnic DataSet

# ## The First Part is Data Preprocessing & Feature Engineering and the Second Part is H2O-GBM implementation
# 
# ## Important Note:  You can leave the first part because the objective of this Notebook is to learn some implementation basics of H20.

# # Part 1

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


from pandasql import sqldf


# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


data_train_all = pd.read_csv("../input/train.csv")
data_test_all=pd.read_csv("../input/test.csv")


# In[ ]:


data_test_all_pred=pd.read_csv("../input/test.csv")      ## This dataframe will be used in the last . This Dataframe doesn't have much significance


# In[ ]:


data_train_all.head()


# In[ ]:


data_train_all.info()

Columns Age , Cabin and Embarked has missing Values in training data
# In[ ]:


data_test_all.info()

Columns Age, Cabin, Fare have Missing Values in Test Set.
# ### Data Pre-Processing

# ##### Column: Name : Feature Engineer Column 'Name'
Regular Expression
# In[ ]:


import re


# In[ ]:


data_train_all['Title']= data_train_all.Name.apply(lambda a:re.search(' ([A-Z][a-z]+)\.',a).group(1))


# In[ ]:


data_test_all['Title'] = data_test_all.Name.apply(lambda a:re.search(' ([A-Z][a-z]+)\.',a).group(1))


# In[ ]:


data_train_all['Title'].head()


# In[ ]:


data_train_all.Title.value_counts()


# In[ ]:


data_test_all.Title.value_counts()

Group the titles of same types
# In[ ]:


data_train_all['Title'] = data_train_all['Title'].replace(['Mlle','Ms'],'Miss')  
data_train_all['Title'] = data_train_all['Title'].replace('Mme','Mrs')
data_train_all['Title'] = data_train_all['Title'].replace(['Capt','Col','Major'],'Army')
data_train_all['Title'] = data_train_all['Title'].replace(['Countess','Don','Dona','Jonkheer','Lady','Sir'],'Noble')


# In[ ]:


data_test_all['Title'] = data_test_all['Title'].replace(['Mlle','Ms'],'Miss')  
data_test_all['Title'] = data_test_all['Title'].replace('Mme','Mrs')
data_test_all['Title'] = data_test_all['Title'].replace(['Capt','Col','Major'],'Army')
data_test_all['Title'] = data_test_all['Title'].replace(['Countess','Don','Dona','Jonkheer','Lady','Sir'],'Noble')


# In[ ]:


data_train_all['Title'].value_counts()


# In[ ]:


data_test_all['Title'].value_counts()

Column:Drop the Column Ticket
# In[ ]:


data_train_all=data_train_all.drop(columns='Ticket')  ## we could have also used (inplace= True) and then we need not do data_train _all= data_train_all.drop(xxxxxx)


# In[ ]:


data_test_all=data_test_all.drop(columns='Ticket')


# #### Column: Cabin: The Nan values in the Cabin column means that the passangers didnt had the cabin.

# In[ ]:


f= lambda x: str(x)[0]


# In[ ]:


data_train_all.Cabin=data_train_all.Cabin.apply(f)


# In[ ]:


data_train_all['Cabin']=data_train_all['Cabin'].replace(['T'],'n')


# In[ ]:


data_test_all.Cabin=data_test_all.Cabin.apply(f)


# In[ ]:


data_train_all.Cabin.value_counts()


# In[ ]:


data_test_all.Cabin.value_counts()


# In[ ]:


data_train_all.groupby(['Cabin'])['Survived'].sum()


# ##### Column: Age: Missing Value Treatment

# In[ ]:


data_train_all.groupby(['Sex'])['Age'].median()


# In[ ]:


data_train_all.Age.median()


# In[ ]:


data_train_all.groupby(['Pclass']).Age.mean()


# In[ ]:


## we will be replaceing Age by median value


# In[ ]:


data_train_all.Age=data_train_all.Age.fillna(data_train_all.Age.median())


# In[ ]:


data_test_all.Age=data_test_all.Age.fillna(data_train_all.Age.median())


# #### Discretization of the Numeric columns : Age 
# There are two ways of doing it
1. Binning by quantiles.
2. Fixed interval Binning
# In[ ]:


# Binning by quantile
#data_train_all.Age=pd.qcut(data_train_all.Age, q=4, labels=False)


# In[ ]:


# Binning by fixed Interval


# In[ ]:


data_train_all.Age=pd.cut(data_train_all.Age, bins=[0,20,40,60,80,100],right=True, labels=False, retbins=0, include_lowest=1)


# In[ ]:


data_test_all.Age=pd.cut(data_test_all.Age, bins=[0,20,40,60,80,100],right=True, labels=False, retbins=0, include_lowest=1)


# In[ ]:


data_test_all.Age.hist()


# #### Discretization of the Numeric columns : Fare

# In[ ]:


data_train_all.Fare.min()


# In[ ]:


data_train_all.Fare=pd.cut(data_train_all.Fare, bins=[0,10,20,30,40,50,100,600],right=True, labels=False, retbins=0, include_lowest=1)


# In[ ]:


data_train_all.Fare.value_counts()


# In[ ]:


data_train_all.Fare.hist(bins=20)


# In[ ]:


data_test_all.Fare=pd.cut(data_test_all.Fare, bins=[0,10,20,30,40,50,100,600],right=True, labels=False, retbins=0, include_lowest=1)


# In[ ]:


data_test_all.Fare.hist(bins=20)


# In[ ]:


data_train_all.info()


# In[ ]:


data_test_all.info()


# In[ ]:


### Missing Value Treatment of Column Fare in Test Set


# In[ ]:


data_test_all.Fare.fillna(0,inplace=True)


# In[ ]:


data_test_all.Fare.value_counts()


# ### Missing Value Treatment of the Column Embark in the Training Set.

# In[ ]:


data_train_all.Embarked.value_counts()


# In[ ]:


data_train_all.Embarked.fillna('S', inplace=True)


# In[ ]:


data_train_all.drop(columns=['Name','PassengerId'],inplace=True)


# In[ ]:


data_train_all.head()


# In[ ]:


data_test_all.drop(columns=['Name','PassengerId'],inplace=True)


# In[ ]:


data_test_all.info()


# In[ ]:


type(data_train_all.Title[2])


# In[ ]:


## Create the dummy variables


# In[ ]:


data_train_all=pd.get_dummies(data_train_all,drop_first=False)  ## In case of categorical variable you dont need to drop one dummy variable.


# In[ ]:


data_test_all=pd.get_dummies(data_test_all,drop_first=False) 


# In[ ]:


data_train_all.head()


# In[ ]:


data_test_all.head()


# In[ ]:


data_train_all.info()


# In[ ]:


## Column Family


# In[ ]:


data_train_all['Family']=data_train_all['SibSp']+data_train_all['Parch']


# In[ ]:


data_test_all['Family']=data_test_all['SibSp']+data_test_all['Parch']


# In[ ]:


data_train_all.drop(columns=['SibSp','Parch'],inplace=True)
data_test_all.drop(columns=['SibSp','Parch'],inplace=True)


# In[ ]:


data_test_all.Family.value_counts()


# In[ ]:


data_test_all.info()


# In[ ]:


data_train_all.Fare=data_train_all.Fare.astype(float)


# In[ ]:


data_train_all.info()


# # Part 2
# # Implementation of GBM in H2O.ai for predicting 

# **Import H2O in python **

# In[ ]:


import h2o
h2o.init()


# **Convert Data from Pandas Data Frame to H2O data Frame**

# In[ ]:


data_train_h2o=h2o.H2OFrame(data_train_all)


# In[ ]:


data_test_h2o=h2o.H2OFrame(data_test_all)


# **Check the type of the data after conversion **

# In[ ]:


type(data_test_h2o)


# **NOTE : Always remember to convert the Target variable in training Data Set as Factor or categorical variable.
# Else the the H2O's GBM algorithm will not create a classification algorithm. **

# In[ ]:


data_train_h2o['Survived']=data_train_h2o['Survived'].asfactor()    ## Converting Target Variable as Factor


# **Import H2O's GBM **

# In[ ]:


from h2o.estimators.gbm import H2OGradientBoostingEstimator  # import gbm estimator


# **Instantiate the H2OGradientBoostingEstimator Class as model with some important parameters **

# In[ ]:


model = H2OGradientBoostingEstimator(## more trees is better if the learning rate is small enough 
  ## here, use "more than enough" trees - we have early stopping
  ntrees = 10000,                                                            

  ## smaller learning rate is better (this is a good value for most datasets, but see below for annealing)
  learn_rate = 0.01,                                                         

  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 

  ## sample 80% of rows per tree
  sample_rate = 0.8,                                                       

  ## sample 80% of columns per split
  col_sample_rate = 0.8,                                                   

  ## fix a random number generator seed for reproducibility
  seed = 1234,                                                             

  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10, nfolds=5, max_depth=3)   ## Instantiating the class


# **Train the model.**
# 
# Note1: We are using train not fit as in scikit learn
# 
# Note2: Since we are using N fold cross validation, and the size of data set being small; I am not spliting the the training set,
#  instead, using the whole training set.
#     
# Note3: Will check the Bias and the Variance of the model from our N fold Cross Validation's mean and Std Dev of accuracy score.

# In[ ]:


model.train(x=data_train_h2o.names[1:],y=data_train_h2o.names[0], training_frame=data_train_h2o, model_id="GBM_Titanic",
            validation_frame=data_train_h2o)

Have a look at all the methods of the object 'model'
# In[ ]:


dir(model)


# **5 fold Cross Validation Score**
# 
# 
# Note: Important Metrics to be seen here are as follows:
# 
#      1. Accuracy Metric   : the mean accuracy of the model is 83.8% and the std dev is 2.3%
#      2. auc Metric        : Mean AUC is 87% 
#      
#      
#      From these metrics we can say that our model is fairly ok.
#      We can further improve it by parameter tuning.

# In[ ]:


model.cross_validation_metrics_summary()


# **Check Model Parameters**

# In[ ]:


model.params


# **Predict the Target variable for the Test Set**

# In[ ]:


f=model.predict(test_data=data_test_h2o)


# **Convert the H2O dataframe to pandas DataFrame**

# In[ ]:


f=f.as_data_frame()             ## Converting Predicted Results to Python Dataframe


# **Create the Submission file for the Kaggle.**

# In[ ]:


submission_H2O = pd.DataFrame({'PassengerId':data_test_all_pred['PassengerId'],'Survived':f['predict']})


# In[ ]:


## submission_H2O.to_csv('D:/Titanic/Titanic Predictions_H2O.csv',index=False)


# In[ ]:




