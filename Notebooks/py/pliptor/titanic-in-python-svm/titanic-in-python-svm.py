#!/usr/bin/env python
# coding: utf-8

# # Titanic in Python SVM
# 
# This notebook investigates working with the Titanic set using only the **Age, Sex, Pclass, and Embarked** features for prediction. 
# 
# This is a supporting kernel for this other [**kernel**](https://www.kaggle.com/pliptor/how-am-i-doing-with-my-score). The goal is not the score maximization but to investigate the problem using a reduced number of features.
# 
# 1. [Reading data](#read_data)
# 2. [Data preparation](#preparation)
# 3. [Modeling](#modeling)
# 4. [Predicting and creating a submission file](#submission)
# 
# [Conclusions](#conclusions)
# 
# 

# # 1) Reading data <a class="anchor" id="read_data"></a>
# 
# We will only read the relevant columns of the data. It will keep the data clean and prevent any accidental leakage of other features into our setup.

# In[ ]:


import pandas as pd
import numpy  as np

np.random.seed(2018)

feature_names = ['Age','Pclass','Embarked','Sex']

# load data sets 
train = pd.read_csv('../input/train.csv', usecols =['Survived','PassengerId'] + feature_names)
test  = pd.read_csv('../input/test.csv',  usecols =['PassengerId'] + feature_names )

# combine train and test for joint processing 
test['Survived'] = np.nan
comb = pd.concat([ train, test ])
comb.head()


# # 2) Data preparation <a class="anchor" id="preparation"></a>  
# 
# Let's first fix two missing values in Embarked. We note here the final result is not impacted with a choice for the filling value ('S','C','Q'). This is probably because altering two rows is not sufficient to change the model's statistics in a significant manner.

# In[ ]:


print('Number of missing Embarked values ',comb['Embarked'].isnull().sum())
comb['Embarked'] = comb['Embarked'].fillna('S')
comb['Embarked'].unique()


# ## 2.1) Age
# 
# We follow the same strategy in the [**divide and conquer kernel**](https://www.kaggle.com/pliptor/divide-and-conquer-0-82296). The idea is to discard most of the Age data and keep just what seems to matter the most (young folks had priority). We also create an indicator flag for those that had no Age data available.

# In[ ]:


comb['NoAge'] = comb['Age'] == np.NAN
comb['Age'] =  comb['Age'].fillna(-1)
comb['Age'].hist(bins=100)
comb['Minor'] = (comb['Age']<14.0)&(comb['Age']>=0)


# ## 2.2) One-hot and Label encode
# 
# While we could use sklearn to perform one-hot and label encoding, we will do these tasks manually since there are only a couple of features to be treated.

# In[ ]:


# one-hot encode Pclass
comb['P1'] = comb['Pclass'] == 1 
comb['P2'] = comb['Pclass'] == 2
comb['P3'] = comb['Pclass'] == 3

# one-hot encode Embarked
comb['ES'] = comb['Embarked'] == 'S' 
comb['EQ'] = comb['Embarked'] == 'Q'
comb['EC'] = comb['Embarked'] == 'C'


# In[ ]:


# Label encode Sex
comb['Sex'] = comb['Sex'].map({'male':0,'female':1})

# drop Pclass, Embarked and Age features
comb = comb.drop(columns=['Pclass','Embarked','Age'])
comb.head()


# Now we split back comb as we are done pre-processing.

# In[ ]:


df_train = comb.loc[comb['Survived'].isin([np.nan]) == False]
df_test  = comb.loc[comb['Survived'].isin([np.nan]) == True]

print(df_train.shape)
df_train.head()


# In[ ]:


print(df_test.shape)
df_test.head()


# We are now ready for modeling!

# # 3) Modeling <a class="anchor" id="modeling"></a>
# 
# We will use support vector machine (SVM) in classification mode for the model and use GridSearchCV to tune it.

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


feature_names = ['Sex','P1','P2','P3','EQ','ES','EC','NoAge','Minor']

from sklearn.svm import SVC
model = SVC()
param_grid = {'C':[1,2,5,10,20,50]} 
grs = GridSearchCV(model, param_grid=param_grid, cv = 10, n_jobs=1, return_train_score = False)
grs.fit(np.array(df_train[feature_names]), np.array(df_train['Survived']))


# Now that the tuning is completed, we print the best parameter found and also the estimated accuracy for the unseen data.  

# In[ ]:


print("Best parameters " + str(grs.best_params_))
gpd = pd.DataFrame(grs.cv_results_)
print("Estimated accuracy of this model for unseen data:{0:1.4f}".format(gpd['mean_test_score'][grs.best_index_]))


# # 4) Predicting and creating a submission file<a class="anchor" id="submission"></a>

# In[ ]:


pred = grs.predict(np.array(df_test[feature_names]))

sub = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':pred})
sub.to_csv('AgeSexPclassEmbarked.csv', index = False, float_format='%1d')
sub.head()


# # Conclusions <a class="anchor" id="conclusions"></a>
# 
# We tackled the Titanic problem using only the **Age, Sex, Embarked and Pclass** features. You should get a public score of **0.78947**. It is argued in [**this kernel**](https://www.kaggle.com/pliptor/how-am-i-doing-with-my-score) that the public score is about 2% lower than cross validation estimates for unseen data when the model heavily relies on the Gender and Pclass features. This kernel supports the findings. The cross validation for unseen data is 0.8249.
# 
# We also note this kernel is similar to [**this kernel**](https://www.kaggle.com/pliptor/minimalistic-titanic-in-python-lightgbm) but with the Age feature added. The addition of the Age feature improved the public score by about 0.01 from **0.77990** to **0.78947**.
# 
# Thanks for reading this kernel and let me know if you have any questions or comments!
