#!/usr/bin/env python
# coding: utf-8

# # Minimalistic Titanic in Python LightGBM
# 
# This notebook investigates working with the Titanic set using only the **Sex, Embarked and Pclass** features for prediction. 
# 
# This is a supporting kernel for this other [**kernel**](https://www.kaggle.com/pliptor/how-am-i-doing-with-my-score). 
# 
# You can find a [**similar kernel in R using XGBoost**](https://www.kaggle.com/pliptor/minimalistic-xgb/notebook). We will use LightGBM. 
# 
# 1. [Reading data](#read_data)
# 2. [Label Encoding](#label_enc)
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

feature_names = ['Sex','Embarked','Pclass']

# load data sets 
train = pd.read_csv('../input/train.csv', usecols =['Survived','PassengerId'] + feature_names)
test  = pd.read_csv('../input/test.csv', usecols =['PassengerId'] + feature_names )

# combine train and test for joint processing 
test['Survived'] = np.nan
comb = pd.concat([ train, test ])
comb.head()


# Let's also fix two missing values in Embarked

# In[ ]:


print('Number of missing Embarked values ',comb['Embarked'].isnull().sum())
comb['Embarked'] = comb['Embarked'].fillna('S')
comb['Embarked'].unique()


# # 2) Label Encoding <a class="anchor" id="label_enc"></a>
# 
# The following steps encode non-numeric features to numeric ones. 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
comb.Embarked = le.fit_transform(comb.Embarked)
comb.Sex      = le.fit_transform(comb.Sex)
comb.head()


# Now we split back comb2 as we are done pre-processing

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
# We will use a LightGBM for the model and use GridSearchCV to tune it.

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


from lightgbm import LGBMClassifier
model = LGBMClassifier()
param_grid = {'n_estimators':[20],'max_depth':[2,3,4]} 
grs = GridSearchCV(model, param_grid=param_grid, cv = 10, n_jobs=4, return_train_score = False)
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
sub.to_csv('minimalistic.csv', index = False, float_format='%1d')
sub.head()


# # Conclusions <a class="anchor" id="conclusions"></a>
# 
# We tackled the Titanic problem using only the **Sex, Embarked and Pclass** features. You should get a public score of **0.77990**. It is argued in [**this kernel**](https://www.kaggle.com/pliptor/how-am-i-doing-with-my-score) that the public score is about 2% lower than cross validation estimates for unseen data when the model heavily relies on the Gender and Pclass features. This kernel supports the findings. The cross validation for unseen data is 0.8114.
# 
# Thanks for reading this kernel and let me know if you have any questions or comments!
