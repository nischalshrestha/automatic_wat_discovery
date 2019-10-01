#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

main_file_path = '../input/train.csv' # this is the path to the Iowa data
training_data = pd.read_csv(main_file_path)

print('Training data loaded')

# Any results you write to the current directory are saved as output.


# In[ ]:


training_data


# In[ ]:


missing_val_count_by_column = (training_data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# # Data Preprocessing
# ### Basic set-up:
# 
# (excluding object variables for now)

# In[ ]:


titanic_target = training_data['Survived']
titanic_predictors = training_data.drop(['Survived'], axis=1)
titanic_numeric_predictors = titanic_predictors.select_dtypes(exclude=['object'])


# ## Deciding our Imputation Method
# 
# Lets determine which imputation method has the lowest MAE:
# 
# Method 1: `SimpleImputer()` - The default behavior fills in the mean value for imputation.
# 
# Method 2: `SimpleImputer()` with added columns - Fills in the mean value for and adds columns indicating which values were added. 

# In[ ]:


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
titanic_numeric_predictors_imputed = my_imputer.fit_transform(titanic_numeric_predictors)


# In[ ]:


print(titanic_numeric_predictors_imputed)

