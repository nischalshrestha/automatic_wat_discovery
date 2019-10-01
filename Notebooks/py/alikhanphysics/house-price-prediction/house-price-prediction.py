#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# The data set for houses in Melbourne:

# In[ ]:


melb_data_filepath = '../input/melbourne-housing-data/melb_data.csv'
melb_data = pd.read_csv(melb_data_filepath)
melb_data.describe()


# The variables describing the houses are:

# In[ ]:


melb_data.columns


# However, not every house in the dataset has entries for all the above variables. We take the subset of the data set which *does* have complete information:

# In[ ]:


melb_data_subset = melb_data.dropna(axis=0)
melb_data_subset.describe()


# We wish to be able to predict the price of houses, so we define 'Price' as the output variable 'y' .

# In[ ]:


y = melb_data_subset.Price


#     We define features (the input variables that we will use to predict the Price) as 'X': 

# In[ ]:


melb_features = ['Rooms', 'Bedroom2', 'Bathroom', 'Landsize', 'YearBuilt']
X = melb_data_subset[melb_features]


# Let's look at the top few rows of the data:

# In[ ]:


X.head()


# We now use ski-kit learn to fit the data.

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
#Define the model and fixing the random state, so that the same model is produced on each run.
melbourne_model = DecisionTreeRegressor(random_state=1)
#Now we fit the data
melbourne_model.fit(X,y)


# Let's apply this model to the first few rows of the input data as a check. I'll also define a variable 'W' the same as 'X' but also including the price, so that we can compare the known prices with predicted prices:

# In[ ]:


melb_features_price = ['Rooms', 'Bedroom2', 'Bathroom', 'Landsize', 'YearBuilt', 'Price']
W = melb_data_subset[melb_features_price]
print(W.head())
print('The predicted prices are:')
print(melbourne_model.predict(X.head()))

