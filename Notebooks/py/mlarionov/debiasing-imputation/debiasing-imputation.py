#!/usr/bin/env python
# coding: utf-8

# # Debiasing Imputation #
# 
# This notebook is about dealing with missing data that does not increase bias (gender bias, race bias, etc.), or even potentially reduce it
# 
# 

# ## Problem statement ##
# Most common way to handle missing data is to drop them. The second most common way is to replace the missing data with the most likely value. For the categorical features it is the most frequent value. For the numerical features it is the mean. `scikit-learn` has a class available for this: [SimpleImputer](http://scikit-learn.org/dev/modules/generated/sklearn.impute.SimpleImputer.html). The problem with this approach is that even though it preserves mean, but it reduces the standard deviation, sometimes very significantly. To demonstrate this, let's consider a simple array, then remove half of the values and replace them with mean, and see what happens with STD:

# In[ ]:


import numpy as np
from scipy.stats import norm, multinomial
original_data = norm.rvs(loc=1.0, scale=0.5, size=1000, random_state=1386)
original_data[:20]


# In[ ]:


#Now replace every other element with the mean 1.0
missing_elements = np.asarray([0,1]*500)
updated_data = original_data * (1-missing_elements) + missing_elements
updated_data[:20]


# In[ ]:


#Now, let's get mean and std of the new distribution:
mean, std = norm.fit(updated_data)
print(f'Mean: {mean}, std: {std}')


# As you see, even though the mean is the same, the standard deviation is much less. While the imputation of data this way increases the performance of the model, it also amplifies the bias that already exists in the data. In order to prevent amplification of the bias, we have to replace the missing values with a sample from the normal distribution with the same mean and standard deviation. For categorical features it would be a multinomial distribution.
# 
# For debiasing we can try to increase the standard deviation of the distribution from which we sample data for numerical features, and a similar transformation for the multinomial distribution. 
# 
# In this notebook I suggest two classes for the numerical and categorical features respectively.

# ## Proposed solution ##

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
import numpy.ma as ma
from sklearn.utils.validation import check_is_fitted
class NumericalUnbiasingImputer(BaseEstimator, TransformerMixin):
    """Un-biasing imputation transformer for completing missing values.
        Parameters
        ----------
        std_scaling_factor : number
            We will multiply std by this factor to increase or decrease bias
    """
    def __init__(self, std_scaling_factor=1, random_state=7294):
        self.std_scaling_factor = std_scaling_factor
        self.random_state = random_state

        
    def fit(self, X: np.ndarray, y=None):
        """Fit the imputer on X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        Returns
        -------
        self : NumericalUnbiasingImputer
        """
        if len(X.shape) < 2:
            X = X.reshape(-1,1)
        mask = np.isnan(X)
        masked_X = ma.masked_array(X, mask=mask)

        mean_masked = np.ma.mean(masked_X, axis=0)
        std_masked = np.ma.std(masked_X, axis=0)
        mean = np.ma.getdata(mean_masked)
        std = np.ma.getdata(std_masked)
        mean[np.ma.getmask(mean_masked)] = np.nan
        std[np.ma.getmask(std_masked)] = np.nan
        self.mean_ = mean
        self.std_ = std * self.std_scaling_factor

        return self
    
     
    def transform(self, X):
        """Impute all missing values in X.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The input data to complete.
        """
        check_is_fitted(self, ['mean_', 'std_'])

        if len(X.shape) < 2:
            X = X.reshape(-1,1)
        mask = np.isnan(X)
        n_missing = np.sum(mask, axis=0)
        
        def transform_single(index):
            col = X[:,index].copy()
            mask_col = mask[:, index]
            sample = np.asarray(norm.rvs(loc=self.mean_[index], scale=self.std_[index], 
                                         size=col.shape[0], random_state=self.random_state))
            col[mask_col] = sample[mask_col]
            return col
            
        
        Xnew = np.vstack([transform_single(index) for index,_ in enumerate(n_missing)]).T
        

        return Xnew
    


# In[ ]:


imputer = NumericalUnbiasingImputer()
missing_indicator = missing_elements.copy().astype(np.float16)
missing_indicator[missing_indicator == 1] = np.nan
data_with_missing_values = original_data + missing_indicator
data_with_missing_values = np.vstack([data_with_missing_values, original_data*5]).T
imputer.fit(data_with_missing_values)
transformed = imputer.transform(data_with_missing_values)
print(transformed[:20,:])
transformed.shape


# In[ ]:


#Let's see how it is different from the original array:
new_mean, new_std = norm.fit(transformed[:,0])
print(f'Mean: {new_mean}, Std: {new_std}')


# Some difference in the standard deviation can be explained, because we fitted the model on the incomplete data.
# 
# Now we need to do the same for the categorical features

# In the multinomial distribution there is no single parameter responsible for standard deviation. However we can observe, that scaling the standard deviation of the normal distribution is equivalent to scaling `x`. If we do a similar transformation in the multinomial distribution, this would be equivalent to raising the parameters to the power of $\frac{1}{s}$, where $s$ is the scaling factor

# In[ ]:


import pandas as pd
class CategoricalUnbiasingImputer(BaseEstimator, TransformerMixin):
    """Un-biasing imputation transformer for completing missing values.
        Parameters
        ----------
        std_scaling_factor : number
            We will multiply std by this factor to increase or decrease bias
    """
    def __init__(self, scaling_factor=1, random_state=7294):
        self.scaling_factor = scaling_factor
        self.random_state = random_state

        
    def fit(self, X: np.ndarray, y=None):
        """Fit the imputer on X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        Returns
        -------
        self : NumericalUnbiasingImputer
        """
        if len(X.shape) < 2:
            X = X.reshape(-1,1)

        def fit_column(column):
            mask = pd.isnull(column)
            column = column[~mask]
            unique_values, counts = np.unique(column.data, return_counts=True)
            total = sum(counts)
            probabilities = np.array([(count/total)**(1/self.scaling_factor) 
                    for count in counts])
            total_probability = sum(probabilities)
            probabilities /= total_probability
            return unique_values, probabilities


        self.statistics_ = [fit_column(X[:,column]) for column in range(X.shape[1])]

        return self
    
     
    def transform(self, X):
        """Impute all missing values in X.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The input data to complete.
        """
        check_is_fitted(self, ['statistics_'])

        if len(X.shape) < 2:
            X = X.reshape(-1,1)
        
        def transform_single(index):
            column = X[:,index].copy()
            mask = pd.isnull(column)
            values, probabilities = self.statistics_[index]

            sample = np.argmax(multinomial.rvs(p=probabilities, n=1,
                                         size=mask.sum(), random_state=self.random_state), axis=1)
            column[mask] = np.vectorize(lambda pick: values[pick])(sample);
            return column
            
        
        Xnew = np.vstack([transform_single(index) for index in range(len(self.statistics_))]).T
        

        return Xnew


# In[ ]:


names = np.array(['one', None, 'two', 'three', 'four', 'one', None, 'one', 'two'])
names = names.reshape(-1,1)
cat_imp = CategoricalUnbiasingImputer(random_state=121)
cat_imp.fit(names)
print(cat_imp.statistics_)
imputed = cat_imp.transform(names)
imputed


# Now we test our utilities on the scikit-learn datasets. Let's try our approach on the famous titanic dataset. Let's see if any of the columns contain nans

# In[ ]:


titanic = pd.read_csv("../input/train.csv")
titanic.isna().sum(axis=0)


# In[ ]:


titanic.info()


# We see that Age is numeric, and Cabin is an object feature. Let's update them using our imputers

# In[ ]:


n_imputer = NumericalUnbiasingImputer()
titanic.Age = n_imputer.fit(titanic.Age.values).transform(titanic.Age.values)


# In[ ]:


c_imputer = CategoricalUnbiasingImputer()
titanic.Cabin = c_imputer.fit(titanic.Cabin.values).transform(titanic.Cabin.values)


# In[ ]:


#Let's see how it transformed Age
titanic.Age.head(20)


# OK, negative age is a bit too much... But keep in mind that the purpose is not to reconstruct the data, but to avoid amplifying bias in the machine learning model. Let's make sure Age and Cabin now is not null

# In[ ]:


print(titanic.Age.isnull().sum())
print(titanic.Cabin.isnull().sum())


# In[ ]:


#Unique values of the Cabin
titanic.Cabin.unique()


# # Next Steps #
# 
# Now we need to find a dataset that is known to have a gender or race bias, and demonstrate, that with this technique we can avoid amplifying bias, and maybe even decrease the bias by applying a scaling factor

# In[ ]:




