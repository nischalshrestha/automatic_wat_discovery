#!/usr/bin/env python
# coding: utf-8

# # Titanic 
# 
# Goal: predict survival on the Titanic  
# 
# Here we are looking into how to apply Logistic Regression to the Titanic dataset.

# # 1. Collect and understand the data

# In[30]:


import pandas as pd


# In[31]:


# get titanic training file as a DataFrame
titanic = pd.read_csv("../input/train.csv")


# In[32]:


titanic.shape


# In[33]:


# preview the data
titanic.head()


# Variable Description
# ---
# Survived: Survived (1) or died (0);  this is the target variable  
# Pclass: Passenger's class (1st, 2nd or 3rd class)    
# Name: Passenger's name  
# Sex: Passenger's sex  
# Age: Passenger's age  
# SibSp: Number of siblings/spouses aboard  
# Parch: Number of parents/children aboard  
# Ticket: Ticket number  
# Fare: Fare  
# Cabin: Cabin  
# Embarked: Port of embarkation

# In[34]:


titanic.describe()


# Not all features are numeric:

# In[35]:


titanic.info()


# # 2. Process the Data
# Categorical variables need to be transformed into numeric variables

# ### Transform the embarkment port

# There are three ports: C = Cherbourg, Q = Queenstown, S = Southampton

# In[36]:


ports = pd.get_dummies(titanic.Embarked , prefix='Embarked')
ports.head()


# Now the feature Embarked (a category) has been trasformed into 3 binary features, e.g. Embarked_C = 0 not embarked in Cherbourg, 1 = embarked in Cherbourg.  
# Finally, the 3 new binary features substitute the original one in the data frame:

# In[37]:


titanic = titanic.join(ports)
titanic.drop(['Embarked'], axis=1, inplace=True) # then drop the original column


# ### Transform the gender feature
# This transformation is easier, being already a binary classification (male or female, this was 1912).
# It doesn't need to create separate dummy categories, a mapping will be enough:

# In[38]:


titanic.Sex = titanic.Sex.map({'male':0, 'female':1})


# ## Extract the target variable
# Create an X dataframe with the input features and an y series with the target (Survived)

# In[39]:


y = titanic.Survived.copy() # copy “y” column values out


# In[40]:


X = titanic.drop(['Survived'], axis=1) # then, drop y column


# ### Drop not so important features
# For the first model, we ignore some categorical features which will not add too much of a signal.

# In[41]:


X.drop(['Cabin'], axis=1, inplace=True) 


# In[42]:


X.drop(['Ticket'], axis=1, inplace=True) 


# In[43]:


X.drop(['Name'], axis=1, inplace=True) 
X.drop(['PassengerId'], axis=1, inplace=True)


# In[44]:


X.info()


# All features are now numeric, ready for regression.  
# But we have still a couple of processing to do.

# ## Check if there are any missing values

# In[45]:


X.isnull().values.any()


# In[46]:


#X[pd.isnull(X).any(axis=1)]  # check which rows have NaNs


# True, there are missing values in the data (NaN) and a quick look at the data reveals that they are all in the Age feature.  
# One possibility could be to remove the feature, another one is to fill the missing value with a fixed number or the average age.

# In[47]:


X.Age.fillna(X.Age.mean(), inplace=True)  # replace NaN with average age


# In[48]:


X.isnull().values.any()


# Now all missing values have been removed.  
# The logistic regression would otherwise not work with missing values.

# ## Split the dataset into training and validation
# 
# The **training** set will be used to build the machine learning models. The model will be based on the features like passengers’ gender and class but also on the known survived flag.
# 
# The **validation** set should be used to see how well the model performs on unseen data. For each passenger in the test set, I use the model trained to predict whether or not they survived the sinking of the Titanic, then will be compared with the actual survival flag.

# In[49]:


from sklearn.model_selection import train_test_split
  # 80 % go into the training test, 20% in the validation test
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)


# # 3. Modelling

# ## Get a baseline
# A baseline is always useful to see if the model trained behaves significantly better than an easy to obtain baseline, such as a random guess or a simple heuristic like all and only female passengers survived. In this case, after quickly looking at the training dataset - where the survival outcome is present - I am going to use the following:
# 

# In[50]:


def simple_heuristic(titanicDF):
    '''
    predict whether or not the passngers survived or perished.
    Here's the algorithm, predict the passenger survived:
    1) If the passenger is female or
    2) if his socioeconomic status is high AND if the passenger is under 18
    '''

    predictions = [] # a list
    
    for passenger_index, passenger in titanicDF.iterrows():
          
        if passenger['Sex'] == 1:
                    # female
            predictions.append(1)  # survived
        elif passenger['Age'] < 18 and passenger['Pclass'] == 1:
                    # male but minor and rich
            predictions.append(1)  # survived
        else:
            predictions.append(0) # everyone else perished

    return predictions


# Let's see how this simple algorithm will behave on the validation dataset and we will keep that number as our baseline:

# In[51]:


simplePredictions = simple_heuristic(X_valid)
correct = sum(simplePredictions == y_valid)
print ("Baseline: ", correct/len(y_valid))


# Baseline: a simple algorithm predicts correctly 73% of validation cases.  
# Now let's see if the model can do better.

# ##  Logistic Regression

# Will use a simple logistic regression, that takes all the features in X and creates a regression line.
# This is done using the LogisticRegression module in SciKitLearn.

# In[52]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[53]:


model.fit(X_train, y_train)


# # 4. Evaluate the model

# In[54]:


model.score(X_train, y_train)


# In[55]:


model.score(X_valid, y_valid)


# Two things:
# - the score on the training set is much better than on the validation set, an indication that could be overfitting and not being a general model, e.g. for all ship sinks.
# - the score on the validation set is better than the baseline, so it adds some value at a minimal cost (the logistic regression is not computationally expensive, at least not for smaller datasets).

# An advantage of logistic regression (e.g. against a neural network) is that it's easily interpretable.  It can be written as a math formula:

# In[27]:


model.intercept_ # the fitted intercept


# In[28]:


model.coef_  # the fitted coefficients


# Which means that the formula is:  
# $$ \boldsymbol P(survive) = \frac{1}{1+e^{-logit}} $$  
#   
# where the logit is:  
#   
# $$ logit = \boldsymbol{\beta_{0} + \beta_{1}\cdot x_{1} + ... + \beta_{n}\cdot x_{n}}$$ 
#   
# where $\beta_{0}$ is the model intercept and the other beta parameters are the model coefficients from above, each multiplied for the related feature:  
#   
# $$ logit = \boldsymbol{1.4224 - 0.9319 * Pclass + ... + 0.2228 * Embarked_S}$$ 

# # 5. Iterate on the model
# The model could be improved, for example transforming the excluded features above or creating new ones (e.g. I could extract titles from the names which could be another indication of the socio-economic status).

# The correlation matrix may give us a understanding of which variables are important

# In[56]:


titanic.corr()


# # 6. Deploy to Kaggle

# The resulting score is **0.75119**  
# Note that the score on the validation set has been a good predictor!

# In[ ]:




