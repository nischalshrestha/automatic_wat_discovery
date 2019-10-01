#!/usr/bin/env python
# coding: utf-8

# # TPOT Tutorial
# **TPOT** is a Python **Automated** Machine Learning tool that optimizes machine learning pipelines using **genetic programming**.
# This notebook is from the TPOT tutorials examples. You can find all the examples [here](https://epistasislab.github.io/tpot/examples/)

# In[ ]:


from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np


# In[ ]:


ls ../input


# In[ ]:


titanic = pd.read_csv('../input/train.csv')
titanic.head(5)


# # Data Exploration

# In[ ]:


titanic.groupby('Sex').Survived.value_counts()


# In[ ]:


titanic.groupby(['Pclass','Sex']).Survived.value_counts()


# In[ ]:


id = pd.crosstab([titanic.Pclass, titanic.Sex], titanic.Survived.astype(float))
id.div(id.sum(1).astype(float), 0)


# # Data Munging

# In[ ]:


titanic.rename(columns={'Survived': 'class'}, inplace=True)


# At present, TPOT requires all the data to be in numerical format. As we can see below, our data set has 5 categorical variables which contain non-numerical values: Name, Sex, Ticket, Cabin and Embarked.

# In[ ]:


titanic.dtypes


# We then check the number of levels that each of the five categorical variables have.

# In[ ]:


for cat in ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']:
    print("Number of levels in category '{0}': \b {1:2d} ".format(cat, titanic[cat].unique().size))


# As we can see, Sex and Embarked have few levels. Let's find out what they are.

# In[ ]:


for cat in ['Sex', 'Embarked']:
    print("Levels for category '{0}': {1}".format(cat, titanic[cat].unique()))


# We then code these levels manually into numerical values. For nan i.e. the missing values, we simply replace them with a placeholder value (-999). In fact, we perform this replacement for the entire data set.

# In[ ]:


titanic['Sex'] = titanic['Sex'].map({'male':0,'female':1})
titanic['Embarked'] = titanic['Embarked'].map({'S':0,'C':1,'Q':2})


# In[ ]:


titanic = titanic.fillna(-999)
pd.isnull(titanic).any()


# Since Name and Ticket have so many levels, we drop them from our analysis for the sake of simplicity. For Cabin, we encode the levels as digits using Scikit-learn's MultiLabelBinarizer and treat them as new features.

# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
CabinTrans = mlb.fit_transform([{str(val)} for val in titanic['Cabin'].values])


# In[ ]:


CabinTrans


# Drop the unused features from the dataset.

# In[ ]:


titanic_new = titanic.drop(['Name','Ticket','Cabin','class'], axis=1)


# In[ ]:


assert (len(titanic['Cabin'].unique()) == len(mlb.classes_)), "Not Equal" #check correct encoding done


# We then add the encoded features to form the final dataset to be used with TPOT.

# In[ ]:


titanic_new = np.hstack((titanic_new.values,CabinTrans))


# In[ ]:


np.isnan(titanic_new).any()


# Keeping in mind that the final dataset is in the form of a numpy array, we can check the number of features in the final dataset as follows.

# In[ ]:


titanic_new[0].size


# Finally we store the class labels, which we need to predict, in a separate variable.

# In[ ]:


titanic_class = titanic['class'].values


# # Data Analysis using TPOT

# To begin our analysis, we need to divide our training data into training and validation sets. The validation set is just to give us an idea of the test set error. The model selection and tuning is entirely taken care of by TPOT, so if we want to, we can skip creating this validation set.

# In[ ]:


training_indices, validation_indices = training_indices, testing_indices = train_test_split(titanic.index, stratify = titanic_class, train_size=0.75, test_size=0.25)
training_indices.size, validation_indices.size


# 
# 
# After that, we proceed to calling the fit, score and export functions on our training dataset. To get a better idea of how these functions work, refer the TPOT documentation [here](http://epistasislab.github.io/tpot/api/).
# 
# 
# 

# In[ ]:


tpot = TPOTClassifier(generations=2,verbosity=2,population_size=40,n_jobs=-1)
tpot.fit(titanic_new[training_indices], titanic_class[training_indices])


# In[ ]:


tpot.score(titanic_new[validation_indices], titanic.loc[validation_indices, 'class'].values)


# In[ ]:


tpot.export('tpot_titanic_pipeline.py')


# Let's have a look at the generated code. As we can see, the random forest classifier performed the best on the given dataset out of all the other models that TPOT currently evaluates on. If we ran TPOT for more generations, then the score should improve further.

# In[ ]:


get_ipython().magic(u'load tpot_titanic_pipeline.py')


# ## Make predictions on the submission data

# In[ ]:


# Read in the submission dataset
titanic_sub = pd.read_csv('../input/test.csv')
titanic_sub.describe()


# The most important step here is to check for new levels in the categorical variables of the submission dataset that are absent in the training set. We identify them and set them to our placeholder value of '-999', i.e., we treat them as missing values. This ensures training consistency, as otherwise the model does not know what to do with the new levels in the submission dataset.

# In[ ]:


for var in ['Cabin']: #,'Name','Ticket']:
    new = list(set(titanic_sub[var]) - set(titanic[var]))
    titanic_sub.loc[titanic_sub[var].isin(new), var] = -999


# We then carry out the data munging steps as done earlier for the training dataset.

# In[ ]:


titanic_sub['Sex'] = titanic_sub['Sex'].map({'male':0,'female':1})
titanic_sub['Embarked'] = titanic_sub['Embarked'].map({'S':0,'C':1,'Q':2})


# In[ ]:


titanic_sub = titanic_sub.fillna(-999)
pd.isnull(titanic_sub).any()


# While calling MultiLabelBinarizer for the submission data set, we first fit on the training set again to learn the levels and then transform the submission dataset values. This further ensures that only those levels that were present in the training dataset are transformed. If new levels are still found in the submission dataset then it will return an error and we need to go back and check our earlier step of replacing new levels with the placeholder value.

# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
SubCabinTrans = mlb.fit([{str(val)} for val in titanic['Cabin'].values]).transform([{str(val)} for val in titanic_sub['Cabin'].values])
titanic_sub = titanic_sub.drop(['Name','Ticket','Cabin'], axis=1)


# In[ ]:


# Form the new submission data set
titanic_sub_new = np.hstack((titanic_sub.values,SubCabinTrans))


# In[ ]:


np.any(np.isnan(titanic_sub_new))


# In[ ]:


# Ensure equal number of features in both the final training and submission dataset
assert (titanic_new.shape[1] == titanic_sub_new.shape[1]), "Not Equal"


# In[ ]:


# Generate the predictions
submission = tpot.predict(titanic_sub_new)


# In[ ]:


# Create the submission file
final = pd.DataFrame({'PassengerId': titanic_sub['PassengerId'], 'Survived': submission})
final.to_csv('submission.csv', index = False)


# In[ ]:


final.shape


# There we go! We have successfully generated the predictions for the 418 data points in the submission dataset, and we're good to go ahead to submit these predictions on Kaggle.
