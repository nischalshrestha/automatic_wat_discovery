#!/usr/bin/env python
# coding: utf-8

# In[158]:


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import Set


# ## Here We Are: Starting on the Titanic Practice Comp
# 
# This will serve as a sandbox notebook for my getting familiar with Kaggle, competitions, and working with a wide variety of datasets.
# 
# Please ignore the following - it is, indeed, scratchspace.
# 
# $\sum^{k/t}$ and $\underset{k\times l}{A}$

# In[159]:


# This is scratchspace where Python code will go.

pd.DataFrame(columns=['test', 'test2']).describe()

# Scratchspace ends here.
# -----------------------------------------------


# In[160]:


# Input data is in "../input". Here we have the Titanic data files.
print(os.listdir("../input"))
PATH = '../input'
file_train = f'{PATH}/train.csv'
file_test = f'{PATH}/test.csv'
file_gender_submission = f'{PATH}/gender_submission.csv'


# In[161]:


data_train_all = pd.read_csv(file_train, sep=',')
data_test = pd.read_csv(file_test, sep=',')
data_test.head()


# In[162]:


# Verify the contents of the files are what we expect.
print(f'{data_train_all.shape} for training\n{data_test.shape} for testing')


# Now that we have the data loaded in, let's...
# 
# ## Process The Data
# 
# We'll start by separating out features as categorical or continuous, omitting those that are not likely to be useful right off the bat.

# In[163]:


features_categorical = [
    'Survived',
    'PassengerId',
    'Pclass',  # Socioeconomic status, generally.
    'Sex'
]

features_continuous = [
    'Age',
    'SibSp', # Sibling or Spouse
    'Parch' # Parents or Children
]


# In[164]:


# When we go to create the data_test object below we want to get all features except 'Survived' since that won't be present outside the training data.
print(features_categorical[1:])


# In[165]:


# Drop the columns we don't plan to use.
data_train_all = data_train_all[features_continuous + features_categorical]
data_test = data_test[features_continuous + features_categorical[1:]]
data_test.head()


# In[166]:


# Let's have a look through the describe() results for the numerical columns.
data_test.describe()


# In[167]:


#data_train_all.get(features_categorical).head()
#data_train_all.get(features_continuous).head()


# We have columns with missing data. Let's find where they are, and determine how best to handle them. For numerical values, we may wish to replace them with an average over the rest of the column. For categorical values, we will likely create a separate class for them, unless we want to try to guess (based on similar passengers' data) what would make most sense for them to have in that particular field.

# In[168]:


for featurename in (features_categorical + features_continuous):
    if featurename not in ['PassengerId']:
        print(f'{featurename} Unique Values:')
        print(data_train_all[featurename].unique())
    #if featurename in ['Age']:
    #    for age in data_train[featurename]:
    #        if 'nan' in str(age).lower():
    #            print(age)


# From this output we can tell that Age contains `nan` values. All other columns (except for `PassengerId` which we intentionally skipped since it's huge to read through) appear to have all sane values.
# 
# Although my gut tells me this is probably not a solid idea, I'll replace all `nan` values in Age with the Age column's average, just to have a complete column.

# In[169]:


mean_age = data_train_all['Age'].mean()  # 29.69911764705882
data_train_all['Age'] = data_train_all['Age'].fillna(mean_age)

if 'nan' in str(data_train_all['Age']).lower():
    print('Missed at least one!')    
    
mean_age_test = data_test['Age'].mean()  # 29.69911764705882.  Wasn't expecting that.
data_test['Age'] = data_test['Age'].fillna(mean_age_test)

if 'nan' in str(data_test['Age']).lower():
    print('Missed at least one!')    
print(f'{mean_age} vs {mean_age_test}')


# Rerunning the check for unique values, we can see that there are no NaN values in Age anymore.

# In[170]:


for featurename in (features_categorical + features_continuous):
    if featurename not in ['PassengerId']:
        print(f'{featurename} Unique Values:')
        print(data_train_all[featurename].unique())


# In[171]:


for featurename in (features_categorical[1:] + features_continuous):
    if featurename not in ['PassengerId']:
        print(f'{featurename} Unique Values:')
        print(data_test[featurename].unique())


# ## Feature Generation
# 
# In the lightest sense, that is. We can encode our categorical features as numerical values where they are not already. Luckily, the only one where this is the case is the Sex column.

# In[172]:


data_train_all['Sex'] = data_train_all['Sex'].replace(to_replace='male', value=0)
data_train_all['Sex'] = data_train_all['Sex'].replace(to_replace='female', value=1)
data_test['Sex'] = data_test['Sex'].replace(to_replace='male', value=0)
data_test['Sex'] = data_test['Sex'].replace(to_replace='female', value=1)


# In[173]:


data_test['Sex'].unique()


# Nice! That makes things easier for us.

# ## Visualizing the Data
# 
# I'm going to skip over this for now. In a real competition or production system, I'd be diving more into the data, but for now I'll forego that in the interest of time.

# ## Modeling!
# 
# Jumping into the fun stuff. Depending on your definition of fun.
# 
# We'll hit the easy button and work this as a logistic regression problem, since that's what we're familiar with.

# In[174]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# In[175]:


# Split the data into training and evaluation sets. This must be done after the preprocessing to avoid duplicating work across two DataFrames.
data_train, data_eval = np.split(
    data_train_all, 
    [
        int(0.8*len(data_train_all))
    ]
)

# Could have done this with sklearn with the shufflesplit. Can try on later notebooks.

print(f'{len(data_train)} / {len(data_eval)}\n{len(data_train)/len(data_train_all)} / {len(data_eval)/len(data_train_all)}')


# In[176]:


X = data_train.drop(columns=['Survived'])
y = data_train['Survived']
X_eval = data_eval.drop(columns=['Survived'])
y_eval = data_eval['Survived']

y_eval.head()


# In[177]:


# Standardize our features.
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_eval_std = scaler.fit_transform(X_eval)


# In[178]:


# Create the Logistic Regression.
clf = LogisticRegression(random_state=0)


# In[179]:


# And now we train!
#model = clf.fit(X_std, y)

# Gonna try feeding in unstandardized features.
model = clf.fit(X, y)


# In[180]:


# Once the model has been trained, we want to pass it a prediction to test its output. Let's build one.
prediction = pd.DataFrame({
    'Age': [12, 30],
    'SibSp': [2, 1],
    'Parch': [0, 1],
    'Pclass': [1, 3],
    'Sex': [0, 1],
    'Survived': [0, 1]
})

model.predict(prediction)
#model.predict_proba(prediction)


# Now that we've got a model and a toy prediction entry to give it, let's process the evaluation set the same way we did the training set, and see how it does.

# In[181]:


evaluation_predictions = model.predict(X_eval)
total_eval_predictions = len(evaluation_predictions)
incorrect_counter = 0


# In[182]:


for i in range(0, total_eval_predictions):
    if evaluation_predictions[i] != y_eval.iloc[i]:
        incorrect_counter += 1
print(f'The resulting accuracy: {(total_eval_predictions - incorrect_counter) / total_eval_predictions * 100}% correct!')


# ## Submission
# 
# Now we craft the CSV to submit.
# 
# Rules:
# 
# 
# 
# You should submit a csv file with exactly 418 entries plus a header row. Your submission will show an error if you have extra columns (beyond PassengerId and Survived) or rows.
# 
# The file should have exactly 2 columns:
# 
#     PassengerId (sorted in any order)
#     Survived (contains your binary predictions: 1 for survived, 0 for deceased)
# ```
# PassengerId,Survived
#  892,0
#  893,1
#  894,0
#  Etc.
# ```

# In[183]:


X_test = data_test
X_test_std = scaler.fit_transform(X_test)

test_prediction_results = model.predict(X_test)


# In[ ]:


submission = pd.DataFrame({
    'PassengerId': X_test['PassengerId'],
    'Survived': test_prediction_results
})


# In[184]:


submission.to_csv('titanic-test-results.csv', sep=',', index=False)
pd.read_csv('titanic-test-results.csv')


# ## And we're done!
