#!/usr/bin/env python
# coding: utf-8

# # Predicting survival of Titanic passengers using FASTAI V1

# This is my First attempt at Kaggle Kernel. So I started with the Titanic Dataset, since it is one of the most famous one, I was able to understand the importance and beauty of FASTAI V1 library that does so much from minimal coding annd feature engineering efforts.
# 
# Forked from https://www.kaggle.com/tamlyn/titanic-fastai/notebook.

# Need v1.0.25 of `fastai`as this notebook was made for this version.

# In[ ]:


get_ipython().system(u'pip install fastai --upgrade')


# In[ ]:


from fastai import *          # Quick accesss to most common functionality
from fastai.tabular import *  # Quick accesss to tabular functionality     # Access to example data provided with fastai


# ## Load data

# In[ ]:


input_path = '/kaggle/input/'
train_df = pd.read_csv(f'{input_path}train.csv')
test_df = pd.read_csv(f'{input_path}test.csv')


# ## Feature engineering
# Extract *Title* from the name colum. Extract *Deck* from the first character of the cabin number.

# In[ ]:


for df in [train_df, test_df]:
    df['Title'] = df['Name'].str.split(',').str[1].str.split(' ').str[1]
    df['Deck'] = df['Cabin'].str[0]


# ## Fastai setup

# In[ ]:


dep_var = 'Survived'
cat_names = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']
cont_names = ['Age', 'Fare', 'SibSp', 'Parch']
procs = [FillMissing, Categorify, Normalize]


# In[ ]:


test = TabularList.from_df(test_df, cat_names=cat_names, cont_names=cont_names);
test


# We can see that we do not even need to apply the same process of filling the Missing Values, Categorify the categorical variables and specify the names of the continuous variables separately for the Test dataset as well.
# FastAI handles it by itself.

# In[ ]:


data = (TabularList.from_df(train_df, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(0,200)))
                           .label_from_df(cols=dep_var)
                           .add_test(test, label=0)
                           .databunch())
data


# ## Training

# In[ ]:


learn = tabular_learner(data, layers=[200,100], metrics=accuracy)


# In[ ]:


learn.fit(8, 1e-2)


# ## Inference
# Predictions come as an array of probabilities of death or survival  for each passenger in the test set. Use `argmax` to convert each to`1` or `0` then construct the submission dataframe and save to CSV.

# In[ ]:


predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)


# In[ ]:


sub_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': labels})
sub_df.to_csv('submission.csv', index=False)


# Check that what we are submitting looks sensible.

# In[ ]:


sub_df.tail()


# Stay tuned for more Kaggle Kernels implemented using FASTAI-V1.
