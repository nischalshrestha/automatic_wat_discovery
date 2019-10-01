#!/usr/bin/env python
# coding: utf-8

# # Predicting survival of Titanic passengers

# Need at least v1.0.25 of `fastai` so update default Kaggle version.

# In[ ]:


get_ipython().system(u'pip install fastai --upgrade')


# In[ ]:


from fastai import *
from fastai.tabular import *


# ## Load data

# In[ ]:


input_path = '/kaggle/input/'
train_df = pd.read_csv(f'{input_path}train.csv')
test_df = pd.read_csv(f'{input_path}test.csv')


# ## Feature engineering
# - Extract *Title* from the name colum. 
# - Extract *Deck* from the first character of the cabin number.
# - Fill in missing *Age* values with the mean age for passengers with the same title.

# In[ ]:


for df in [train_df, test_df]:
    df['Title'] = df['Name'].str.split(',').str[1].str.split(' ').str[1]
    df['Deck'] = df['Cabin'].str[0]

# find mean age for each Title across train and test data sets
all_df = pd.concat([train_df, test_df], sort=False)
mean_age_by_title = all_df.groupby('Title').mean()['Age']
# update missing ages
for df in [train_df, test_df]:
    for title, age in mean_age_by_title.iteritems():
        df.loc[df['Age'].isnull() & (df['Title'] == title), 'Age'] = age


# 
# ## Fastai setup

# In[ ]:


dep_var = 'Survived'
cat_names = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']
cont_names = ['Age', 'Fare', 'SibSp', 'Parch']
procs = [FillMissing, Categorify, Normalize]

test = TabularList.from_df(test_df, cat_names=cat_names, cont_names=cont_names, procs=procs)
data = (TabularList.from_df(train_df, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(0,200)))
                           .label_from_df(cols=dep_var)
                           .add_test(test, label=0)
                           .databunch())


# ## Training

# In[ ]:


np.random.seed(101)
learn = tabular_learner(data, layers=[60, 20], metrics=accuracy)
learn.fit(5)


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


# In[ ]:


create_download_link(sub_df)


# In[ ]:


from IPython.display import HTML
import base64

def create_download_link( df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = f'<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    return HTML(html)

