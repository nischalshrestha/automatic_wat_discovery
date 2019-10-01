#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Dataset Submission

# ## References
# * [1] https://stackoverflow.com/questions/16353729/pandas-how-to-use-apply-function-to-multiple-columns
# * [2] https://stackoverflow.com/questions/44061607/pandas-lambda-function-with-nan-support
# * [3] https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/
# * [4] https://keras.io/losses/
# * [5] https://keras.io/callbacks/#earlystopping
# * [6] http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# * [7] https://www.kaggle.com/omarelgabry/a-journey-through-titanic

# In[96]:


import warnings
warnings.filterwarnings('ignore')

import re
import os
import json
import random
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import seaborn as sns
from datetime import datetime
from pandas.plotting import scatter_matrix

from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


# ## Loading datasets

# In[3]:


df_train = pd.read_csv('../input/train.csv')
df_submission = pd.read_csv('../input/test.csv') 
df_train.columns = map(str.lower, df_train.columns)
df_submission.columns = map(str.lower, df_submission.columns)


# ### Let's see what the data looks like? 

# In[119]:


df_train.head()


# In[120]:


df_train.info()


# In[121]:


df_submission.info()


# ### Looks like lot of NULL data

# In[122]:


df_train.isnull().sum()


# In[123]:


df_submission.isnull().sum()


# ## Completing data

# ### Some custom definitions to complete the datasets

# In[124]:


def fill_age(row, df):  
    if pd.isnull(row['age']): # [2]
        return round(df[(df['pclass'] == row['pclass']) & (df['sex'] == row['sex'])]['age'].mean())
    else:
        return row['age'] 
    
def fill_fare(row, df):
    if pd.isnull(row['fare']):  # [2]
        return round(df[(df['pclass'] == row['pclass']) & (df['sex'] == row['sex'])]['fare'].mean(), 2)
    else:
        return row['fare']  


# In[125]:


df_train.groupby('survived').size()


# ### I don't think synthetic data is required here...

# In[126]:


print(100 - 342 * 100.0 / (342+549), 342 * 100.0 / (342+549))


# In[127]:


df_train.groupby('embarked').size()


# In[128]:


# we shall fill the Embarked with 'Q' str
def fill_embarked(row):
    pass
    
# we shall fill the Cabin with 'None' str
def fill_cabin(row):
    pass


# ### We use all data to fill-in the best possible guess for missing values. We don't want any surprises during data submission!

# In[129]:


df_all = pd.concat([df_train, df_submission], ignore_index=True)


# In[130]:


df_train['cabin'].fillna(value='none', inplace=True)
df_train['age'] = df_train.apply(fill_age, axis=1, df=df_all)  # [1]
df_train['fare'] = df_train.apply(fill_fare, axis=1, df=df_all)  # [1]
df_train['embarked'].fillna(value='q', inplace=True)

df_submission['cabin'].fillna(value='none', inplace=True)
df_submission['age'] = df_submission.apply(fill_age, axis=1, df=df_all)  # [1]
df_submission['fare'] = df_submission.apply(fill_fare, axis=1, df=df_all)  # [1]
df_submission['embarked'].fillna(value='q', inplace=True)


# ### Confirmation! No missing data 

# In[131]:


df_train.isnull().sum()


# In[132]:


df_submission.isnull().sum()


# ### We like it clean, we love lowercase! We don't need any excess dimensions when we convert our categorical features 

# In[133]:


cleaner_lambda = lambda x: ''.join(re.findall(r"[\w]+",str.replace(x,' ','_').lower()))
df_train['name'] = df_train['name'].apply(cleaner_lambda)
df_train['ticket'] = df_train['ticket'].apply(cleaner_lambda)
df_train['sex'] = df_train['sex'].apply(cleaner_lambda)
df_train['cabin'] = df_train['cabin'].apply(cleaner_lambda)
df_train['embarked'] = df_train['embarked'].apply(cleaner_lambda)

df_submission['name'] = df_submission['name'].apply(cleaner_lambda)
df_submission['ticket'] = df_submission['ticket'].apply(cleaner_lambda)
df_submission['sex'] = df_submission['sex'].apply(cleaner_lambda)
df_submission['cabin'] = df_submission['cabin'].apply(cleaner_lambda)
df_submission['embarked'] = df_submission['embarked'].apply(cleaner_lambda)


# In[134]:


df_train.describe()


# In[135]:


df_submission.describe()


# ### No data analysis is complete without a correlation plot!

# In[136]:


def correlation_heatmap(df, figsize=(8, 6)):   # [7]
    _ , ax = plt.subplots(figsize=figsize)
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    plt.title('Pearson Correlation of Features', y=1.05, size=15)


# In[137]:


correlation_heatmap(df_train)


# ### Lot's of overlap in this dimension! LOL WUT!

# In[138]:


fig, ax = plt.subplots()
df_train[['age', 'survived']].plot(kind='scatter', x='age', y='survived', alpha=0.6, figsize=(16,4), c='survived', colormap='RdYlGn', ax=ax);
ax.set_xlabel("age")
plt.show()


# ## Creating data

# In[139]:


df_train['has_cabin'] = df_train['cabin'].map(lambda c: 0 if c == 'none' else 1)
df_train['family_size'] = df_train['parch'] + df_train['sibsp'] + 1
df_train['is_alone'] = df_train['family_size'].map(lambda f: 1 if f == 1 else 0)
df_train['name_length'] = df_train['name'].map(len)
df_train['ticket_length'] = df_train['ticket'].map(len)
df_train['name_length'] = df_train['name_length'].astype('float')
df_train['ticket_length'] = df_train['ticket_length'].astype('float')

df_train['sin_fare'] = df_train['fare'].map(np.sin)
df_train['cos_fare'] = df_train['fare'].map(np.cos)
df_train['sin_age'] = df_train['age'].map(np.sin)
df_train['cos_age'] = df_train['age'].map(np.cos)
df_train['embarked'] = df_train['embarked'].map({'s': 0, 'c': 1, 'q': 2})
df_train['sex'] = df_train['sex'].map({'male': 0, 'female': 1})
df_train['log_fare'] = df_train['fare'].map(lambda f: 0 if f == 0.0 else np.log(f))
df_train['log_age'] = df_train['age'].map(lambda a: 0 if a == 0.0 else np.log(a))

df_submission['has_cabin'] = df_submission['cabin'].map(lambda c: 0 if c == 'none' else 1)
df_submission['family_size'] = df_submission['parch'] + df_submission['sibsp'] + 1
df_submission['is_alone'] = df_submission['family_size'].map(lambda f: 1 if f == 1 else 0)
df_submission['name_length'] = df_submission['name'].map(len)
df_submission['ticket_length'] = df_submission['ticket'].map(len)
df_submission['name_length'] = df_submission['name_length'].astype('float')
df_submission['ticket_length'] = df_submission['ticket_length'].astype('float')

df_submission['sin_fare'] = df_submission['fare'].map(np.sin)
df_submission['cos_fare'] = df_submission['fare'].map(np.cos)
df_submission['sin_age'] = df_submission['age'].map(np.sin)
df_submission['cos_age'] = df_submission['age'].map(np.cos)
df_submission['embarked'] = df_submission['embarked'].map({'s': 0, 'c': 1, 'q': 2})
df_submission['sex'] = df_submission['sex'].map({'male': 0, 'female': 1})
df_submission['log_fare'] = df_submission['fare'].map(lambda f: 0 if f == 0.0 else np.log(f))
df_submission['log_age'] = df_submission['age'].map(lambda a: 0 if a == 0.0 else np.log(a))


# ### We use all data to create the best bins possible. We don't want any surprises during data submission!

# In[140]:


df_all = pd.concat([df_train, df_submission], ignore_index=True)   # [7]


# In[141]:


ser_fare, bins_fare = pd.qcut(df_all["fare"], 5, retbins=True, labels=range(5))
ser_age, bins_age = pd.cut(df_all['age'].astype(int), 5, retbins=True, labels=range(5)) 

df_train['fare_bin'] = pd.cut(df_train["fare"], bins=bins_fare, labels=range(5), include_lowest=True)
df_train['age_bin'] = pd.cut(df_train['age'].astype(int), bins=bins_age, labels=range(5), include_lowest=True)
df_train['fare_bin'] = df_train['fare_bin'].astype('int')
df_train['age_bin'] = df_train['age_bin'].astype('int')

df_submission['fare_bin'] = pd.cut(df_submission["fare"], bins=bins_fare, labels=range(5), include_lowest=True)
df_submission['age_bin'] = pd.cut(df_submission['age'].astype(int), bins=bins_age, labels=range(5), include_lowest=True)
df_submission['fare_bin'] = df_submission['fare_bin'].astype('int')
df_submission['age_bin'] = df_submission['age_bin'].astype('int')


# ## Ticket please!!!

# In[142]:


def fill_ticket_class(row):
    ticket_classes = {'pc':0, 'ston':1, 'ca':2, 'soton':3, 'a4':4, 'a5':5, 'sc':6, 'wc':7, 'soc':8 }
    for k, v in ticket_classes.items():
        if k in row['ticket']:
            return v
    if row['ticket'].isnumeric():
        return 9
    else:
        return 10


# In[143]:


df_train['ticket_class'] = df_train.apply(fill_ticket_class, axis=1)  # [1]
df_submission['ticket_class'] = df_submission.apply(fill_ticket_class, axis=1)  # [1]


# In[144]:


df_train.describe()


# In[145]:


df_submission.head()


# In[146]:


correlation_heatmap(df_train, (16, 16))


# ### We use all possible data to create our categorical binarizers. We don't want any unrecognized labels during data submission!

# In[147]:


df_all = pd.concat([df_train, df_submission], ignore_index=True)   # [7]


# In[148]:


label_columns = ['pclass', 'sex', 'cabin', 'embarked', 'fare_bin', 'age_bin', 'is_alone', 'ticket_class', 'has_cabin']
binarizers = {}
for col in label_columns:
    binarizers[col] = MultiLabelBinarizer()
    binarizers[col].fit([list(df_all[col].unique())])


# ### Train-Test split! Just before creating the MinMaxScalers

# In[149]:


train_X, test_X, train_y, test_y = train_test_split(df_train[list(df_train.columns.difference(['survived']))],df_train['survived'], train_size=0.8, test_size=0.2, shuffle=True, random_state=0)


# In[150]:


print('train_X.shape', train_X.shape)
print('train_y.shape', train_y.shape)
print('test_X.shape', test_X.shape)
print('test_y.shape', test_y.shape)


# ### Unlike MultiLabelBInarizers we use only the train_X to create our MinMaxScalers

# In[151]:


float_columns = ['age', 'fare', 'sin_age', 'cos_age', 'sin_fare', 'cos_fare', 'ticket_length', 'name_length', 'log_fare', 'log_age']
minmaxscalers = {}
for col in float_columns:
    minmaxscalers[col] = MinMaxScaler()
    minmaxscalers[col].fit(train_X[col].as_matrix().reshape((train_X[col].as_matrix().shape[0], 1)))


# ## The right way to encode your data 

# In[152]:


train_X_transformed = {}
for col in label_columns:
    df_x = train_X[col].apply(lambda x: binarizers[col].transform([[x]])[0])
    train_X_transformed[col] = pd.DataFrame([tuple(x) for x in df_x.as_matrix()], columns=[col + '_' + str(l) for l in binarizers[col].classes_])


# In[153]:


for col in float_columns:
    df_x = train_X[col].apply(lambda x: minmaxscalers[col].transform([[x]])[0])
    train_X_transformed[col] = pd.DataFrame([tuple(x) for x in df_x.as_matrix()], columns=[col])


# In[154]:


train_X_transformed.keys()


# In[155]:


test_X_transformed = {}
for col in label_columns:
    df_x = test_X[col].apply(lambda x: binarizers[col].transform([[x]])[0])
    test_X_transformed[col] = pd.DataFrame([tuple(x) for x in df_x.as_matrix()], columns=[col + '_' + str(l) for l in binarizers[col].classes_])


# In[156]:


for col in float_columns:
    df_x = test_X[col].apply(lambda x: minmaxscalers[col].transform([[x]])[0])
    test_X_transformed[col] = pd.DataFrame([tuple(x) for x in df_x.as_matrix()], columns=[col])


# In[157]:


test_X_transformed.keys()


# In[158]:


submission_X_transformed = {}
for col in label_columns:
    df_x = df_submission[col].apply(lambda x: binarizers[col].transform([[x]])[0])
    submission_X_transformed[col] = pd.DataFrame([tuple(x) for x in df_x.as_matrix()], columns=[col + '_' + str(l) for l in binarizers[col].classes_])


# In[159]:


for col in float_columns:
    df_x = df_submission[col].apply(lambda x: minmaxscalers[col].transform([[x]])[0])
    submission_X_transformed[col] = pd.DataFrame([tuple(x) for x in df_x.as_matrix()], columns=[col])


# In[160]:


submission_X_transformed.keys()


# ## Let's make some final varibles to play!

# In[175]:


# Lot of room to play with, choose which columns you want to experiment with!! I just took them all
attributes = label_columns + float_columns


# In[161]:


X = pd.concat([v for k, v in train_X_transformed.items() if k in attributes], axis=1)
X.head()


# In[162]:


X = X.as_matrix()


# In[164]:


train_y.head()


# In[165]:


y = train_y.as_matrix()


# In[166]:


tX = pd.concat([v for k, v in test_X_transformed.items() if k in attributes], axis=1)
tX.head()


# In[ ]:


tX = tX.as_matrix()


# In[167]:


test_y.head()


# In[168]:


ty = test_y.as_matrix()


# In[169]:


X_submission = pd.concat([v for k, v in submission_X_transformed.items() if k in attributes], axis=1)
X_submission.head()


# In[170]:


X_submission = X_submission.as_matrix()


# ## Here comes the KernalPCA Transformation

# In[171]:


kpca = KernelPCA(kernel='rbf')   # [3]
kpca.fit(X)


# In[172]:


kpca_X = kpca.transform(X)
kpca_tX = kpca.transform(tX)
kpca_X_submission = kpca.transform(X_submission)


# ## Fancy stats describe*^%&%

# In[173]:


scipy.stats.describe(X[0])


# In[174]:


scipy.stats.describe(kpca_X[0])


# ## You just found Keras! Deep Learning for Humans!

# In[76]:


def build_model(nb_features):
    model = Sequential()
    model.add(Dense(16, input_shape=(nb_features,), activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.005), metrics=['accuracy'])  # [4]
    return model


# In[79]:


model = build_model(kpca_X.shape[1])
early_stop = EarlyStopping(monitor="val_loss", patience=1)    # [5]
model.fit(kpca_X, y, epochs=100, batch_size=10, validation_split=0.1, callbacks=[early_stop], verbose=True)


# In[80]:


preditions = model.predict(kpca_tX)
preditions = np.where(preditions > 0.5, 1, 0)
print(accuracy_score(ty, preditions))


# ## It's a long way to the top if you wanna rock and roll!!!

# In[81]:


submission_preditions = model.predict(kpca_X_submission)
submission_preditions = np.where(submission_preditions > 0.5, 1, 0)


# In[82]:


def deploy_submission(y, output='submission.csv'):
    submission = pd.DataFrame({
        'PassengerId': df_submission.passengerid.values,
        'Survived': y.reshape(-1)
    })
    submission.to_csv(output, index=False)


# In[111]:


deploy_submission(submission_preditions)

