#!/usr/bin/env python
# coding: utf-8

# # Data Analysis
# 

# ## import library

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

get_ipython().magic(u'matplotlib inline')
sns.set(style="darkgrid", color_codes=True, palette='deep')


# ## csv path

# In[ ]:


train_csv = "../datasets/train.csv"
test_csv = "../datasets/test.csv"


# ## Read csv files

# In[ ]:


train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)


# ## Show dataframe 

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# ## Df describe

# In[ ]:


train_df.describe()


# In[ ]:


test_df.describe()


# ### Take away
# - Mean age is slice different between train and test (29.699 and 30.27)
# - Max age is 80 in train data and 76 in test data.
# - Max parch is 6 in train data and 9 in test data. This shold be consider when building  prediction model.
# - Max Sibsp is 8 both of train and test data.

# ## Confirm NaN value counts

# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# ### Take away
# - Age shourd be fill by any method.
# - Cabin columns has many null value.

# ## honoric of Name 

# In[ ]:


def get_honoric(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for df in [train_df, test_df]:
    df['Honoric'] = df['Name'].apply(get_honoric)


# In[ ]:


# lets us plot many diffrent shaped graphs together 

train_df['Honoric'].value_counts().plot(kind='bar', fontsize=12, figsize=(8, 3))
plt.title('train data Honoric histgram')
plt.show()

test_df['Honoric'].value_counts().plot(kind='bar', fontsize=12, figsize=(8, 3))
plt.title('test data Honoric histgram')
plt.show()


# According to  wikipedia, 
# - Mlle (French) equivalent in English is "Miss"
# https://en.wikipedia.org/wiki/Mademoiselle_(title)
# - Mme (French) equivalent in English is "Mrs"
# https://en.wikipedia.org/wiki/Madam
# 
# so, let's replace Mlle as Miss and Mme.   
# And, Ms can count as "Miss", replace Ms as Miss.  
# Other honorifics replace as "Rare".

# In[ ]:


# Replace
for df in [train_df, test_df]:
    df['Honoric'] = df['Honoric'].replace({'Mlle' :'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    rare_honorics = set(df['Honoric']) - set(['Mr', 'Miss', 'Mrs', 'Master'])
    replace_dict = {rh: 'Rare' for rh in rare_honorics}
    df['Honoric'] = df['Honoric'].replace(replace_dict)


# In[ ]:


# lets us plot many diffrent shaped graphs together 
train_df['Honoric'].value_counts().plot(kind='bar', fontsize=12, figsize=(8, 3))
plt.title('train data Honoric histgram')
plt.show()

test_df['Honoric'].value_counts().plot(kind='bar', fontsize=12, figsize=(8, 3))
plt.title('test data Honoric histgram')
plt.show()


# ## Create New Feature

# In[ ]:


for df in [train_df, test_df]:
    df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Name_length'] = df['Name'].apply(len)
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1


# ## Find estimated age columns
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

# In[ ]:


for df in [train_df, test_df]:
    is_estimated = df['Age'].apply(lambda x: True if x - np.floor(x) == 0.5 and x >1 else False)
    df['is_estimated_Age'] = is_estimated


# In[ ]:


train_df[train_df['is_estimated_Age'] == True]


# ## bining

# In[ ]:


# create bins
def make_bins(df, bins, col='Age'):
    bins_col_name = col + '_bins'
    bin_labels = []
    for i in range(len(bins) - 1):
        bins_string = str(bins[i]) + '~' + str(bins[i + 1] - 1)
        bin_labels.append(bins_string)
    df[bins_col_name] = pd.cut(df[col], bins, labels=bin_labels, right=False)
    
# bining Age
age_bins = np.arange(0, 90, 5)
make_bins(train_df, bins=age_bins, col='Age')
make_bins(test_df, bins=age_bins, col='Age')

# bining Fare
fare_bins = np.arange(0, 600, 50)
make_bins(train_df, bins=fare_bins, col='Fare')
make_bins(test_df, bins=fare_bins, col='Fare')


# # Explore Train data

# ## Correlation coefficient

# In[ ]:


plt.figure(figsize=(10, 10))
sns.heatmap(train_df.corr(), square=True, annot=True, cmap='Greens')
plt.show()


# - PclassとSurvivedに負の相関関係（階級が1 = 高級なほど生存しやすい）が見られる
# - SurvivedとFareに相関関係が見られる（Fareが高いほど生存しやすい）
# - PclassとFareに負の相関関係（階級が1 = 高級なほどFareが高い）
# - ParchとSibspに相関関係（親がいると兄弟もいる）

# In[ ]:


# Fare > 100 Passengers are all 1 Pclass
print(train_df[train_df['Fare'] > 100]['Pclass'].unique())

plt.scatter(train_df['Fare'], train_df['Pclass'])
plt.title('scatter plot by Fare and Pclass')
plt.ylabel('Pclass')
plt.xlabel('Fare')
plt.show()


# ## Original plot function

# In[ ]:


def survival_plot(df, ax, col='Sex', stacked=True):
    df.groupby([col, 'Survived']).size().unstack().plot(kind='bar', stacked=True, ax=ax)
    plt.title("Survival by {}, (1 = Survived)".format(col))
    
def age_kde_plot(df, ax):
    df.groupby('Pclass')['Age'].plot(kind='kde')
    plt.xlabel("Age")
    plt.title("Age Distribution within classes")
    # sets our legend for our graph.
    plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 
    
    
def  original_plot(df, survival_plot=survival_plot, age_kde_plot=age_kde_plot):
    fig = plt.figure(figsize=(18,30)) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    rows = 3
    columns = 5

    ax = plt.subplot2grid((columns, rows),(0,0))
    survival_plot(df=df, ax=ax, col='Sex', stacked=True)

    ax  = plt.subplot2grid((columns, rows),(0,1))
    survival_plot(df=df, ax=ax, col='Age_bins', stacked=True)

    ax = plt.subplot2grid((columns, rows),(0,2))
    survival_plot(df=df, ax=ax, col='Pclass', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(1,0))
    survival_plot(df=df, ax=ax, col='Fare_bins', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(1,1))
    survival_plot(df=df, ax=ax, col='Has_Cabin', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(1,2))
    survival_plot(df=df, ax=ax, col='Embarked', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(2,0))
    survival_plot(df=df, ax=ax, col='SibSp', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(2,1))
    survival_plot(df=df, ax=ax, col='Parch', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(2,2))
    survival_plot(df=df, ax=ax, col='FamilySize', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(3,0))
    survival_plot(df=df, ax=ax, col='IsAlone', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(3,1), colspan=2)
    survival_plot(df=df, ax=ax, col='Name_length', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(4,0), colspan=3)
    age_kde_plot(df=df, ax=ax)
    
    plt.show()


# In[ ]:


original_plot(train_df)


# ### Normalize plot

# In[ ]:


def survival_plot_normed(df, ax, col='Sex', stacked=True):
    (df.groupby([col, 'Survived']).size().unstack().T / df.groupby([col, 'Survived']).size().unstack().sum(axis=1)).T.plot(kind='bar', stacked=stacked, ax=ax)
    plt.title("Survival by {}, (1 = Survived)".format(col))


# In[ ]:


original_plot(train_df, survival_plot_normed)


# ### Take away
# - 女性が多く助かっている
# - 子供が多く助かっている
# - 65歳以上はもれなく死亡
# - Classは3が最も多い（低級）
# - 年齢が高いほどClassが高い傾向
# - borading locationはSが一番多い
# - 年齢の補完にクラス毎の年齢分布が使えそう
# - 名前が長いほど生存率が高い？

# ## VS test data

# In[ ]:


train_df['is_train'] = True
test_df['is_train'] = False
merged_df = pd.concat([train_df, test_df])


# In[ ]:


def survival_plot_merged(df, ax, col='Sex', stacked=False):
    (df.groupby([col, 'is_train']).size().unstack() / df.groupby([col, 'is_train']).size().unstack().sum(axis=0)).plot(kind='bar', ax=ax)
    plt.title("is_train by {}, (1 = Survived)".format(col))
    
def age_kde_plot_merged(df, ax):
    df.groupby('is_train')['Age'].plot(kind='kde')
    plt.xlabel("Age")
    plt.title("Age Distribution within merged_data")
    # sets our legend for our graph.
    plt.legend(('Train', 'Test'),loc='best') 
    
original_plot(merged_df, survival_plot=survival_plot_merged, age_kde_plot=age_kde_plot_merged)


# ### Take away
# - 0 ~ 4歳の割合が学習データの方が高い
# - 20 ~ 24歳の割合がテストデータの方が高い

# ## Save csv data

# In[ ]:


train_df.to_csv('../datasets/new_train.csv')
test_df.to_csv('../datasets/new_test.csv')
merged_df.to_csv('../datasets/new_merged.csv')

