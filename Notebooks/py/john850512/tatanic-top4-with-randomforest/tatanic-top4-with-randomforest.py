#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import RFECV

default_path = '../input/'
sns.set_palette('hls')


# In[ ]:


train_df = pd.read_csv(default_path+'train.csv')
test_df = pd.read_csv(default_path+'test.csv')


# In[ ]:


dataset_df = pd.concat([train_df, test_df]).reset_index(drop=True)
dataset_df.info()


# ## Data Analysis

# ### 1.Sex

# In[ ]:


sns.countplot('Sex',hue='Survived', data=dataset_df)
dataset_df[['Sex', 'Survived']].groupby('Sex').mean()


# ### 2.Pclass

# In[ ]:


print(dataset_df[['Pclass', 'Survived']].groupby('Pclass').mean())
fig, [ax, ax1] = plt.subplots(1, 2)
fig.set_size_inches(9, 4)
sns.countplot(x='Pclass', data=dataset_df, ax=ax)
g = sns.factorplot('Pclass', 'Survived', data=dataset_df, kind='bar', ax=ax1)
plt.close(g.fig)


# ### 3.Embarked

# In[ ]:


print(dataset_df[['Embarked', 'Survived']].groupby('Embarked').mean())
fig, [ax, ax1] = plt.subplots(1, 2)
fig.set_size_inches(9, 4)
sns.countplot('Embarked', data=dataset_df, ax=ax)
g = sns.factorplot('Embarked', 'Survived', data=dataset_df, kind='bar',ax=ax1)
plt.close(g.fig)


# ### 4.Fare

# In[ ]:


dataset_df['Fare'] = dataset_df['Fare'].fillna(dataset_df['Fare'].median())
sns.distplot(dataset_df['Fare'], color='b', label='skewness:%.2f'%dataset_df['Fare'].skew()).legend(loc='best')


# ### 5.Parch

# In[ ]:


print(dataset_df[['Parch','Survived']].groupby('Parch').mean())
sns.factorplot('Parch', 'Survived',data=dataset_df, kind='bar', palette='hls')


# ### 6.SibSp

# In[ ]:


print(dataset_df[['SibSp','Survived']].groupby('SibSp').mean())
sns.factorplot('SibSp', 'Survived',data=dataset_df, kind='bar', palette='hls')


# ### 7.Age

# In[ ]:


sns.FacetGrid(data=dataset_df, col='Survived').map(sns.distplot, 'Age', color='b')


# ## Feature Engineering

# ### 1. Fare

# In[ ]:


# make log for 'Fare' feature in order to decrease skewness
dataset_df['Fare'] = dataset_df['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
sns.distplot(dataset_df['Fare'], color='r', label='skewness:%.2f'%dataset_df['Fare'].skew()).legend(loc='best')


# In[ ]:


# doing bins_cut for 'Fare' feature
# we dont know which bins_cuts is better, so we do RFE latter(in feature selection part) 

fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharey=True)
fig.set_size_inches(15, 4)

for q, ax in zip([4,5,6],[ax1, ax2, ax3]):
    diff_fare_bin_names = 'FareBins_'+str(q)
    fare_bins = pd.qcut(dataset_df['Fare'], q=q, labels=False )
    dataset_df[diff_fare_bin_names] = pd.DataFrame(fare_bins)
    print(dataset_df[[diff_fare_bin_names,'Survived']].groupby(diff_fare_bin_names).mean())
    # factorplot會自行產生一個fig，所以我們要另外關掉
    # https://stackoverflow.com/questions/33925494/seaborn-produces-separate-figures-in-subplots
    g = sns.factorplot(diff_fare_bin_names, 'Survived', data=dataset_df, kind='bar', palette='hls', ax=ax)
    plt.close(g.fig)


# ### 2.FamilySize

# In[ ]:


dataset_df['FamilySize'] = dataset_df['Parch'] + dataset_df['SibSp']


# In[ ]:


sns.factorplot('FamilySize', 'Survived',data=dataset_df, kind='bar')


# In[ ]:


# doing bins_cut for 'FamilySize' feature
# 0 for (-1, 0] ->0
# 1 for (0, 3]  ->1~3
# 2 for (3, 10] ->4~10

# but, unfortunately, this feature seems make overfitting(when I submit and see result on LB)...
# so I wont use this feature to training.. 
bins = [-1, 0, 3, 10]
dataset_df['FamilySizeBins_3'] = pd.cut(dataset_df['FamilySize'], bins=bins, labels=False)
sns.factorplot('FamilySizeBins_3', 'Survived', data=dataset_df, kind='bar')


# ### 3.Embarked
# it seems dont work better for this training... when I removed this feature then LB scores up
# 
# I think the reason is this feature relate to 'Fare' feature(e.x. people who from 'S' embarked likely rich than other, so its 'Fare' higher too...)

# In[ ]:


dataset_df['Embarked'] = dataset_df['Embarked'].fillna('S')


# In[ ]:


dataset_df['Embarked'] = dataset_df['Embarked'].map({'S':0, 'C':1, 'Q':2})
sns.factorplot('Embarked', 'Survived', data=dataset_df, kind='bar')


# ### 4.Sex

# In[ ]:


dataset_df['Sex'] = dataset_df['Sex'].map({'male':0,'female':1})


# ### 5.Title

# In[ ]:


dataset_df['Title'] = pd.DataFrame([i.split(",")[1].split(".")[0].strip() for i in dataset_df['Name']])
g = sns.countplot(dataset_df['Title'])
g = plt.setp(g.get_xticklabels(), rotation=45)


# In[ ]:


dataset_df['Title'] = dataset_df['Title'].replace(['Don', 'Rev', 'Dr', 'Mme',
                                                   'Ms','Major', 'Lady', 'Sir',
                                                   'Mlle', 'Col', 'Capt', 'the Countess',
                                                   'Jonkheer', 'Dona'],'Rare')
print(dataset_df['Title'].unique())
sns.countplot(dataset_df['Title'])


# In[ ]:


dataset_df['Title'] = dataset_df['Title'].map({'Master': 0, 'Mr':1, 'Miss':2, 'Mrs':2, 'Rare':3})
sns.factorplot('Title', 'Survived', data=dataset_df, kind='bar')


# ### 6.Age

# In[ ]:


# Observe the fact that missing value of age maybe cause inbalance data distribution
dataset_df['HasAge'] = dataset_df['Age'].isnull().map(lambda i : 1 if i == True else 0)
fig, [ax, ax1, ax2] = plt.subplots(3, 2)
fig.set_size_inches(14, 10)
sns.countplot('Sex', hue='HasAge', data=dataset_df, ax = ax[0]).legend(loc=1)
sns.countplot('Parch', hue='HasAge', data=dataset_df, ax = ax[1]).legend(loc=1)
sns.countplot('SibSp', hue='HasAge', data=dataset_df, ax = ax1[0]).legend(loc=1)
sns.countplot('Pclass', hue='HasAge', data=dataset_df, ax = ax1[1]).legend(loc=1)
sns.countplot('FareBins_6', hue='HasAge', data=dataset_df, ax = ax2[0]).legend(loc=1)
sns.countplot('Title', hue='HasAge', data=dataset_df, ax = ax2[1]).legend(loc=1)


# In[ ]:


print(dataset_df[['Sex', 'Age']].groupby('Sex').median())
print('-' * 30)
# ahhh...'Master' there is mean little boy
# It seems that use 'Title' median can make better distinguish than 'Sex' feature
print(dataset_df[['Title', 'Age']].groupby('Title').median())


# In[ ]:


# fill missing value with 'Title' median
AgeBins = dataset_df[['Age','Title']].groupby('Title').median().values
dataset_df['NewAge'] = dataset_df['Age'].copy()

NullAge_idx = dataset_df.loc[dataset_df['Age'].isnull()==True]['Title'].index.values
NullAgeBins_idx = dataset_df.loc[dataset_df['Age'].isnull()==True]['Title'].values
dataset_df['NewAge'][NullAge_idx] = AgeBins[NullAgeBins_idx].ravel()
dataset_df['NewAge'] = dataset_df['NewAge'].astype('int')


# In[ ]:


dataset_df['AgeLessThan17'] = (dataset_df['NewAge'] < 17) * 1 #multiply 1 makes boolean change to numbers
sns.factorplot('AgeLessThan17', 'Survived', data=dataset_df, kind='bar')


# ### 7.Others

# In[ ]:


dataset_df['Ticket'].describe()


# In[ ]:


deplicate_ticket = []
for tk in dataset_df.Ticket.unique():
    tem = dataset_df.loc[dataset_df.Ticket == tk, 'Fare']
    #print(tem.count())
    if tem.count() > 1:
        #print(df_data.loc[df_data.Ticket == tk,['Name','Ticket','Fare']])
        deplicate_ticket.append(dataset_df.loc[dataset_df.Ticket == tk,['Name','Ticket','Fare','Cabin','FamilySize','Survived']])
deplicate_ticket = pd.concat(deplicate_ticket)
deplicate_ticket.head(20)


# In[ ]:


df_fri = deplicate_ticket.loc[(deplicate_ticket.FamilySize == 0) & (deplicate_ticket.Survived.notnull())].head(7)
df_fami = deplicate_ticket.loc[(deplicate_ticket.FamilySize > 0) & (deplicate_ticket.Survived.notnull())].head(7)
display(df_fri,df_fami)
print('people keep the same ticket: %.0f '%len(deplicate_ticket))
print('friends: %.0f '%len(deplicate_ticket[deplicate_ticket.FamilySize == 0]))
print('families: %.0f '%len(deplicate_ticket[deplicate_ticket.FamilySize > 0]))


# In[ ]:


# the same ticket family or friends
dataset_df['Connected_Survival'] = 0.5 # default 
for _, df_grp in dataset_df.groupby('Ticket'):
    if (len(df_grp) > 1):
        for ind, row in df_grp.iterrows(): #相同tickets的逐列枚舉
            smax = df_grp.drop(ind)['Survived'].max() #扣掉自己找剩下列的最大值
            smin = df_grp.drop(ind)['Survived'].min() #扣掉自己找剩下列的最小值
            passID = row['PassengerId']
            if (smax == 1.0 and smin == 1.0): #如果最大和最小都是1，代表全部都活著
                dataset_df.loc[dataset_df['PassengerId'] == passID, 'Connected_Survival'] = 1
            elif (smax == 0.0 and smin==0.0): #如果最大和最小都是0，代表全部都死亡
                dataset_df.loc[dataset_df['PassengerId'] == passID, 'Connected_Survival'] = 0
#print
print('people keep the same ticket: %.0f '%len(deplicate_ticket))
print("people have connected information : %.0f" 
      %(dataset_df[dataset_df['Connected_Survival']!=0.5].shape[0]))
dataset_df.groupby('Connected_Survival')[['Survived']].mean().round(3)


# ## Data Preprocess

# In[ ]:


dataset_df.head()


# ## Feature Selection

# In[ ]:


dataset_df.columns


# In[ ]:


# results tell us choose n fatures for best cv scores
# and then we can choose feature depends on ranking_

# and it's seems like FareBins_6 will cause overfiting(result of LB decrease)
temp_df_for_feature_selection = ['Connected_Survival','AgeLessThan17', 'Embarked', 'Pclass', 'Parch', 'SibSp', 'Sex', 'FamilySize', 'FareBins_4', 'FareBins_5', 'FareBins_6']
X_train_for_feature_selection = (dataset_df[:len(train_df)])[temp_df_for_feature_selection]
y_train_for_feature_selection = train_df['Survived']
RFC = RandomForestClassifier(n_estimators=250, n_jobs=4)
RFEselector = RFECV(estimator=RFC, cv=10, n_jobs=4)
RFEselector.fit(X_train_for_feature_selection, y_train_for_feature_selection)
print(RFEselector.ranking_)
print(RFEselector.n_features_)
print(RFEselector.grid_scores_ * 100)

# free memory
del temp_df_for_feature_selection, X_train_for_feature_selection, y_train_for_feature_selection, RFC, RFEselector


# ## Training

# In[ ]:


data = dataset_df[['Title', 'Connected_Survival', 'AgeLessThan17', 'Sex',
                   'Pclass', 'FareBins_5', 'Survived']]
train_data = data[:len(train_df)]
test_data = data[len(train_df):]
test_data = test_data.drop(['Survived'],axis=1)

X_train = train_data.values[:, :-1]
y_train = train_data.values[:, -1:].ravel()
X_test = test_data.values


# In[ ]:


kfold = StratifiedKFold(n_splits=10)
RFC = RandomForestClassifier(random_state=2, n_estimators=250, oob_score=True, n_jobs=4).fit(X_train, y_train)
print(RFC.oob_score_)

cv = cross_val_score(RFC, X_train, y_train, scoring='accuracy', cv=kfold, n_jobs=4, verbose=1)
print(cv.mean(), cv.std())


# In[ ]:


prediction = RFC.predict(X_test)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': prediction.astype(int)})
submission.to_csv('submission.csv', index=False)
submission.head()

