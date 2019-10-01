#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import CategoricalDtype

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from matplotlib import style
style.use('seaborn')

import os
print(os.listdir("../input"))

get_ipython().magic(u'matplotlib inline')


# In[ ]:


pd.options.display.float_format = '{:,.2f}'.format


# ## Dataset Import

# In[ ]:


PATH = '../input/'


# In[ ]:


df = pd.read_csv(PATH + 'train.csv', low_memory=False, index_col=0)
test = pd.read_csv(PATH + 'test.csv', low_memory=False, index_col=0)
df.head()


# In[ ]:


df.shape, test.shape


# In[ ]:


df.describe(include='all')


# In[ ]:


test.describe(include='all')


# Both the training set and the test set numerical features appear to have a similar distribution, which is promising for applying ML solutions.

# ## Dataset Initial Analysis

# There were a total of 2224 passengers and crew on board the Titanic when the ship left for her maiden voyage. 1502 people died during the catastrophe that ensued after colliding with the iceberg, leaving the survival rate at roughly 32.5%.
# 
# If we only look at passengers though, there were approximately 1300 leaving with the ship and 812 that died, leaving the passenger survival rate at a higher level of 37%, which is close to the one in our training sample - 38%. Thus, we expect the test sample to display a similar survival rate.
# 
# Let's look closer at some of the features in our dataset:

# **Pclass**
# 
# The ticket class, indicating wealth and cabin placement on the deck - the higher the class, the closer the assigned cabins were to the life boats.

# In[ ]:


df[['Survived']].groupby(df['Pclass']).mean().plot.bar()
plt.show()


# Not surprisingly, the survival rate appears to be decreasing sharply for higher class numbers

# **Sex**

# In[ ]:


df[['Survived']].groupby(df['Sex']).mean().plot.bar()
plt.show()


# Women had a much higher survival rate - not surprising as they most likely had priority over the male passengers into the lifeboats.

# **Age**

# In[ ]:


bins=[b for b in range(0, 91, 5)]
fig = plt.figure()
# ax0 = plt.subplot2grid((1, 2), (0, 0), colspan=2)
# plt.title("Age Histogram")

ax1 = plt.subplot2grid((1, 2), (0, 0))
df['Age'][df['Survived'] == 1].hist(bins=bins, color='g')
plt.title("Surviving passengers")
ax2 = plt.subplot2grid((1, 2), (0, 1), sharey=ax1)
df['Age'][df['Survived'] == 0].hist(bins=bins, color='r')
plt.title("Deceased passengers")
plt.tight_layout()
plt.show()


# Notable difference between the surviving and deceased passenger age groups is that small children were more significantly more likely to survive.
# 
# In general, the conclusions we draw so far is that indeed, the principle of *"women and children first"* was followed.

# In[ ]:


threshold = 7
df[['Survived']].groupby(df['Age'].apply(lambda x: f'below {threshold}' if x < threshold else f'above {threshold}')).mean().plot.bar()
plt.show()


# **Name & title**

# We extract the passenger title from the Name column:

# In[ ]:


df[['Survived']].groupby(df['Name'].apply(lambda x: x.split(sep = ',')[1].split(sep = ".")[0].strip())).mean().sort_values(by="Survived").plot.bar()
plt.show()


# Anyone with the title Mr. was very likely to die that day - this comes as no surprise after our analysis of passenger sex.

# **SibSp and Parch**

# In[ ]:


df[['Survived']].groupby([df['SibSp']]).mean().plot.bar()
plt.show()


# In[ ]:


df[['Survived']].groupby([df['Parch']]).mean().plot.bar()
plt.show()


# It's difficult to see any pattern in the above features. Given the fact they both relate to number of family members, perhaps we should look at them in combination:

# In[ ]:


df[['Survived']].groupby([df['Parch'] + df['SibSp']]).mean().plot.bar()
plt.show()


# In[ ]:


df[['Survived']].groupby([df['Parch'] + df['SibSp']]).count()/df.shape[0]


# It looks like most passengers did not have family members onboard and those who did had a bigger chance of surviving - perhaps families 

# **Embarked**

# In[ ]:


df[['Survived']].groupby([df['Embarked']]).mean().plot.bar()
plt.show()


# In[ ]:


df[['Survived']].groupby([df['Embarked']]).count()


# Passengers from Cherbourg managed to survive more frequently for some reason.

# ## Data Pre-processing

# **Missing Values**

# We analyse the percentage of missing values in the dataset features:

# In[ ]:


df.isna().sum()[df.isna().sum() != 0]/df.shape[0] * 100


# In[ ]:


test.isna().sum()[test.isna().sum() != 0]/test.shape[0] * 100


# Let's look at cabin first, where over 77% observations are missing...

# **Cabin**

# In[ ]:


df['Cabin'].value_counts()


# First, we make note the cabin numbers alone most likely introduce little new information to what we already know from analysing the Pclass and the cabin number is too passenger specific anyway. However, from looking at the picture below we see that interesting conclusions can be drawn from the letter idicating the deck as the upper decks are closer to the rescue boats as we hinted earlier:
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Titanic_cutaway_diagram.png/687px-Titanic_cutaway_diagram.png" width="400">
# 
# That's why for the purpose of our analysis, we will create a new feature Deck - the extracted deck information from the Cabin feature.
# In the process, we also replace the missing values with 'NA's.
# 

# In[ ]:


df['Deck'] = df['Cabin'].fillna(value='NA').apply(lambda x: ''.join(filter(str.isalpha, x))[0] if x != 'NA' else x)
test['Deck'] = test['Cabin'].fillna(value='NA').apply(lambda x: ''.join(filter(str.isalpha, x))[0] if x != 'NA' else x)


# In[ ]:


df[['Survived']].groupby(df['Deck']).mean().plot.bar()
plt.title("Survival rate by deck")
plt.show()


# **Age**

# Age is going to be a little more tricky. Our observation so far has been that children had a higher survival rate. Therefore it would be beneficial for the model accuracy to differentiate between children and adults in the missing age data group.
# 
# Let's have a look at the passenger names then:

# In[ ]:


df[['Name', 'Sex', 'Age']].sort_values('Age')


# It looks like young boys were addressed as *Master* in the register. In general, the median age differs depending on the title:

# In[ ]:


titles = ['Mr.', 'Master.', 'Rev.', 'Dr.', 'Sir.', 'Don.', 'Capt.', 'Lady.', 'Miss.', 'Ms.', 'Mrs.']

for title in titles:
    print(f"Title: {title}, Median of {df[['Name', 'Age']][df['Name'].str.contains(title)].median()}")


# In[ ]:


df[['Name', 'Age']][~df['Name'].str.contains(', M')]


# We'll therefore insert the missing age values using the median of the title:

# In[ ]:


def db_mod(db, title): db.loc[db[db['Name'].str.contains(title) & (db['Age'].isnull())].index, 'Age'] = df["Age"][df['Name'].str.contains(title)].median()

for title in titles: 
    db_mod(df, title=title)
    db_mod(test, title=title)


# **Other missing values**

# We are only missing a small number of values for Embarked in the training set, which we will fill with 'NA's

# In[ ]:


df['Embarked'].fillna(value='NA', inplace=True)


# As for the missing fare information in the Test dataset, we will substitute it with the Pclass mean from the training set:

# In[ ]:


df[['Pclass', 'Fare']].groupby('Pclass').mean()


# In[ ]:


test[test['Fare'].isnull()]


# Turns out this is just a single observation. Let's fill it in:

# In[ ]:


test.loc[test[(test['Pclass'] == 3) & (test['Fare'].isnull())].index, 'Fare'] = df['Fare'][df['Pclass'] == 3].mean()


# ## Feature Selection

# We start off by adding new features into our dataset:

# In[ ]:


df['FamSize'] = df['SibSp'] + df['Parch']
test['FamSize'] = test['SibSp'] + test['Parch']


# In[ ]:


df['Title'] = df['Name'].apply(lambda x: x.split(sep = ',')[1].split(sep = ".")[0].strip())
test['Title'] = test['Name'].apply(lambda x: x.split(sep = ',')[1].split(sep = ".")[0].strip())


# In[ ]:


df.columns


# We will now classify our features thet we'll be using for further analysis as categorical, dummy categorical, continuous and dependant variable:

# In[ ]:


cat_list = ['Sex', 'FamSize', 'Pclass']
dummy_list = ['Embarked', 'Deck', 'Title']
cont_list = ['Age', 'Fare']
dep = 'Survived'


# In[ ]:


train_set = df[cat_list + cont_list + dummy_list + [dep]].copy()
test_set = test[cat_list + cont_list + dummy_list].copy()


# In order to transform features into numerical values, we transform the appropriate variables in the training set:

# In[ ]:


def process_df(df, is_train=True, train=train_set):
    for c in cat_list + cont_list + dummy_list:
        if is_train:
            if c in cat_list: df[c] = df[c].astype("category").cat.as_ordered()
        else:
            if c in cat_list: df[c] = df[c].astype(CategoricalDtype(train[c].cat.categories))
        if c in cont_list:
            df[c] = df[c].astype("float32")
        if c in dummy_list:
            cols = pd.get_dummies(df[c], prefix=c + "_")
            for col in cols.columns: df[col] = cols[col]
            df.drop(columns=c, inplace=True)


# In[ ]:


process_df(train_set)


# We do a similar transformation on the test set, making sure the category mapping remains the same as in the training set:

# In[ ]:


process_df(test_set, is_train=False)


# Then we transform the training set into a numerical representation:

# In[ ]:


train_num = train_set.copy()

for c in train_num.columns: train_num[c] = train_num[c].cat.codes if c in cat_list else train_num[c]


# Now, let's look at the full correlation matrix for the selected features:

# In[ ]:


plt.figure(figsize=(20, 20))
plt.title("Correlation table")
# sns.heatmap(train_num[[dep] + cat_list + cont_list].corr(), annot=True, cmap="seismic")
sns.heatmap(train_num.corr(), annot=True, cmap="seismic")
plt.tight_layout()
plt.show()


# In[ ]:


test_num = test_set.copy()

for c in test_num.columns: test_num[c] = test_num[c].cat.codes if c in cat_list else test_num[c]


# ## Modelling

# In[ ]:


def acc(targ, pred): return (targ == pred).mean()


# **Training / validation split**
# 
# Since the test set comprises ca. 30% of overall passangers, we will use a similar ratio to obtain a validation set:

# In[ ]:


r = 0.3
idxs = np.random.choice(np.arange(len(train_num)), size = int(len(train_num) * 0.3), replace=False)
idxs_mask = train_num.index.isin(idxs)


# In[ ]:


X_train = train_num.drop(columns='Survived')[~idxs_mask]
X_val = train_num.drop(columns='Survived')[idxs_mask]
y_train = train_num['Survived'][~idxs_mask]
y_val = train_num['Survived'][idxs_mask]


# We compare the distributions in the validation and test sets:

# In[ ]:


X_val.describe(include='all')


# In[ ]:


test_num.describe(include='all')


# The feature distributions appear to be reasonably similar.

# **Random Forest classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# In[ ]:


accurs = []
min = 1
max = 15
step = 1
for p in range(min, max, step):
    m = RandomForestClassifier(n_estimators=45, max_features=0.88, min_samples_leaf=p,
                              n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)
    preds = m.predict(X_val)
    print("-"*30, f'''
    n-estimators: {p}
    Training score: {m.score(X_train, y_train)*100:.2f}%
    Validation score: {m.score(X_val, y_val)*100:.2f}%
    Out-of-Bag score: {m.oob_score_*100:.2f}%
    Accuracy: {acc(y_val, preds)*100:.2f}%
    ''')
    accurs.append([p, acc(y_val, preds)])
accurs = np.array(accurs)
accurs[np.unravel_index(accurs[:, 1].argmax(), accurs[:, 1].shape)[0], :]


# In[ ]:


m = RandomForestClassifier(n_estimators=70, max_features=0.5, min_samples_leaf=1,
                          n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
preds = m.predict(X_val)
print("-"*30, f'''
Training score: {m.score(X_train, y_train)*100:.2f}%
Validation score: {m.score(X_val, y_val)*100:.2f}%
Out-of-Bag score: {m.oob_score_*100:.2f}%
Accuracy: {acc(y_val, preds)*100:.2f}%
''')


# In[ ]:


def cross_val(X, y, cv=10):
    accuracies = cross_val_score(estimator = m,
                                 X = X,
                                 y = y,
                                 cv = cv) # k: number of folds - typically 10
    print("Average accuracy:", round(accuracies.mean()*100,1),"%")
    print("Standard deviation:", round(accuracies.std()*100,1),"%")


# In[ ]:


cross_val(X_train, y_train)


# ## Analysing the results

# We'll look at the variable contribution to our overall accuracy:

# In[ ]:


accs = []
targ = m.score(X_train, y_train)
num_features = 15

for c in X_train.columns:
    X = X_train.copy()
    X[c] = X[[c]].sample(frac=1).set_index(X.index)[c]  # random shuffle of one column
    accs.append(targ - m.score(X, y_train))
    

FI = sorted([[c, float(a)] for c, a in zip(X.columns, accs)], key=lambda x: x[1], reverse=True)[:num_features]
pd.DataFrame({'Score loss': [FI[i][1] for i in range(len(FI))], 'Features': [FI[i][0] for i in range(len(FI))]}).set_index('Features').sort_values(by='Score loss', ascending=True).plot.barh()
plt.show()


# It looks like most variables do not add too much to the overall accuracy of the model. Let's reestimate based only on the top contributors:

# In[ ]:


top = 8
selected = [FI[i][0] for i in range(len(FI))][:top]
Xt = X_train[selected].copy()
Xv = X_val[selected].copy()
t = test_num[selected].copy()


# In[ ]:


m = RandomForestClassifier(n_estimators=70, max_features=0.5, min_samples_leaf=5,
                          n_jobs=-1, oob_score=True)
m.fit(Xt, y_train)
preds = m.predict(Xv)
print("-"*30, f'''
Training score: {m.score(Xt, y_train)*100:.2f}%
Validation score: {m.score(Xv, y_val)*100:.2f}%
Out-of-Bag score: {m.oob_score_*100:.2f}%
Accuracy: {acc(y_val, preds)*100:.2f}%
''')
cross_val(Xt, y_train)


# The accuracy is reasonably close to where it has been previously. Let's look at the variable contribution again:

# In[ ]:


accs = []
targ = m.score(Xt, y_train)

for c in Xt.columns:
    X = Xt.copy()
    X[c] = X[[c]].sample(frac=1).set_index(X.index)[c]  # random shuffle of one column
    accs.append(targ - m.score(X, y_train))
    
pd.DataFrame({'Score loss': accs}, index=X.columns).sort_values(by='Score loss', ascending=True).plot.barh()
plt.title('Feature Importance')
plt.show() 


# ## Submission

# In[ ]:


m.predict(t).sum()/t.shape[0]


# The predicted survival rate is in the same order of magnitude as the one observed in our training data.

# In[ ]:


my_submission = pd.DataFrame({'Survived': m.predict(t)}, index=t.index)
my_submission.to_csv('submission.csv')


# In[ ]:




