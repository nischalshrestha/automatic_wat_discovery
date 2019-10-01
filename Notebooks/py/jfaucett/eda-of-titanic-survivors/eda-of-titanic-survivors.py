#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings('ignore')


# ### EDA
# 
# We're going to start by exploring the dataset in detail to try and figure out what are likely features which will help us predict who survived and who did not.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../input/train.csv")
df.head(2)


# In[ ]:


print('size: {}'.format(len(df)))
df.info()


# In[ ]:


display(np.corrcoef(df['Parch'], df['Survived']))
df['Parch'].value_counts().plot(kind='bar')


# In[ ]:





# In[ ]:


from sklearn.preprocessing import LabelEncoder

def normalize_embarked(df):
    embarked = df['Embarked'].astype('category').values.copy()
    embarked[df['Embarked'].isnull()] = 'S'

    # convert categorical to numeric encodings
    le = LabelEncoder()
    le.fit(embarked)
    return le.transform(embarked)
    
    
plt.hist(normalize_embarked(df))
plt.show()
np.corrcoef(normalize_embarked(df), df['Survived'])


# 1. No null values which is nice
# 
# ### Pclass
# 
# The Pclass is the Ticket class. It should have the values 
# 
# 1 = 1st, 2 = 2nd, 3 = 3rd

# In[ ]:


print("P Classes : {}".format(np.unique(df['Pclass'].values)))
display(np.corrcoef(df['Pclass'], df['Survived']))
plt.hist(df['Pclass'], bins=3)


# ### Pclass : Summary
# 
# 1. The Ticket Class is negatively correlated with survival.
# 2. First class ticket holders where more likely to survive than 2nd or third. and 2nd more so than 3rd.
# 3. The majority of ticket holders were 3rd class.
# 
# ### Sex 
# 
# 1. I would expect more females to survive than males, that is there should be a negative correlation b/t sex and survival. (When males are 1, females are 0)
# 2. We need to convert Male / Female values into binary 1 / 0.

# In[ ]:


from sklearn.preprocessing import LabelEncoder

def normalize_sex(df):
    sex_label_encoder = LabelEncoder()
    sex_label_encoder.fit(df['Sex'].values)
    sex_array = sex_label_encoder.transform(df['Sex'].values)
    return sex_array
    
    
plt.hist(normalize_sex(df), bins=2)
plt.show()
print(np.corrcoef(sex_array, df['Survived']))


# ### Sex Summary
# 
# 1. Women are much more likely to survive than men.
# 2. Men make up almost 2x as many samples as women.
# 
# ### Age
# 1. I would expect a negative correlation b/t age and survival, since I would think both men and women would sacrifice themselves in order to save children.
# 
# Let's see if that hypothesis holds up.

# In[ ]:


def normalize_age(df):
    age  = df['Age'].values.copy()
    
    # find mean of age for all non-null values
    age_mean = np.mean(age[~df['Age'].isnull()]) 
    
    # Set unknown values to the mean age
    age[df['Age'].isnull()] = age_mean
    
    age_max = np.max(age)
    age_min = np.min(age)
    
    return (age - age_min) / (age_max - age_min)

def age_info(df):
    display(df['Age'].describe())
    
    age_df = df[~df['Age'].isnull()]
    print('% of non-null values: {}'.format(len(age_df) / len(df)))
    plt.hist(age_df['Age'])
    plt.show()

    age_10 = age_df[age_df['Age'] <= 10]
    print(np.corrcoef(age_10['Age'], age_10['Survived']))
    print(len(age_10[age_10['Survived'] == 1]) / len(age_10))

    age_20 = age_df[np.logical_and(age_df['Age'] > 10, age_df['Age'] <= 20)]
    print(np.corrcoef(age_20['Age'], age_20['Survived']))
    print(len(age_20[age_20['Survived'] == 1]) / len(age_20))

    age_40 = age_df[np.logical_and(age_df['Age'] > 20, age_df['Age'] <= 40)]
    print(np.corrcoef(age_40['Age'], age_40['Survived']))
    print(len(age_40[age_40['Survived'] == 1]) / len(age_40))

    age_50 = age_df[np.logical_and(age_df['Age'] > 40, age_df['Age'] <= 50)]
    print(np.corrcoef(age_50['Age'], age_50['Survived']))
    print(len(age_50[age_50['Survived'] == 1]) / len(age_50))

    age_60 = age_df[np.logical_and(age_df['Age'] > 50, age_df['Age'] <= 60)]
    print(np.corrcoef(age_60['Age'], age_60['Survived']))
    print(len(age_60[age_60['Survived'] == 1]) / len(age_60))

    age_80 = age_df[np.logical_and(age_df['Age'] > 60, age_df['Age'] <= 80)]
    print(np.corrcoef(age_80['Age'], age_80['Survived']))
    print(len(age_80[age_80['Survived'] == 1]) / len(age_80))
    
plt.hist(normalize_age(df), bins=4)
plt.show()
np.corrcoef(normalize_age(df), df['Survived'])


# ### Age : Summary
# 
# It looks like there are 3 ages here which are actually important.
# 
# 1. $0 \leq x \leq 10$ - this age group was much more likely to survive as they got younger (i.e. 3 year old more likely to survive than a 6 year old) And 60% of this group survived.
# 2. $10 < x \leq 60$ - roughly 40% of this age group survived and there wasn't a strong correlation anywhere.
# 3. $60 < x \leq 80$ - only 22% of this age group survived and there wasn't a strong correlation b/t age and survival in any direction.

# In[ ]:



def normalize_fares(df):
    orig_fares = df['Fare'].values.copy()
    
    # find mean of age for all non-null values
    fares_mean = np.mean(orig_fares[~df['Fare'].isnull()])
    
    # Set unknown values to the mean age
    orig_fares[df['Fare'].isnull()] = fares_mean
    
    maxf = np.max(orig_fares)
    minf = np.min(orig_fares)
    
    return (orig_fares - minf) / (maxf - minf)
    
plt.hist(df['Fare'], bins=32)
plt.show()


np.corrcoef(df['Fare'].values, df['Survived'].values)


# In[ ]:


def build_features(df):
    n = len(df)
    # Features
    age_data = normalize_age(df)
    fare_data = normalize_fares(df)
    sex_data = normalize_sex(df)
    ticket_class_data = df['Pclass'].values.copy()
    parch_data = df['Parch'].values.copy()
    embarked_data = normalize_embarked(df)
    sibsp_data = df['SibSp'].values.copy()
    
    # Targets
    survival_data = df['Survived'].values.copy() if 'Survived' in df else []
    
    features = []
    for index in range(n):
        # age, fare, sex, ticket_class, parch (# parents + children on board), the port of embarkation, # of siblings + spouses
        feature_vec = [age_data[index],fare_data[index], sex_data[index], ticket_class_data[index], parch_data[index], embarked_data[index]]
        features.append(feature_vec)
    
    features = np.array(features)
    assert len(features) == n
    
    return features, survival_data


def test_train_split(df, train=0.8, test=0.2):
    np.random.seed(42)
    
    n = len(df)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    train_end_idx = int(train * n)
    
    features, survival_data = build_features(df)
    
    features_final = features[indices]
    survival_data_final = survival_data[indices]
    
    x_train = features_final[:train_end_idx]
    y_train = survival_data_final[:train_end_idx]
    x_test  = features_final[train_end_idx:]
    y_test  = survival_data_final[train_end_idx:]
    
    return x_train, y_train, x_test, y_test
    
x_train, y_train, x_test, y_test = test_train_split(df)


# In[ ]:


from sklearn import svm

def build_classifier(x,y):
    clf = svm.SVC()
    clf.fit(x,y)
    return clf

clf = build_classifier(x_train,y_train)
y_preds = clf.predict(x_test)


# In[ ]:


from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

def print_stats(y_true, y_pred):
    print("Accuracy: {}".format(accuracy_score(y_true, y_pred)))
    
print_stats(y_test, y_preds)


# In[ ]:


df_test = pd.read_csv("../input/test.csv")
test_features, _survival_data = build_features(df_test)
test_preds = clf.predict(test_features)
results = pd.DataFrame({ 'PassengerId' : df_test['PassengerId'].values, 'Survived' : test_preds })
results.head(10)


# In[ ]:


results.to_csv("./predictions1.csv", index=False)


# In[ ]:


test_features[0]


# ## What about Deep Learning Models?
# 
# The question now is whether or not a Deep Learning Model can give better results?

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

n_cols = len(test_features[0])

early_stopping_monitor = EarlyStopping(patience=2)

def build_model():
    model = Sequential()
    model.add(Dense(512, activation='relu',input_shape=(n_cols,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    
    return model

model = build_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train,y_train, validation_data=(x_test, y_test), callbacks=[early_stopping_monitor], epochs=10, batch_size=16)

