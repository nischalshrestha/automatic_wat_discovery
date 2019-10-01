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


# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV,  RandomizedSearchCV
pd.options.display.max_rows=999
pd.options.display.max_columns=999

#read supplied files and concat to simply processing
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train_size = train.shape[0]
df = pd.concat([train, test], axis=0)
df['Name'] = df['Name'].str.replace('\"','').str.strip()

#start creating fields
titlemap = {'Don': 1, 'Dona': 1, 'Mme': 5, 'Mlle': 1, 'Jonkheer': 1, 'Capt' :1, 'Col': 1, 'Major': 1, 'Countess': 1,  
            'Mr': 2, 'Dr': 3, 'Ms': 4, 'Mrs': 5, 'Miss': 6,  'Rev': 1, 'Master': 8, 'Sir': 1, 'Lady': 1}
df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
df['TitleCat'] = df['Title'].map(titlemap)
for atitle in ['Miss','Mr', 'Mrs', 'Master', 'Dr', 'Ms']:    
    df.loc[ (df['Age'].isnull()) & (df['Title'] == atitle), 'Age'] = df[ (df['Title'] == atitle) ]['Age'].median()

df['CabinCat'] = pd.Categorical.from_array(df.Cabin.fillna(0)).codes
df['EmbarkedCat'] = pd.Categorical.from_array(df.Embarked.fillna('C')).codes
df['Female'] = (df['Sex'] == 'female')
df.loc[ df.Fare.isnull(), 'Fare' ] = df[ df.Pclass==3 ].Fare.median()

df['FamilySize'] = df['SibSp'] + df['Parch']
df['NameLength'] = df.Name.fillna('').str.len()
df['NameLength'] = df.Name.fillna('').str.len()

# did an older relative survive (from in training set only - using test set data would be cheating...)
df['Surname'] = df['Name'].str.extract('([A-Za-z]+)\,', expand=False)
train['Surname'] = train['Name'].str.extract('([A-Za-z]+)\,', expand=False)
alive = train[ (train.Survived == 1) ]['Surname'].dropna().unique()
df['AliveRelative'] = (df['Surname'].isin(alive)) & (df.Age < 20)

# create train and test data
drop_columns = ['Ticket', 'Cabin', 'PassengerId', 'Name', 'Embarked', 'Sex', 'Title','Surname']

X_trainx = df.drop(drop_columns + ['Survived'], axis=1).iloc[:train_size]
X_train = StandardScaler().fit_transform(X_trainx)
y_train = df['Survived'].iloc[:train_size]
X_testx  = df.drop(drop_columns + ['Survived'], axis=1).iloc[train_size:]
X_test = StandardScaler().fit_transform(X_testx)

#create and run model
survived = df[ df['Survived'] == 1]['Survived'].count()  /  df['Survived'].count()
param_dist = {"C": np.linspace(1000, 15000, 100),
              "class_weight": [{0: 1-survived, 1: survived}, {0: 0.542, 1: 0.458}],
              'gamma': np.linspace(0.0021, 0.0025, 50),
              }
SVC_model = SVC()
#model = RandomizedSearchCV(SVC_model, param_distributions=param_dist, n_iter=1000, n_jobs=-1)
model = SVC(C=4360, gamma=0.0023, class_weight={0: 0.542, 1: 0.458} ) #0.818181818
#model = SVC(C=4350, gamma=0.0023, class_weight={0: 1-survived, 1: survived} ) #0.80622

model.fit(X_train, y_train)
#print (model.best_params_)
preds = model.predict(X_test).astype(int)


#generate predictions csv file
predictions = pd.DataFrame()
predictions['PassengerId'] = test['PassengerId']
predictions['Survived'] = preds
predictions.set_index('PassengerId', inplace=True, drop=True)
predictions.to_csv('titanic_predictions.csv')


from subprocess import check_output
print(check_output(["ls", "."]).decode("utf8"))

