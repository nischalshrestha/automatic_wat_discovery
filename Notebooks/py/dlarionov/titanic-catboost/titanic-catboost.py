#!/usr/bin/env python
# coding: utf-8

# ## Loading data

# In[1]:


import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool, cv

# paramaters tuning tool
import hyperopt


# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_size = train.shape[0] # 891
test_size = test.shape[0] # 418

data = pd.concat([train, test])


# ## Feture engineering

# In[3]:


data['Surname'] = data['Name'].str.extract('([A-Za-z]+)\,', expand=False)

data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Don', 'Capt', 'Col', 'Major', 'Sir', 'Jonkheer', 
                                       #'Rev', 
                                       #'Dr'
                                      ], 'Honored')
data['Title'] = data['Title'].replace(['Lady', 'Dona', 'Mme', 'Countess'], 'Mrs')
data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')

data['FamilySize'] = (data['SibSp'] + data['Parch']).astype(int)
data['IsAlone'] = (data['SibSp'] + data['Parch'] == 0).astype(int)

'''
age_ref = data.groupby('Title').Age.mean()
data = data.assign(
    Age = data.apply(lambda r: r.Age if pd.notnull(r.Age) else age_ref[r.Title] , axis=1)
)
del age_ref

data['Room'] = data['Cabin'].str.slice(1,5).str.extract('([0-9]+)', expand=False).fillna(-999).astype(float) # non cat feature
data['RoomBand'] = pd.cut(data['Room'], 5, labels=range(5)).astype(int)

data.loc[data.Cabin=='T', 'Cabin'] = None # 1 item
data['Deck'] = data['Cabin'].str.slice(0,1).fillna('Unknown')
data['Cabin'] = data['Cabin'].fillna('Unknown')

data.loc[data.Ticket=='LINE', 'Ticket'] = 'LINE0' # 4 items
data['Odd'] = data['Ticket'].str.slice(-1).astype(int).map(lambda x: x % 2 == 0).astype(int)
'''


# In[4]:


data['Embarked'] = data['Embarked'].fillna('S') 
data['Cabin'] = data['Cabin'].fillna('Undefined')
data['Fare'] = data['Fare'].fillna(13.30) # Pclass=3 mean


# ## Training

# In[5]:


cols = [
    'Pclass',
    #'Name',
    'Sex',
    'Age',    
    'SibSp',
    'Parch',
    #'Ticket',
    'Fare',
    'Cabin',
    'Embarked', 
    'Title',
    'Surname',
    #'Deck',
    #'Room',
    #'RoomBand',
    #'Odd',    
    'FamilySize',
    'IsAlone'    
]
X_train = data[:train_size][cols]
Y_train = data[:train_size]['Survived'].astype(int)
X_test = data[train_size:][cols]

categorical_features_indices = [0,1,6,7,8,9,11] #np.where(X_train.dtypes != np.float)[0]
X_train.head()


# In[6]:


train_pool = Pool(X_train, Y_train, cat_features=categorical_features_indices)


# Parameters tuning

# In[12]:


def hyperopt_objective(params):
    model = CatBoostClassifier(
        l2_leaf_reg=int(params['l2_leaf_reg']),
        #learning_rate=params['learning_rate'],
        depth=params['depth'],
        iterations=500,
        eval_metric='Accuracy',
        od_type='Iter',
        od_wait=40,
        random_seed=42,
        logging_level='Silent',
        allow_writing_files=False
    )
    
    cv_data = cv(
        train_pool,
        model.get_params()
    )
    best_accuracy = np.max(cv_data['test-Accuracy-mean'])    
    
    print(params, best_accuracy)
    return 1 - best_accuracy # as hyperopt minimises

params_space = {
    'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 2, 1),
    'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
    'depth': hyperopt.hp.choice('depth', [3,4,5,6,8]),
}

trials = hyperopt.Trials()

best = hyperopt.fmin(
    hyperopt_objective,
    space=params_space,
    algo=hyperopt.tpe.suggest,
    max_evals=50,
    trials=trials
)

print(best)


# In[13]:


model = CatBoostClassifier(
    l2_leaf_reg=int(best['l2_leaf_reg']),
    learning_rate=best['learning_rate'],
    depth=best['depth'],
    iterations=500,
    eval_metric='Accuracy',
    od_type='Iter',
    od_wait=40,
    random_seed=42,
    logging_level='Silent',
    allow_writing_files=False
)

cv_data = cv(
    train_pool,
    model.get_params(),
    plot=False
)

print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
    np.max(cv_data['test-Accuracy-mean']), 
    cv_data['test-Accuracy-std'][cv_data['test-Accuracy-mean'].idxmax(axis=0)],
    cv_data['test-Accuracy-mean'].idxmax(axis=0)
))
print('Precise validation accuracy score: {}'.format(np.max(cv_data['test-Accuracy-mean'])))

model.fit(train_pool);
model.score(X_train, Y_train)


# In[14]:


feature_importances = model.get_feature_importance(train_pool)
feature_names = X_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))


# ## Submission

# In[ ]:


Y_pred = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": data[train_size:]["PassengerId"], 
    "Survived": Y_pred.astype(int)
})
submission.to_csv('submission.csv', index=False)

