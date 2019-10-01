#!/usr/bin/env python
# coding: utf-8

# Machine Learning Attempt on classifying using XGBoost

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


print ('Training dataset row count is', len(train))
print ('Test dataset row count is', len(test))
print ('Missing values in Training and Test Data is seen in')
print (train.count())
print (test.count())
print ('A preview of the dataset in Training')
print (train.head())


# In[ ]:


print (train.Cabin.value_counts())
train['Cabin'] = data['Cabin'].fillna('U')
    dataset['Cabin'] = dataset.Cabin.str.extract('([A-Za-z])', expand=False)


# In[ ]:


# Replacing the missing values
# train - Age, Cabin, Embarked
# test - Age, Fare, Cabin

# 1. Replace the Age in Train
tr_avage = train.Age.mean()
tr_sdage = train.Age.std()
tr_misage = train.Age.isnull().sum()
rand_age = np.random.randint(tr_avage - tr_sdage, tr_avage + tr_sdage, size=tr_misage)
train['Age'][np.isnan(train['Age'])] = rand_age
train['Age'] = train['Age'].astype(int)

# 2. Replace the Age in Test
te_avage = test.Age.mean()
te_sdage = test.Age.std()
te_misage = test.Age.isnull().sum()
rand_age = np.random.randint(te_avage - te_sdage, te_avage + te_sdage, size=te_misage)
test['Age'][np.isnan(test['Age'])] = rand_age
test['Age'] = test['Age'].astype(int)


# In[ ]:


# 3. Replace the Embarked in Train
# Distribution of Embarked in train S-644, C-168, Q-77
train['Embarked'] = train['Embarked'].fillna('S')

# 4. Treat the cabin for both test and train as a new varibale "Is_Cabin"
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# 5. Replace the Fare in test with a median value
med =  test.Fare.median()
test['Fare'] =  test['Fare'].fillna(med)


# In[ ]:





# In[ ]:


# Create new Features - 1. FamilySize 2. Solo traveller 3. Age bucket

# 1. FamilySize
train['FamilySize'] = train['SibSp'] + train['Parch']
test['FamilySize'] = test['SibSp'] + test['Parch']

# 2. Create New Feature Solo Traveller
train['Solo'] = train['FamilySize'].apply(lambda x: 0 if x>0 else 1)
test['Solo'] = test['FamilySize'].apply(lambda x: 0 if x>0 else 1)

# For Train
train['Age'] = train['Age'].astype(int)
test['Age'] = test['Age'].astype(int)

def Age(row):
    if row['Age'] < 16:
        return 'VY'
    elif row['Age'] < 32:
        return 'Y'
    elif row['Age'] < 48:
        return 'M'
    elif row['Age'] < 64:
        return 'O'
    else:
        return 'VO'
    
train['CategoricalAge'] = train.apply(lambda row: Age(row), axis=1)
test['CategoricalAge'] = test.apply(lambda row: Age(row), axis=1)


# In[ ]:


# Final Feature Selection Droping the ones which may look not necessary
drop_list = ['PassengerId', 'Name', 'Cabin', 'Ticket', 'Age']
ftrain = train.drop(drop_list, axis = 1)
ftest = test.drop(drop_list, axis = 1)


# In[ ]:


# labelling the Dataset before passing to a model
# 1. Map the variable Sex
ftrain['Sex'] = ftrain['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
ftest['Sex'] = ftest['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
# 2. Map the variable Embarked
ftrain['Embarked'] = ftrain['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
ftest['Embarked'] = ftest['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
# 3. Map the Categorical Age
ftrain['CategoricalAge'] = ftrain['CategoricalAge'].map( {'VY': 0, 'Y': 1, 'M': 2, 'O': 3, 'VO': 4} ).astype(int)
ftest['CategoricalAge'] = ftest['CategoricalAge'].map( {'VY': 0, 'Y': 1, 'M': 2, 'O': 3, 'VO': 4} ).astype(int)


# In[ ]:


# Creating the X and Y for both Train and Test
y_train = ftrain['Survived'].ravel()
ftrain = ftrain.drop(['Survived'], axis=1)
x_train = ftrain.values # Creates an array of the train data
x_test = ftest.values # Creats an array of the test data


# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


# In[ ]:


model = XGBClassifier()
n_estimators = [110, 120]
max_depth = [2, 4, 6, 8]
print(max_depth)
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(x_train, y_train)
#model.fit(x_train, label_encoded_y)


# In[ ]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
# plot results
scores = np.array(means).reshape(len(max_depth), len(n_estimators))
for i, value in enumerate(max_depth):
    plt.plot(n_estimators, scores[i], label='depth: ' + str(value))
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('Log Loss')
plt.show()


# The above graph indicates the least log_loss for the data set can be achieved using the parameters for the XGBoost as 'max_depth': 2, 'n_estimators': 120. We will now train and fit our dataset using the identified parameters.

# In[ ]:


clf2 = XGBClassifier(max_depth=2, n_estimators=120)
clf2.fit(x_train, y_train)
pred2 = clf2.predict(x_test)


# In[ ]:


final_sub2 = pd.DataFrame({ 'PassengerId': test.PassengerId,
                            'Survived': pred2 })
final_sub2.to_csv("Sub4.csv", index=False)


# In[ ]:




