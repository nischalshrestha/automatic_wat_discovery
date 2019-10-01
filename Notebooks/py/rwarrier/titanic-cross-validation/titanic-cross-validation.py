#!/usr/bin/env python
# coding: utf-8

# Titanic - Cross Validation

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
get_ipython().magic(u'matplotlib inline')
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import mode
import matplotlib.pyplot as plt 


# Load the test and Train datasets into Pandas Dataframe. There are missing values in both Test and Train sets. Once we train the model using the train set, we should be able to use the model to predict the test set. For that we need to make sure the structure of the train and test datasets are the same and all the features that we are creating is available in both the datasets. It's better to combine the test and train datasets and then split it before training the model.

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
full = pd.concat([train, test], keys=['train','test'])
#full = pd.concat([train, test])
full.head()


# Split the Name column and create 2 new columns - LastName and Title. The Title column will come in handy later to guess the missing age values of the passengers

# In[ ]:


full['LastName'] = full.Name.str.split(',').apply(lambda x: x[0]).str.strip()
full['Title'] = full.Name.str.split("[\,\.]").apply(lambda x: x[1]).str.strip()


# print(full.Title.value_counts())
# 
# Mr              757;
# Miss            260;
# Mrs             197;
# Master           61;
# Dr                8;
# Rev               8;
# Col               4;
# Ms                2;
# Mlle              2;
# Major             2;
# Dona              1;
# Lady              1;
# Sir               1;
# Mme               1;
# the Countess      1;
# Don               1;
# Jonkheer          1;
# Capt              1;
# the Countess      1;
# Jonkheer          1
# 
# As you can see, there are too many titles. We'll try and consolidate the titles to the commonly used ones. We'll consolidate the titles into the main 4 categories - Mr, Mrs, Miss and Master

# In[ ]:


##if the title is Dr and the sex is female, we'll update the Title as Miss
full.loc[(full.Title == 'Dr') & (full.Sex == 'female'), 'Title'] = 'Mrs'

##if the title is in any of the following, we'll update the Title as Miss
full.loc[full.Title.isin(['Lady','Mme','the Countess','Dona']), 'Title'] = 'Mrs'

##if the title is in any of the following, we'll update the Title as Miss
full.loc[full.Title.isin(['Ms','Mlle']), 'Title'] = 'Miss'

##if the title is Dr and the sex is female, we'll update the Title as Mr
full.loc[(full.Title == 'Dr') & (full.Sex == 'male'), 'Title'] = 'Mr'

##if the title is Rev and the sex is male, we'll update the Title as Mr
full.loc[(full.Title == 'Rev') & (full.Sex == 'male'), 'Title'] = 'Mr'

## Setting all the Rev, Col, Major, Capt, Sir --> Mr
full.loc[full.Title.isin(['Rev','Col','Major','Capt','Sir','Don','Jonkheer']) & (full.Sex == 'male'), 'Title'] = 'Mr'


# Now we will define a new column - PassengerType to categorize the passengers into Adults and Children

# In[ ]:


def passenger_type (row):
   if row['Age'] < 2 :
      return 'Infant'
   elif (row['Age'] >= 2 and row['Age'] < 12):
      return 'Child'
   elif (row['Age'] >= 12 and row['Age'] < 18):
      return 'Youth'
   elif (row['Age'] >= 18 and row['Age'] < 65):
      return 'Adult'
   elif row['Age'] >= 65:
      return 'Senior'
   elif row['Title'] == 'Master':
      return 'Child'
   elif row['Title'] == 'Miss':
      return 'Child'
   elif row['Title'] == 'Mr' or row['Title'] == 'Mrs':
      return 'Adult'
   else:
      return 'Unknown'


# In[ ]:


full['PassengerType'] = full.apply(lambda row: passenger_type(row),axis=1)


# full['PassengerType'].value_counts()
# 
# Adult     1083;
# Child      128;
# Youth       63;
# Infant      22;
# Senior      13;

# In[ ]:


#factorize the PassengerType to make it numeric values
full['PassengerType'] = pd.factorize(full['PassengerType'])[0]
#full['PassengerType'].value_counts()
#full = pd.get_dummies(full, columns=['PassengerType'])


# In[ ]:


#factorize the PassengerType to make it numeric values
full['Title'] = pd.factorize(full['Title'])[0]
#full = pd.get_dummies(full, columns=['Title'])
#full['Title'].value_counts()


# #### There is one Fare that is null. We'll update that to the median fare for the Class and Embarked Combination

# In[ ]:


full.loc[full.Fare.isnull()]
full.loc[full.Fare.isnull(), 'Fare'] = full.loc[(full.Embarked == 'S') & (full.Pclass == 3),'Fare'].median()


# Now let's check for nulls in the Embarked column. 
# 
# full.loc[full.Embarked.isnull()].shape
# 
# (2, 15)
# 
# There are 2 rows where Embarked is null. Both the passengers are in first class.
# 
# print(full.groupby(['Pclass', 'Embarked'])['Fare'].median())
# 
# Pclass  Embarked
# 1       C           76.7292;
#         Q           90.0000;
#         S           52.0000;
# 2       C           15.3146;
#         Q           12.3500;
#      (   S           15.3750;
# 3       C            7.8958;
#         Q            7.7500;
#         S            8.0500;
# 
# The median fare for passengers embarked from "C" and have a First class ticket is \$77. Close to the \$80 that the two paasengers paid. Based on the median fare, we'll assume that they both Embarked from Port C

# In[ ]:


full.loc[full.Embarked.isnull(), 'Embarked'] = 'C'


# We'll now create a bin for the Fare ranges. splitting into 6 groups seems to be a reasonable split.

# # Divide all fares into quartiles
# full['Fare_bin'] = pd.qcut(full['Fare'], 8)
# # qcut() creates a new variable that identifies the quartile range, but we can't use the string so either
# # factorize or create dummies from the result
# #full['Fare_bin_id'] = pd.factorize(full['Fare_bin'])[0]
# full = pd.get_dummies(full, columns=['Fare_bin'],  prefix='Fare')

# Create a family size variable to see if there's any reason to believe that smaller families had a better chance of survival

# In[ ]:


#Creating new family_size column
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1


# In[ ]:


#The fare for the 2 rows is 80. Let's see which class and Embarked combination gives the closest Median Fare to 80
print(full.groupby(['Pclass', 'Embarked'])['Fare'].median())

#Boxplot to show the median values for different groups. (1,c) has a median value of 80
medianprops = dict(linestyle='-', linewidth=1, color='k')
full.boxplot(column='Fare',by=['Pclass','Embarked'], medianprops=medianprops, showmeans=False, showfliers=False)


# In[ ]:


#full = pd.get_dummies(full, columns=['Embarked'])
full['Embarked'] = pd.factorize(full['Embarked'])[0]
full['Gender'] = pd.factorize(full['Sex'])[0]
full.info()


# In[ ]:


full.rename(columns={"Fare_[0, 7.75]": "Fare_1"
                                ,"Fare_(7.75, 7.896]": "Fare_2"
                                ,"Fare_(7.896, 9.844]": "Fare_3"
                                ,"Fare_(9.844, 14.454]": "Fare_4"
                                ,"Fare_(14.454, 24.15]": "Fare_5"
                                ,"Fare_(24.15, 31.275]": "Fare_6"
                                ,"Fare_(31.275, 69.55]": "Fare_7"
                                ,"Fare_(69.55, 512.329]": "Fare_8"}, inplace=True)
full.info()


# full.rename(columns={"Fare_[0, 7.775]": "Fare_1"
#                                 ,"Fare_(7.775, 8.662]": "Fare_2"
#                                 ,"Fare_(8.662, 14.454]": "Fare_3"
#                                 ,"Fare_(14.454, 26]": "Fare_4"
#                                 ,"Fare_(26, 53.1]": "Fare_5"
#                                 ,"Fare_(53.1, 512.329]": "Fare_6"}, inplace=True)
# full.info()

# In[ ]:


AgeNotNull = full.loc[full.Age.notnull(),:].copy()
AgeNull = full.loc[full.Age.isnull(),:].copy()
#full.head()


# In[ ]:


cols = full.columns.tolist()
cols


# In[ ]:


#Fill the null values in Age column with -1
#full['Age'] = full['Age'].fillna(-1)
feature_cols = ['Fare', 'Parch', 'SibSp', 'Pclass', 'FamilySize', 'Title','PassengerType', 'Gender']
#feature_cols = ['Pclass', 'FamilySize', 'PassengerType_Adult', 'PassengerType_Child', 'PassengerType_Infant', 
#                'PassengerType_Senior', 'PassengerType_Youth', 'Gender', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
#                'Fare_1', 'Fare_2', 'Fare_3', 'Fare_4', 'Fare_5', 'Fare_6', 'Fare_7', 'Fare_8', 
#                'Embarked_C', 'Embarked_Q', 'Embarked_S']
X = AgeNotNull[feature_cols]
y = AgeNotNull.Age

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)

#lm = svm.SVR()
#lm.fit(X, y)
# print intercept and coefficients
#print(lm.intercept_)
#print(lm.coef_)
# pair the feature names with the coefficients
#zip(feature_cols, lm.coef_)


# In[ ]:


# predict for a new observation
p = lm.predict(AgeNotNull[feature_cols])

# Now we can constuct a vector of errors
err = abs(p-y)
#print(y[:10])
#print(p[:10])
# Let's see the error on the first 10 predictions
print (err[:10])


# In[ ]:


# Dot product of error vector with itself gives us the sum of squared errors
total_error = np.dot(err,err)
print(total_error)
# Compute RMSE
rmse = np.sqrt(total_error/len(p))
print(rmse)


# In[ ]:


# predict for a new observation
p1 = lm.predict(AgeNull[feature_cols])
print(p1[:10])
p1.shape
AgeNull.shape


# In[ ]:


AgeSer = full.loc[full.Age.notnull(),'Age']
plt.hist(AgeSer)
plt.ylabel("Count")
plt.xlabel("Age")
plt.show()


# In[ ]:


full.loc[full.Age.isnull(), 'Age'] = p1


# In[ ]:


full.loc[full.Age.isnull(), 'Age'] = p1
#plt.hist(full.Age)
#plt.ylabel("Count")
#plt.xlabel("Age")
#plt.show()
AgeNotNull['Age'].plot.kde()
AgeSer.plot.kde()
plt.show()


# In[ ]:


# Plot outputs
get_ipython().magic(u'matplotlib inline')
import pylab as pl
pl.plot(p, y,'ro')
pl.plot([0,50],[0,50], 'g-')
pl.xlabel('predicted')
pl.ylabel('real')
pl.show()


# In[ ]:


# predict for a new observation
output = lm.predict(AgeNotNull[feature_cols])

# calculate the R-squared
lm.score(X, y)
#confusion_matrix(AgeNotNull.Age.astype(int), output.astype(int))
accuracy_score(AgeNotNull.Age.astype(int), output.astype(int))

print(AgeNotNull.Age.astype(int).values)
print(output.astype(int))

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((output - AgeNotNull.Age) ** 2))


# ### Scikit-learn - Training the model
# We now build a pipeline to enable us to first impute the mean value of the column Age on the portion of the training data 
# we are considering, and second, assess the performance of our tuning parameters.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model, datasets


# In[ ]:


train = full.loc['train']
test = full.loc['test']


# In[ ]:


y = train.loc[:,'Survived']
#X = train.loc[:,['PassengerId', 'Age', 'Pclass', 'PassengerType_Adult', 'PassengerType_Child', 'PassengerType_Infant', 
#                 'PassengerType_Senior', 'PassengerType_Youth', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 
#                 'Fare_1', 'Fare_2', 'Fare_3', 'Fare_4', 'Fare_5', 'Fare_6', 'Fare_7', 'Fare_8', 'FamilySize', 
#                 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Gender']]
X = train.loc[:,['PassengerId','Age','Fare', 'Pclass','Title','PassengerType','FamilySize','Embarked','Gender']]


# In[ ]:


train_data = train.values
train_X = X.values
train_y = y.values


# In[ ]:


from time import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.30, random_state=22)


# Using K-Fold Cross Validation

# In[ ]:


k_fold = KFold(len(train_y), n_folds=10, shuffle=True, random_state=0)
#clf = RandomForestClassifier(n_estimators=est, min_samples_split=min_samples)
clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, min_samples_split=2, criterion='entropy', 
                               max_depth=None, bootstrap=True, max_features=9)
scoring = 'accuracy'
results = cross_val_score(clf, train_X, train_y, cv=k_fold, n_jobs=1, scoring=scoring)
results


# In[ ]:


est_range = range(75, 120)
print(est_range)
# use a full grid over all parameters
#param_grid = {"max_depth": [3, None],
#              "max_features": [1, 5, 10],
#              "min_samples_split": [1, 3, 5, 7, 9, 11],
#              "min_samples_leaf": [5, 10, 15],
#              "bootstrap": [True, False],
#              "criterion": ["gini", "entropy"]}
# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1,3,5,7,9],
              "min_samples_split": [2,3,4,5,6,7,8,9],
              "min_samples_leaf": [5,10,15,20,25],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
param_grid


# In[ ]:


# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(train_X, train_y)


# In[ ]:


print(grid_search.best_score_)
print(grid_search.best_params_)


# grid_search.best_score_
# grid_search.best_params_

# def grid_scores_to_df(grid_scores):
#     """
#     Convert a sklearn.grid_search.GridSearchCV.grid_scores_ attribute to a tidy
#     pandas DataFrame where each row is a hyperparameter-fold combinatination.
#     """
#     rows = list()
#     for grid_score in grid_scores:
#         for fold, score in enumerate(grid_score.cv_validation_scores):
#             row = grid_score.parameters.copy()
#             row['fold'] = fold
#             row['score'] = score
#             rows.append(row)
#     df = pd.DataFrame(rows)
#     return df

# k_fold = KFold(len(train_y), n_folds=10, shuffle=True, random_state=0)
# 
# est_range = range(93, 102)
# min_samples_range = range(8, 12)
# auc_values = {}
# result = {}
# for est in est_range:
#     for min_samples in min_samples_range:
#         scores = []
#         clf = RandomForestClassifier(n_estimators=est, min_samples_split=min_samples)
#         scoring = 'roc_auc'
#         results = cross_val_score(clf, train_X, train_y, cv=k_fold, n_jobs=1, scoring=scoring)
#         scores.append(results.mean())
#         scores.append(results.std())
#         #print(scores)
#         result[min_samples] = scores
#         print(est, min_samples, results.mean(), results.std())
#     auc_values[est] = result
# 
#         
# #        #print("n_estimatore:" + str(est) + "min_samples_split:" + str(min_samples) + " AUC: %.3f (%.3f)" % (results.mean(), results.std()))
# #scoring = 'accuracy'
# #scoring = 'roc_auc'
# #results = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
# #results = cross_val_score(clf, train_X, train_y, cv=k_fold, n_jobs=1, scoring=scoring)
# #print (cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1))
# #print(clf.score)
# #print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))
# #print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
# #print(auc_values)

# In[ ]:


#model = RandomForestClassifier(n_estimators = 1000)
#model = RandomForestClassifier(n_estimators=1000, min_samples_split=10)
#model = RandomForestClassifier(n_estimators=100, min_samples_leaf=15, min_samples_split=3, criterion='gini', 
#                               max_depth=3, bootstrap=False, max_features=10)
#model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, min_samples_split=2, criterion='entropy', 
#                               max_depth=None, bootstrap=True, max_features=9)
#model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, min_samples_split=8, criterion='gini', 
#                               max_depth=3, bootstrap=True, max_features=21)
model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, min_samples_split=5, criterion='gini', 
                               max_depth=3, bootstrap=True, max_features=7)
#model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
#model = linear_model.LogisticRegression(C=1e5)
#model = model.fit(X_train,y_train)
model = model.fit(train_X, train_y)


# In[ ]:


from sklearn import svm
model = svm.SVC()
model.fit(train_X, train_y) 


# In[ ]:


#output = model.predict(train_X)
y_pred = model.predict(X_test)

#result = np.c_[y.astype(int), output.astype(int)]
#train_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
tn, fp, fn, tp = confusion_matrix(y_test.astype(int), y_pred.astype(int)).ravel()
print(confusion_matrix(y_test.astype(int), y_pred.astype(int)))
print('accuracy: ' + str(accuracy_score(y_test, y_pred)))
#fpr, tpr, thresholds = roc_curve(y_test.astype(int), y_pred.astype(int), pos_label=2)
sensitivity = tp/(fp+tp)
print('sensitivity: ' + str(sensitivity))
specificity = tn/(fn+tn)
print('specificity: ' + str(specificity))


# ### Scikit-learn - Making predictions

# In[ ]:


#test_X = test.loc[:,['PassengerId','Age','Fare', 'Pclass','Title','PassengerType','Fare_bin_id','FamilySize','Embarked_Id','Gender']].values
#test_X = test.loc[:,['PassengerId', 'Age', 'Pclass', 'PassengerType_Adult', 'PassengerType_Child', 'PassengerType_Infant', 
#                 'PassengerType_Senior', 'PassengerType_Youth', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 
#                 'Fare_1', 'Fare_2', 'Fare_3', 'Fare_4', 'Fare_5', 'Fare_6', 'Fare_7', 'Fare_8', 'FamilySize', 
#                 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Gender']].values
test_X = test.loc[:,['PassengerId','Age','Fare', 'Pclass','Title','PassengerType','FamilySize','Embarked','Gender']].values
#test_X = test[test.columns.tolist()[1:]].values()
output = model.predict(test_X)


# ### Pandas - Preparing for submission

# In[ ]:


result = np.c_[test_X[:,0].astype(int), output.astype(int)]

df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('../input/submission606.csv', index=False)

