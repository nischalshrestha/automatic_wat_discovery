#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().magic(u'matplotlib inline')


# **This is my first Kernel, traditionally based on famous Titanic dataset.**
# 
# Please be patient, but all advices are extremely welcom :) The main goal of this kernel is not to get 100% accuracy in predicting, but to learn in data pre-processing, different models tuning and validation.
# 
# The first part is related to importing numpy, pandas, etc, and loading input data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')


# Now let's explore our training data a little bit.

# In[ ]:


train.info()
train.describe()


# In[ ]:


train.head(10)


# Ok, obviously **PassengerId** and **Ticket** variables will be useless in our prediction and could be dropped off. Probably the first letter of **Cabin** (the name of deck or part of the ship) can tell us something but there are only *204* not NA values of *891* entries totally. So in my opinion it could be dropped as well.

# In[ ]:


del train['PassengerId']
del train['Ticket']
del train['Cabin']


# Now let's get useful info from the **Name**, as each name has some title like *Mr*, *Mrs*, *Master* and so on.

# In[ ]:


#Function to get title substring of the name string
def get_title(name):
    return ((name.split(','))[1]).split(' ')[1]

#And apply this function to get new variable 'Title'
train['Title'] = train['Name'].apply(func=get_title)

#Now we can remove 'Name' column
del train['Name']


# In[ ]:


train['Title'].value_counts()


# Ok, now let's decide which titles could be combined. 
# * "Mlle" (Mademoiselle) is unmarried woman - the same as "Miss". 
# * "Mme" (Madame) is married as well as "Mrs". 
# * "Ms." marital status is unknown so we'll join it to the majority, which is "Miss".
# * "Major.", "Col.", "Capt." - are all officers.
# * "Jonkheer", "Dr.", "Sir.", "Don.", "Lady." - are related to nobility. Not difficult to find out, that what has been left as "the" after get_title function is "the Countess", so belongs to nobility as well.
# * "Rev." stands for the Reverend - that's the matter of taste, it could be megred with nobility, but I preffer to leave clerics separately.
# *  "Master." means boy, and we will leave Mr. and Master separately.
# 
# So time to combine them.
# 

# In[ ]:


train.Title = train.Title.replace("Mlle.", "Miss.")
train.Title = train.Title.replace("Mme.", "Mrs.")
train.Title = train.Title.replace("Ms.", "Miss.")
train.Title = train.Title.replace("Major.", "Officer")
train.Title = train.Title.replace("Col.", "Officer")
train.Title = train.Title.replace("Capt.", "Officer")
train.Title = train.Title.replace("Jonkheer.", "Nobility")
train.Title = train.Title.replace("Dr.", "Nobility")
train.Title = train.Title.replace("Sir.", "Nobility")
train.Title = train.Title.replace("Don.", "Nobility")
train.Title = train.Title.replace("Lady.", "Nobility")
train.Title = train.Title.replace("the", "Nobility")
train['Title'].value_counts()


# There are only two missed **Embarked** values, so we will fill them manually with most frequent value which is "*S*".

# In[ ]:


train['Embarked'].value_counts()
train['Embarked'] = train['Embarked'].fillna(value="S")


# Now let's take closer look to the family values such as **SibSp** and **Parch**. 

# In[ ]:


sns.countplot(x="SibSp", data=train);
train["SibSp"].value_counts()


# In[ ]:


sns.countplot(x="Parch", data=train);
train["Parch"].value_counts()


# For both **SibSp** and **Parch** variables we can leave we can leave only 0, 1 and 'More' values. What does it mean? You are alone, or you have just one relative to care of, or more than 1 and it could be a problem to gather all family together in this case.. So now these variables can be treated as categorical.

# In[ ]:


train.SibSp[train.SibSp>1] = 2
train.Parch[train.Parch>1] = 2


# For **Age** we have *177* missing values, which is about 20% of the total quantity. I think we will try different ways to deal with this missing values and compare our results later on. So we will try:
# * Just to impute missed values with the mean age
# * To predict missed age based on other independent variables
# 
# But firstly let's play with visualization. Time to import seaborn. And we will start with **Age** variable exploring (just to decide how we will fill missed values)

# In[ ]:


#Boxplot of Age grouped by Survived
sns.boxplot( x="Survived", y="Age", data = train);


# Well, seems like no big difference in **Age** between **Survived** groups, but anyway we won't predict independent variable from dependent one. Let's check boxplots for **Age** by other independent categorical values

# In[ ]:


sns.boxplot( x="Sex", y="Age", data = train);


# In[ ]:


sns.boxplot( x="Pclass", y="Age", data = train);


# In[ ]:


sns.boxplot( x="Embarked", y="Age", data = train);


# In[ ]:


sns.boxplot( x="Title", y="Age", data = train);


# So we see correlation of **Age** and **Title**. And that is logically clear - Master is always young as the boys had this title. Miss is usually younger than Mrs. Officers are usually older than middle age. First class passengers are older. Male and Female groups have about the same age. And the port of embarkation does not actually affect the age.
# 
# 
# As below, **Age** and **Fare** don't look to be correlated
# 

# In[ ]:


sns.lmplot(x="Fare", y="Age", data=train);


# And once again all together:

# In[ ]:


train.corr()


# So based on all above let's fill missing **Age** values based on **Pclass** and **Title** values, and **SibSp** / **Parch** values as well. 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

def impute_age(data):
    temp_train = pd.DataFrame()
    temp_train['Pclass'] = data['Pclass']
    temp_train['Title'] = data['Title']
    temp_train['SibSp'] = data['SibSp']
    temp_train['Parch'] = data['Parch']
    temp_train['Age'] = data['Age']
    labelencoder_title = LabelEncoder()
    temp_train['Title'] = labelencoder_title.fit_transform(temp_train['Title'])

    train_with_age = temp_train[temp_train.Age.notnull()]

    y_age_imputing = train_with_age['Age'].values
    X_age_imputing = train_with_age.iloc[:, train_with_age.columns != 'Age'].values
    age_regressor = RandomForestRegressor()

    scores_age = cross_val_score(age_regressor, X_age_imputing, y_age_imputing, 
                                 scoring='neg_mean_absolute_error')
    print('Mean Absolute Error for Age value imputing prediction %2f' %(-1 * scores_age.mean()))

    age_regressor.fit(X_age_imputing, y_age_imputing)
    temp_train.Age[temp_train.Age.isnull()] = age_regressor.predict(temp_train.iloc[:, temp_train.columns != 'Age'].values)
    return temp_train.Age
    
train.Age = impute_age(train)
train.info()


# Finally, we've got all values fiiled. Now we will do dummy variables for our categories. and split our set.

# In[ ]:


train = pd.get_dummies(train, columns = ["Pclass", "Sex", "SibSp", "Parch", "Embarked", "Title"], drop_first=True)
X_train = train.iloc[:, train.columns != 'Survived'].values
y_train = train['Survived'].values


# So finally we can build and tune our model.

# In[ ]:


import xgboost as xgb
#First time on default parameters so we can tune it with Grid Search later
XGB_classifier = xgb.XGBClassifier()

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = XGB_classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()


# In[ ]:


# Applying Grid Search to find the best model and the best parameters
#from sklearn.model_selection import GridSearchCV
#parameters = [{'max_depth': [5, 6, 7, 8], 'learning_rate': [0.018, 0.02, 0.022],
#              'n_estimators': [150, 155, 160, 165, 170], 'objective': ['binary:logistic'], 'booster': ['gbtree'],
#              'gamma':[0.002, 0.003, 0.004, 0.005, 0.006], 'scale_pos_weight':[0.9, 1, 1.1]}]
#grid_search = GridSearchCV(estimator = XGB_classifier,
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv = 10)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#print(best_accuracy)


# In[ ]:


#best_parameters = grid_search.best_params_
#print(best_parameters)


# I put GridSearch under the comment, as it is not quick process, so after playing with it several times I've got accuracy 0.842873176207.
# And the best parameters are the following
# {'booster': 'gbtree', 'gamma': 0.003, 'learning_rate': 0.02, 'max_depth': 6, 'n_estimators': 165, 'objective': 'binary:logistic', 'scale_pos_weight': 1}
# So let's rebuild our model with the parameters we've got.

# In[ ]:


XGB_classifier = xgb.XGBClassifier(booster = 'gbtree', gamma = 0.003, learning_rate = 0.02,
                             max_depth = 6, n_estimators = 165, objective = 'binary:logistic',
                             scale_pos_weight = 1)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = XGB_classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

XGB_classifier.fit(X_train, y_train)


# Now let's prepare test set and submit our result. Later on I will try other algoritms like Random Forest and Kernel SVM.

# In[ ]:


#So let's repeat exactly the same steps with test set as we did with train one
test.info()


# In[ ]:


PassengerId = test['PassengerId'] #we will need it for submitting result later
del test['PassengerId']
del test['Ticket']
del test['Cabin']
test['Title'] = test['Name'].apply(func=get_title)
del test['Name']
test['Title'].value_counts()


# In[ ]:


test.Title = test.Title.replace("Ms.", "Miss.")
test.Title = test.Title.replace("Col.", "Officer")
test.Title = test.Title.replace("Dr.", "Nobility")
test.Title = test.Title.replace("Dona.", "Nobility")
test['Title'].value_counts()


# In[ ]:


test.SibSp[test.SibSp>1] = 2
test.Parch[test.Parch>1] = 2


# We have missing value for **Fare**, but just one, so I'll fill it with mean.

# In[ ]:


test['Fare'] = test['Fare'].fillna(np.mean(test['Fare']))
test.info()


# In[ ]:


test.Age = impute_age(test)
test.info()


# In[ ]:


test = pd.get_dummies(test, columns = ["Pclass", "Sex", "SibSp", "Parch", "Embarked", "Title"], drop_first=True)
X_test = test.values


# In[ ]:


y_pred_XGB = XGB_classifier.predict(X_test)


# Let's submit it!

# In[ ]:


my_submission_XGB = pd.DataFrame({'PassengerId': PassengerId, 'Survived': y_pred_XGB})
my_submission_XGB.to_csv('submission_XGB.csv', index=False)


# And we've got 0.77990 score. Not bad for the first Kernel. But let's play with model and try other algoritmes. Firstly, Random Forest.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier()
RF_accuracies = cross_val_score(estimator = RF_classifier, X = X_train, y = y_train, cv = 10)
RF_accuracies.mean()


# In[ ]:


#RF_parameters = [{'n_estimators': [60, 70, 80, 90], 
#                  'max_features':['sqrt', 'log2', 'auto'],
#                  'max_depth':[20, 30, 40, 50], 'min_samples_split':[2, 3, 4],
#                  'min_samples_leaf':[2, 3, 4, 5]}]
#RF_grid_search = GridSearchCV(estimator = RF_classifier,
#                           param_grid = RF_parameters,
#                           scoring = 'accuracy',
#                           cv = 10)
#RF_grid_search = RF_grid_search.fit(X_train, y_train)
#RF_best_accuracy = RF_grid_search.best_score_
#print(RF_best_accuracy)


# In[ ]:


#RF_best_parameters = RF_grid_search.best_params_
#print(RF_best_parameters)


# For Random Forest maximum accuracy I've got is 0.843995510662. With the following parameters
# {'max_depth': 40, 'max_features': 'log2', 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 80}
# It is worth a try to submit result with Random Forest as well, because even if accuracy on train set is lower than for XGBoost, the difference is very small and we still can have better score on test set.

# In[ ]:


RF_classifier = RandomForestClassifier(max_depth=40, max_features='log2',min_samples_leaf=3,
                                      min_samples_split=2, n_estimators=80)
RF_classifier.fit(X_train, y_train)
y_pred_RF = RF_classifier.predict(X_test)
my_submission_RF = pd.DataFrame({'PassengerId': PassengerId, 'Survived': y_pred_RF})
my_submission_RF.to_csv('submission_rf.csv', index=False)


# And we've got 0.77033 which is lower than for XGBoost. Finally let's try SVC. But now we need to scale our features.

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
from sklearn.svm import SVC
SVC_classifier = SVC()
SVC_accuracies = cross_val_score(estimator = SVC_classifier, X = X_train, y = y_train, cv = 10)
SVC_accuracies.mean()


# In[ ]:


#SVC_parameters = [{'C':[5, 20, 50, 80], 'kernel':['linear']},
#                {'C': [40, 50, 60], 'gamma': [0.005, 0.01, 0.03, 0.05], 'kernel': ['rbf']}]
#SVC_grid_search = GridSearchCV(estimator = SVC_classifier,
#                           param_grid = SVC_parameters,
#                           scoring = 'accuracy',
#                           cv = 10)
#SVC_grid_search = SVC_grid_search.fit(X_train, y_train)
#SVC_best_accuracy = SVC_grid_search.best_score_
#print(SVC_best_accuracy)


# In[ ]:


#SVC_best_parameters = SVC_grid_search.best_params_
#print(SVC_best_parameters)


# The best accuracy for Kernel SVM I've got is 0.826038159371 with the parameters {'C': 50, 'gamma': 0.01, 'kernel': 'rbf'}. It's less than for previous models, but let's check the score anyway.

# In[ ]:


X_test = sc.transform(X_test)
SVC_classifier = SVC(C=50, gamma=0.01, kernel='rbf')
SVC_classifier.fit(X_train, y_train)
y_pred_SVC = SVC_classifier.predict(X_test)
my_submission_SVC = pd.DataFrame({'PassengerId': PassengerId, 'Survived': y_pred_SVC})
my_submission_SVC.to_csv('submission_svc.csv', index=False)


# Still worse than XGBoost - 0.76555. But anyway all three models are quite close to each other.

# So let's make small final conclusion. The best result we've got was 0.77990 with XGBoost. Of course, there are many ways still to impove it. I see for example the following:
# * to not drop Cabin variable, and try to impute it as well
# * to impute Age different way (probably based on other variable/model)
# * to pre-process and group Name and Title variable with differrent approach
# * to leave SibSp and Parch variable as they are, or group them other way (for, example - to create Family binary variable)
# * to use other classification algoritms (Naive Bayes or ANN, for example) or continue to play with tuning of used
# * and more and more
# But I don't want to make this Titanic to be a project of all my life, and is going to move to another challenges.
