#!/usr/bin/env python
# coding: utf-8

# # An exploration of using some classifiers to predict survival
# 
# I'm new to Kaggle and have tried to apply some recommended techniques. However, I seem to be obtaining dismal results (unless I use an earlier approach which got me to the leaderboard).
# 
# Perhaps some grand masters can point me in the right direction?

# In[ ]:


import pandas as pd
import numpy as np
import uuid
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.svm import SVC
import re
from sklearn.metrics import classification_report

def dictize(data_frame, full_data, columns):
    names = set([])
    for column in columns:
        names = names.union(list(full_data[column].unique()))
    names = list(enumerate(names))
    dictionary = { name: i for i, name in names }

    for column in columns:
        data_frame[column] = data_frame[column].map( dictionary )
    return dictionary

def grid_search(clf_rf, param_grid, X_train, y_train, X_test, y_test, cv=3):
    print ('Training Classifier...')

    clf = GridSearchCV(clf_rf, param_grid, cv=cv)

    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    return clf.best_params_
    
def accuracy(y_pred, y_true):
    correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct = correct + 1
    return float(correct) / float(len(y_pred))


# Let do some exploratory data analysis to find out more about the data

# In[ ]:


train_df = pd.read_csv("../input/train.csv")

embarked_dict = { 'S': 0, 'C': 1, 'Q': 2}

# Replace Sex with 1 for male and 0 for female
train_df["Sex"] = train_df["Sex"].map(lambda x: 1 if x == "male" else 0)
train_df["Embarked"].fillna("S", inplace=True)
train_df["Embarked"] = train_df["Embarked"].map(embarked_dict)
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1


test_df = pd.read_csv("../input/test.csv")
# Replace Sex with 1 for male and 0 for female
test_df["Sex"] = train_df["Sex"].map(lambda x: 1 if x == "male" else 0)
test_df["Embarked"].fillna("S", inplace=True)
test_df["Embarked"] = test_df["Embarked"].map(embarked_dict)
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

combined_df = train_df.drop('Survived', axis=1).append(test_df)

# Let's look at the combined test and train data frame
combined_df.describe()


# Age and sex should be important factors in survival, after all, a ship is evacuated with "Women and Children first". So, we need to classify passengers as children or women.
# 
# Number of Siblings and Spouses (SibSp), number of parents and children (Parch), the title of the person (for example "Master", "Miss", "Mr", etc.), the Age and the Sex should be an important feature for this classification.
# 
# However, age is missing for a lot of rows, so we will need to impute it.
# 
# Only one Fare is missing so we can impute that with the median fare.
# 
# Below, we add the missing Fare entry, add another column for title:

# In[ ]:


# 1. Add the missing fare entry in the frames
median_fare = combined_df['Fare'].dropna().median()
train_df.Fare.fillna(median_fare, inplace=True)
test_df.Fare.fillna(median_fare, inplace=True)
combined_df.Fare.fillna(median_fare, inplace=True)

# Functions that returns the title from a name. All the name in the dataset has the format "Surname, Title. Name"
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

train_df["Title"] = train_df.Name.map(get_title)
test_df["Title"] = train_df.Name.map(get_title)
combined_df["Title"] = train_df.Name.map(get_title)

combined_df["Title"].value_counts()


# Next, we can standardise the less frequent titles to Mr, Mrs, Miss, Master, Dr, Rev, and Col:

# In[ ]:


title_mapping = {
    "Mr": 1,        # A man
    "Miss": 2,      # An unmarried lady
    "Mrs": 3,       # An married lady
    "Master": 4,    # A young man
    "Dr": 5,        # A doctor
    "Rev": 6,       # A priest
    "Major": 7,     # An army man
    "Col": 7,       # An army man
    "Mlle": 2,      # An unmarried lady
    "Mme": 3,       # An married lady
    "Don": 1,       # A man
    "Dona":3,       # A married lady
    "Lady": 3,      # An married lady
    "Countess": 3,  # An married lady
    "Jonkheer": 3,  # An married lady
    "Sir": 1,       # A man
    "Capt": 7,      # An army man
    "Ms": 3}        # A divorced lady
train_df["Title"] = train_df['Title'].map(title_mapping)
test_df["Title"] = test_df['Title'].map(title_mapping)
combined_df["Title"] = combined_df['Title'].map(title_mapping)

combined_df.describe()


# Let us now impute age based on the 1046 rows with age present in the train and test combined frame:

# In[ ]:


et_regressor = ExtraTreesRegressor(n_estimators=200)
rf_regressor = RandomForestRegressor(n_estimators=200)

predictors = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Title', 'Fare', 'Title']

X_train = combined_df[combined_df['Age'].notnull()][predictors]
Y_train = combined_df[combined_df['Age'].notnull()]['Age']
X_test = combined_df.loc[combined_df.Age.isnull(), predictors]

et_regressor.fit(X_train, np.ravel(Y_train))
rf_regressor.fit(X_train, np.ravel(Y_train))
predictions_et = et_regressor.predict(X_test)
predictions_rf = rf_regressor.predict(X_test)

predictions = (predictions_et + predictions_rf) / 2

imputed_frame = combined_df.copy()
imputed_frame.loc[combined_df.Age.isnull(), ['Age']] = predictions


# In[ ]:


# Clear any previous plots
plt.clf()
# Let's look at the distribution of the imputed ages
#fig, axes = plt.subplots(nrows=1, ncols=2)
predictions_frame = pd.DataFrame(predictions)
predictions_frame.columns = ['Age']

plt.subplot(1, 2, 1)
predictions_frame['Age'].plot(kind='hist', title='Imputed Age Distribution')
plt.subplot(1, 2, 2)
combined_df[combined_df['Age'].notnull()]['Age'].plot(kind='hist', title='Actual Age Distribution')
plt.show()


# We can see from the above two graphs, that the imputation matches the existing distibution of ages. 
# 
# Next, we merge the the imputed values into the train and test dataframe and add column features to detect if a person is a child or is a mother.

# In[ ]:


train_df_imputed = pd.merge(train_df, imputed_frame[['PassengerId', 'Age']], on='PassengerId')
train_df_imputed.drop('Age_x', axis=1, inplace=True)
train_df_imputed.rename(columns={'Age_y': 'Age'}, inplace=True)

test_df_imputed = pd.merge(test_df, imputed_frame[['PassengerId', 'Age']], on='PassengerId')
test_df_imputed.drop('Age_x', axis=1, inplace=True)
test_df_imputed.rename(columns={'Age_y': 'Age'}, inplace=True)

combined_df_imputed = pd.merge(combined_df, imputed_frame[['PassengerId', 'Age']], on='PassengerId')
combined_df_imputed.drop('Age_x', axis=1, inplace=True)
combined_df_imputed.rename(columns={'Age_y': 'Age'}, inplace=True)

def is_mother(row):
    index, item = row
    age = item['Age']
    title = item['Title']
    sex = item['Sex']
    parch = item['Parch']
    if age > 18 and title != title_mapping["Miss"] and sex == 0 and parch > 1:
        return 1
    else:
        return 0
    
def is_child(row):
    index, item = row
    age = item['Age']
    if age < 18:
        return 1
    else:
        return 0
    
train_df_imputed['IsMother'] = [is_mother(row) for row in train_df_imputed.iterrows()]
test_df_imputed['IsMother'] = [is_mother(row) for row in test_df_imputed.iterrows()]
combined_df_imputed['IsMother'] = [is_mother(row) for row in combined_df_imputed.iterrows()]

train_df_imputed['IsChild'] = [is_child(row) for row in train_df_imputed.iterrows()]
test_df_imputed['IsChild'] = [is_child(row) for row in test_df_imputed.iterrows()]
combined_df_imputed['IsChild'] = [is_child(row) for row in combined_df_imputed.iterrows()]

combined_df_imputed.describe()


# Let us now look at the proportion of males to females who survived:

# In[ ]:


survivors = train_df_imputed[train_df_imputed['Survived'] == 1]
male_survivors = survivors[survivors['Sex'] == 1]
female_survivors = survivors[survivors['Sex'] == 0]

dead = train_df_imputed[train_df_imputed['Survived'] == 0]
males_dead = dead[dead['Sex'] == 1]
females_dead = dead[dead['Sex'] == 0]

print ("Male:Female survival ratio is {}:{}".format(male_survivors['PassengerId'].count(), female_survivors['PassengerId'].count()))
print ("Male:Female death ratio is {}:{}".format(males_dead['PassengerId'].count(), females_dead['PassengerId'].count()))


# Does family size affect survival?

# In[ ]:


family_sizes_survived = pd.DataFrame(train_df_imputed[train_df_imputed['Survived'] == 1]['FamilySize'].value_counts()).reset_index()
family_sizes_survived.columns = ['Family Size', 'Survived']

family_sizes_perished = pd.DataFrame(train_df_imputed[train_df_imputed['Survived'] == 0]['FamilySize'].value_counts()).reset_index()
family_sizes_perished.columns = ['Family Size', 'Perished']

family_size = pd.merge(family_sizes_survived, family_sizes_perished, on="Family Size")

family_size.sort_values(by='Family Size', axis=0, inplace=True)

plt.clf()
family_size.plot(kind='bar', x='Family Size', title='Family Size Analysis')
plt.show()


# Single people have a greater tendency to perish. Let us categorise family size as:
# 
# - Size 1 = 1
# - 1 < Size < 5 = 2
# - Size >= 5 = 3

# In[ ]:


def size_categorize(size):
    if size == 1:
        return 1
    elif size > 1 and size < 5:
        return 2
    else:
        return 3

train_df_imputed['FamilySizeCategory'] = train_df_imputed['FamilySize'].map(size_categorize)
test_df_imputed['FamilySizeCategory'] = test_df_imputed['FamilySize'].map(size_categorize)
combined_df_imputed['FamilySizeCategory'] = combined_df_imputed['FamilySize'].map(size_categorize)


# We add a feature/predictor for surname to investigate whether families survived or died together

# In[ ]:


def get_family_id(row):
    index, item = row
    pclass = item['Pclass']
    parch = item['Parch']
    sibsp = item['SibSp']
    name = item['Name']
    
    family_size = parch + sibsp + 1
    
    if family_size > 1:
        return name.split(',')[0].lower() + "_" + str(pclass) + "_" + str(family_size)
    else:
        return name.split(',')[0].lower() + "_" + str(pclass) + "_" + str(uuid.uuid4())
    
combined_df_imputed['FamilyId'] = [get_family_id(row) for row in combined_df_imputed.iterrows()]
test_df_imputed['FamilyId'] = np.nan
train_df_imputed['FamilyId'] = np.nan

family_id_dict = dictize(train_df_imputed, combined_df_imputed, ['FamilyId'])

test_df_with_family = pd.merge(test_df_imputed, combined_df_imputed[['PassengerId', 'FamilyId']], on='PassengerId')
test_df_with_family.drop('FamilyId_x', axis=1, inplace=True)
test_df_with_family.rename(columns={'FamilyId_y': 'FamilyId'}, inplace=True)

train_df_with_family = pd.merge(train_df_imputed, combined_df_imputed[['PassengerId', 'FamilyId']], on='PassengerId')
train_df_with_family.drop('FamilyId_x', axis=1, inplace=True)
train_df_with_family.rename(columns={'FamilyId_y': 'FamilyId'}, inplace=True)

family_id_dict = dictize(test_df_with_family, combined_df_imputed, ['FamilyId'])
family_id_dict = dictize(train_df_with_family, combined_df_imputed, ['FamilyId'])

# If a family member (other than self) survived, it is likely that more than one family member survived
survived_families = list(train_df_with_family[train_df_with_family['Survived'] == 1]['FamilyId'])
survived_passengers = list(train_df_with_family[train_df_with_family['Survived'] == 1]['PassengerId'])

def has_family_member_survived(row):
    index, item = row
    if item['PassengerId'] not in survived_passengers and item['FamilyId'] in survived_families:
        return 1
    else:
        return 0

train_df_with_family['FamilyMemberSurvived'] = [has_family_member_survived(row) for row in train_df_with_family.iterrows()]
test_df_with_family['FamilyMemberSurvived'] = [has_family_member_survived(row) for row in test_df_with_family.iterrows()]
combined_df_imputed['FamilyMemberSurvived'] = [has_family_member_survived(row) for row in combined_df_imputed.iterrows()]

combined_df_imputed.describe()


# Define variables for testing.

# In[ ]:


predictors = ['Pclass', 'Sex', 'SibSp', 'Parch',
                'Fare', 'Embarked', 'Title', 'IsMother', 'IsChild',
                'FamilySizeCategory', 'FamilyMemberSurvived', 'FamilyId']

train = train_df_with_family.copy()
test = test_df_with_family.copy()

y_true = train['Survived'].values
X_data = train[predictors].values


# Let's look for good features:

# In[ ]:


# Identify best features
# All the predictors in the model

# For ET - {'min_samples_split': 2, 'n_estimators': 200, 'min_samples_leaf': 4}
# For RF - {'min_samples_split': 2, 'n_estimators': 300, 'min_samples_leaf': 8}

ft = ExtraTreesClassifier(n_jobs=4, min_samples_split=2, n_estimators=200, min_samples_leaf=4, random_state=3)
ft.fit(X_data, y_true)

ftr = RandomForestClassifier(n_jobs=4, min_samples_split=2, n_estimators=300, min_samples_leaf=8, random_state=1)
ftr.fit(X_data, y_true)

importances_et = ft.feature_importances_
importance_et_df = pd.DataFrame(importances_et).reset_index()
importance_et_df.columns = ['Feature', 'Importance']
importance_et_df['Feature'] = importance_et_df['Feature'].map(lambda x: predictors[x])
importance_et_df.sort_values(by='Importance', axis=0, inplace=True, ascending = False)

importances_rf = ftr.feature_importances_
importance_rf_df = pd.DataFrame(importances_rf).reset_index()
importance_rf_df.columns = ['Feature', 'Importance']
importance_rf_df['Feature'] = importance_rf_df['Feature'].map(lambda x: predictors[x])
importance_rf_df.sort_values(by='Importance', axis=0, inplace=True, ascending = False)

plt.clf()
plt.figure()
importance_et_df.plot(kind='bar', x='Feature', title='Feature Analysis (ExtraTrees)')
importance_rf_df.plot(kind='bar', x='Feature', title='Feature Analysis (RandomForest)')
plt.show()


# In[ ]:


# Perform feature selection
selector = SelectKBest(f_classif, k=10)
selector.fit(train[predictors], train["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()


# Next, we optimise the classifiers for the training data set

# In[ ]:


top_features_et = ['Sex', 'Pclass', 'Fare', 'FamilyMemberSurvived', 'IsChild']
top_features_rf = ['Title', 'Fare', 'Pclass', 'FamilyId', 'FamilySizeCategory']
top_features_svm = ['Pclass', 'Sex', 'Fare', 'Title', 'FamilyMemberSurvived']

X_data_et = train[top_features_et].values
X_data_rf = train[top_features_rf].values
X_data_svm = preprocessing.scale(train[top_features_svm].values)

# Divide records in training and testing sets.
X_train_et, X_test_et, y_train_et, y_test_et = train_test_split(X_data_et, y_true, test_size=0.3, random_state=3, stratify=y_true)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_data_rf, y_true, test_size=0.3, random_state=3, stratify=y_true)
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_data_svm, y_true, test_size=0.3, random_state=3, stratify=y_true)

print ("Tagged records were split into training and test sets")

param_grid = [
      {'n_estimators': [1000], 
       'min_samples_split': [2, 4, 6, 8],
       'min_samples_leaf': [2, 4, 6, 8]
      }
    ]

clf_et = ExtraTreesClassifier(random_state=1, n_jobs=4)
clf_rf = RandomForestClassifier(random_state=1, n_jobs=4)
clf_svm = SVC(random_state=1, probability=True)

optimal_et = grid_search(clf_et, param_grid, X_train_et, y_train_et, X_test_et, y_test_et, cv=5)
optimal_rf = grid_search(clf_rf, param_grid, X_train_rf, y_train_rf, X_test_rf, y_test_rf, cv=5)
clf_svm.fit(X_train_svm, y_train_svm)


# Let's look at the Area Under ROC as a metric

# In[ ]:


# Fit to training data
clf_et.fit(X_train_et, y_train_et)
clf_rf.fit(X_train_rf, y_train_rf)

# Plot the results.
colors = ['b', 'r', 'g']
classifiers = ['ExtraTrees', 'RandomForest', 'SVC']
plt.figure(figsize=(20,10))
for i, cl in enumerate([clf_et, clf_rf, clf_svm]):
    if i == 0:
        y_test_roc = y_test_et
        probas_ = cl.predict_proba(X_test_et)
    elif i ==1:
        y_test_roc = y_test_rf
        probas_ = cl.predict_proba(X_test_rf)
    else:
        y_test_roc = y_test_svm
        probas_ = cl.predict_proba(X_test_svm)
    fpr, tpr, thresholds = roc_curve(y_test_roc, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label=classifiers[i]+' (AUC = %0.2f)' % (roc_auc))
    
plt.plot([0, 1], [0, 1], '--', color=colors[i], label='Random (AUC = 0.50)')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])   
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.axes().set_aspect(1)
plt.legend(loc="lower right")
plt.show()


# Create the submission frame ad get the accuracy

# In[ ]:


# Get accuracy on the training data
#{'min_samples_split': 2, 'n_estimators': 300, 'min_samples_leaf': 4}
clf_et = ExtraTreesClassifier(random_state=1, n_jobs=-1, min_samples_split=optimal_et["min_samples_split"], min_samples_leaf=optimal_et["min_samples_leaf"], n_estimators=optimal_et["n_estimators"])
clf_et.fit(X_data_et, y_true)

clf_rf = RandomForestClassifier(random_state=1, n_jobs=-1, min_samples_split=optimal_rf["min_samples_split"], min_samples_leaf=optimal_rf["min_samples_leaf"], n_estimators=optimal_rf["n_estimators"])
clf_rf.fit(X_data_rf, y_true)

clf_svm = SVC(random_state=1, probability=True)
clf_svm.fit(X_data_svm, y_true)

y_pred_et = clf_et.predict(X_data_et)
y_pred_rf = clf_rf.predict(X_data_rf)
y_pred_svm = clf_svm.predict(X_data_svm)

y_pred = (y_pred_rf + y_pred_svm) / 2.0

y_pred = [1 if y > 0.5 else 0 for y in y_pred]

print ("Accuracy {}".format(accuracy(y_pred, y_true)))

passenger_ids = test["PassengerId"]

X_test_et = test[top_features_et].values
y_pred_et = clf_et.predict(X_test_et)

X_test_rf = test[top_features_rf].values
y_pred_rf = clf_rf.predict(X_test_rf)

X_test_svm = preprocessing.scale(test[top_features_svm].values)
y_pred_svm = clf_svm.predict(X_test_svm)

y_pred = (y_pred_rf  + y_pred_svm) / 2.0

y_pred = [1 if y > 0.5 else 0 for y in y_pred]

submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": y_pred
    })
submission.head()


# In[ ]:


print ("{} predicted survivors out of {}".format(submission["Survived"].sum(), submission["Survived"].count()))


# In[ ]:


submission.to_csv('titanic.csv', index=False)


# This strategy was adapted from the excellent analysis presented in R:
# https://www.kaggle.com/tylerph3/titanic/exploring-survival-on-the-titanic/run/416227
