#!/usr/bin/env python
# coding: utf-8

# # Environment Setup

# In[ ]:


# Check who is the user running Jupyter.
who_am_i = get_ipython().getoutput(u'whoami')

# Define our data base path.
base_path_data = '../input' if who_am_i[0] == 'root' else '../../data'


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir(base_path_data))

# Any results you write to the current directory are saved as output.


# # Data Set Loading

# In[ ]:


df = pd.read_csv(base_path_data + '/train.csv', sep=",", header=0, encoding='utf-8')
df.shape


# # Statistics

# In[ ]:


df.info()


# In[ ]:


print('\nStats')
print(df.describe())


# In[ ]:


# Check which columns have missing data.
print('\nMissing values')
print(df.isnull().any())


# In[ ]:


print('Column types')
print(df.dtypes)


# In[ ]:


# Copied from article: https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation
def num_missing(x):
    return sum(x.isnull())

def print_missing_values(df, axis):
    print("Missing values per %s:" % ('column' if axis == 0 else 'row'))
    print(df.apply(num_missing, axis=axis)[:df.shape[1]]) # axis=0 to apply on each column

# Applying per column
print_missing_values(df, 0)

# Applying per row:
print_missing_values(df, 1)


# In[ ]:


{'Survived':df.query('Survived == 1').count()[0], 'Did not':df.query('Survived == 0').count()[0]}


# In[ ]:


# Check people age under 1.
df.query('Age < 1')


# # Visualization of Data

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import seaborn as sns
sns.set(style="darkgrid")

# Remove outliers.
# df = df[df.Fare < 300]

# http://seaborn.pydata.org/tutorial/relational.html
sns.relplot(x="Age", y="Fare", hue="Sex", data=df) # size="Fare", sizes=(0, 100), 
sns.relplot(x="Age", y="Fare", hue="Sex", col="Embarked", palette="ch:r=-.5,l=.75", data=df) # size="Fare", sizes=(0, 100), 
# plt.title('Fare per age')
plt.show()


# In[ ]:


import seaborn as sns
sns.set(style="darkgrid")

# Check the correlation between features.
corr = df.corr()
sns.heatmap(corr, cmap=sns.diverging_palette(20, 50, as_cmap=True), square=True, annot=True, linecolor='k', linewidths=.5)
plt.show()


# In[ ]:


# Group by age and count to see the distribution.
age_2_count = df.groupby('Age')['PassengerId'].count().reset_index(name="Count")

# Transpose for the sake of visibility.
age_2_count.loc[:15,].transpose()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# Set the plot.
sns.scatterplot(x="Age", y="Count", hue="Count", palette="ch:r=-.5,l=.75", data=age_2_count) # size="Fare", sizes=(0, 100), 

plt.title('Number of persons per age')
plt.show()


# # Data Set Preparation (Pre-processing)

# In[ ]:


df_tmp = df.copy(deep=True)


# ## Feature Engineering

# In[ ]:


import re

def extract_title(name):
#     print(name)
    m = re.search(".*,((\s*\w)+\.).*", name)
    if not m:
        print(name)
    return m.groups(0)[0].strip()

# print(extract_title('Pain, Dr. Alfred').strip())

# Find passenger with Dr title
# print(df[df['Name'].str.contains('Dr')])

# Title needs to be engineered using values from the test set or use it differently.
# For now, leave it out.
# df_tmp['Title'] = df_tmp['Name'].apply(extract_title)

df_tmp.head()


# In[ ]:


sns.boxplot(x="Sex", y="Age", hue='Pclass', palette=["m", "g"], data=df_tmp)
plt.show()


# In[ ]:


sns.boxplot(x="Survived", y="Age", palette=["m", "g"], data=df_tmp)
plt.show()


# ## Imputation of Missing Values

# In[ ]:


print_missing_values(df_tmp, 0)


# In[ ]:


# One method to show entries with NA values for Embarked column.
# nans = lambda df: df[df.isnull().any(axis=1)]
# nans(df_raw.loc[:, ['Embarked']])

# Another method to show entries, not working
# print(df_raw[df_raw['Embarked'].apply(np.isnan)])

# Preferred method to show entries where Embarked is null, NaN, etc.
print(df_tmp.query('Age != Age or Age == Age').loc[:, ['Name', 'Parch', 'SibSp', 'Age']].iloc[:15,])
print('\n' + str(df_tmp.isnull().any()))

# Show the most frequent value in the features.
df_tmp['Age'].mode().values[0]


# In[ ]:


from sklearn.impute import SimpleImputer
class DataPreparator:
    def impute(self, df, df_for_fitting=None, strategy='mean', column=None):
        """
        Impute the data frame using a strategy on a column.

        Parameters
        ----------
        df: DataFrame: the data frame to transform with the fitted value.
        df_for_fit: None or the data frame to fit the Imputer. If None then df is used for fitting.
        strategy: a string for the strategy name defined from sklearn.preprocessing.Imputer.
        column: a string for the name of the column to apply the imputation.

        Returns
        -------
        DataFrame: the 'df' with a new column called 'column'_imputed instead of the 'column'.

        """
        # Define the imputer working on columns. 
        imp = SimpleImputer(missing_values=np.nan, strategy=strategy)

        # Get the dataframe used to fit the imputer.
        df_fit = df_for_fitting if df_for_fitting is None else df

        # Column extractor
        extract_column_df = lambda df: np.array([df[column]]).transpose()

        # Fit the imputer with a DF of shape (*, 1).
        df_ = extract_column_df(df_fit)
        model = imp.fit(df_)

        # Transform the df of shape (*, 1).
        df_ = extract_column_df(df)
        res = pd.DataFrame(model.transform(df_))
        res.columns = [column]

        # Make a copy to add the new column and remove the old one.
        df_tmp = df.drop(column, axis=1)
        new_column = column + '_imputed'
        df_tmp[new_column] = np.array(res[column])

        return df_tmp

data_prep = DataPreparator()


# In[ ]:


# Proceed to impute the Age column.
# X_aug = X_train2 # pd.concat([X, X_train.loc[:, 'Age']], axis=1)
# print(X_aug['Age'])

# from sklearn.impute import SimpleImputer
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# cc = 'Age'
# aaa = np.array([X_train2[cc]]).transpose()
# print(aaa.shape)
# model_imp2 = imp.fit(aaa)
# model_imp2.transform(np.array([X_test[cc]]).transpose())

# print(X_train.query('Age != Age').loc[:, ['PassengerId','Age']].iloc[:10, :])
df_tmp = data_prep.impute(df_tmp, df_tmp, column='Age')
df_tmp = data_prep.impute(df_tmp, df_tmp, column='Cabin', strategy='most_frequent')
df_tmp = data_prep.impute(df_tmp, df_tmp, column='Embarked', strategy='most_frequent')

print(df.shape, df_tmp.shape)

print_missing_values(df_tmp, 0)


# ## Features Selection

# In[ ]:


# First keep the columns of interest.

def select_features(df, features_to_keep=['Pclass', 'Sex', 'Fare', 'Age_imputed']):
    return df.loc[:, features_to_keep]

# Add the function as a class method to the DataPreparator.
#     features_to_keep = list(set(df.columns) - set(['PassengerId', 'Survived', 'Name', 'SibSp', 'Ticket', 'Cabin_imputed']))

DataPreparator.select_features = lambda self, df, features_to_keep=['Pclass', 'Sex', 'Fare', 'Age_imputed']: select_features(df, features_to_keep)


# In[ ]:


df_tmp = data_prep.select_features(df_tmp)

# Check the data.
print_missing_values(df_tmp, 0)
print(df_tmp.dtypes)


# ## One-hot Encoding of Categorical Features

# In[ ]:


def get_one_hot(df, features=None):
    """
    Does the one-hot vectorization of features.
    """
    X = df.loc[:, features]

    # Use one-hot encoding for categorical data.
    X_dummies = pd.get_dummies(X, columns=features, dtype=np.uint8)
    X = df.drop(features, axis=1)
    return pd.concat([X, X_dummies], axis=1)

# Add the one-hot method to the DataPreparator class.
DataPreparator.get_one_hot = lambda self, df, features=[]: get_one_hot(df, features)    
    
# Prepare the data sets.
X_final = data_prep.get_one_hot(df_tmp, features=['Sex', 'Pclass'])
Y_final = df.loc[:, 'Survived'] # Get the label from the original data frame df.

# Check data.
print(X_final.shape, Y_final.shape)
X_final.iloc[:5,], df_tmp.iloc[:5,]


# In[ ]:


# Check the data.
print_missing_values(X_final, 0)
print(X_final.dtypes)


# ## Scaling the Dataframe

# In[ ]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_scaled = pd.DataFrame(ss.fit_transform(X_final))
X_scaled.columns = X_final.columns
X_scaled.iloc[:2,]


# In[ ]:


type(X_final)


# # Split Dataset into Training and Validation datasets

# In[ ]:


from sklearn.model_selection import train_test_split

# Divide into training and cross-validation datasets.
X_train, X_validation, Y_train, Y_validation = train_test_split(X_scaled, Y_final, test_size=0.2, random_state=42)
# X_train, X_validation, Y_train, Y_validation = X_final, X_final, Y_final, Y_final

# Check the data.
print(X_train.shape, Y_train.shape)
print(X_validation.shape, Y_validation.shape)


# # 5-fold Cross Validation on 4 Different Classifiers

# In[ ]:


from sklearn.base import clone

class HelperCrossValidation:
    """
    Helper class for running cross validations.
    """
    def print_features_importance(self, df, clf):
        if not hasattr(clf, 'feature_importances_'):
            return
        
        print(type(df))
        print(type(clf))
        
        # Display the features by descending importance.
        df_disp = pd.DataFrame(list(zip(list(df.columns), clf.feature_importances_)))
        df_disp.columns = ['Feature', 'Percentage']
        print(df_disp.sort_values(by='Percentage', ascending=False)[:10].to_string(index=False))

    def cross_val_score_do(self, name, clf, X, Y, cv):
        # Cross validate the classifier.
        scores = cross_val_score(clf, X, Y, cv=cv)
        print("\n%s Score: %.10f" % (name, scores.mean()))

        # Fit the classifier and show the feature importance.
        clf_cloned = clone(clf)
        model = clf_cloned.fit(X, Y)

        # Display the features by descending importance.
        self.print_features_importance(X, clf_cloned)

        return model


# # Setting Training Dataset and Classifiers

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Setup.
X = X_train
Y = Y_train

# Instanciate a HelperCrossValidation
helper_cv = HelperCrossValidation()

# Model.
clf_lr = LogisticRegression(penalty='l2', C=0.1)
model_lr = helper_cv.cross_val_score_do('RandomForestClassifier', clf_lr, X, Y, 5)

clf_rfc = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=101)
model_rfc = helper_cv.cross_val_score_do('RandomForestClassifier', clf_rfc, X, Y, 5)

clf_etc = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=290)
model_etc = helper_cv.cross_val_score_do('ExtraTreesClassifier', clf_etc, X, Y, 5)

clf_dtc = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=13)
model_dtc = helper_cv.cross_val_score_do('DecisionTreeClassifier', clf_dtc, X, Y, 5)

clf_gbc = GradientBoostingClassifier(n_estimators=70, learning_rate=0.5, max_depth=1, random_state=43)
model_gbc = helper_cv.cross_val_score_do('GradientBoostingClassifier', clf_gbc, X, Y, 5)

# SGDClassifier(max_iter=5)


# # ROC Visualization

# In[ ]:


from sklearn import metrics

def plot_roc(models={}, X=None, Y=None):
    for name in models:
        model = models[name]
        
        # Evaluate.
        y_pred_proba = model.predict_proba(X)[::, 1] # Dont know why take index=1 or 0?!?!
        print("%s Accuracy :%.15f" % (name, metrics.accuracy_score(Y, model.predict(X))))

        # Compute the probabilities.
        fpr, tpr, _ = metrics.roc_curve(Y, y_pred_proba, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)

        # Plot the ROC.
        fpr_tpr = pd.DataFrame(list(zip(fpr.ravel(), tpr.ravel())))
        fpr_tpr.columns = [ 'fpr', 'tpr']
        sns.lineplot(x="fpr", y="tpr", data=fpr_tpr, label='%s ROC fold (AUC = %0.2f)' % (name, roc_auc))

    sns.lineplot([0, 1], [0, 1], linestyle='--', lw=2, label='Chance', alpha=.8)
    plt.legend(loc=4)
    plt.title('Receiver Operating Characteristic')
    plt.show()
    
plot_roc({'lr' : model_lr, 'dtc' : model_dtc, 'etc' : model_etc, 'gbc' : model_gbc, 'rfc' : model_rfc}, X, Y)


# # 5-fold Cross Validation on Best Classifier for Hyperparameters Tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV

def do_cross_validation(clf, X, Y, k):
    """
    Run the k-fold GridSearchCV on X and Y
    
    Parameters
    ----------
    clf: the original classifier to clone for testing the hyperparameters.
    X: the data set to use for splitting into training and cross validation datasets.
    Y: the labels.
    k; the number of cross validation datasets split from X.
    
    Returns
    -------
    clf: the best classifier after evaluating all the possible hyperparameter settings.
    
    """
    parameters = {'max_features': [1, len(X.columns)], 'n_estimators': [5, 50, 250, 400, 1000]}

    clf_cv = GridSearchCV(clf, parameters, cv=k)

    get_ipython().magic(u'timeit')
    clf_model = clf_cv.fit(X, Y)

    # Display the scores.
#     for row in clf_model.cv_results_:
#         print(row)

    return clf_model.best_estimator_

clf = do_cross_validation(clf_etc, X, Y, 5)


# # Evaluation of the Best Classifier

# In[ ]:


from sklearn.metrics import confusion_matrix

# Print features importance.
helper_cv.print_features_importance(X, clf)


# In[ ]:


import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def display_metrics(clf, X, Y):
    # Predict the values on X.    
    X_prediction = clf.predict(X)

    print("Score: %.15f" % clf.score(X, Y))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y, X_prediction)
    # tn, fp, fn, tp = cnf_matrix.ravel()

    # Compute confusion matrix
    np.set_printoptions(precision=2)

    class_names = ['Did Not', 'Survived']

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


# In[ ]:


display_metrics(clf, X, Y)


# # Test on X_test Dataset

# In[ ]:


# Create the test sets.
Xt = X_validation
Yt = Y_validation

print(Xt.shape, Yt.shape)
print(Xt.columns)

print_missing_values(Xt, 0)


# In[ ]:


# Score the Classifier that performed best on the test set.
print("Score: %.15f" % clf.score(Xt, Yt))

# Print features importance.
helper_cv.print_features_importance(Xt, clf)

# Display metrics.
display_metrics(clf, Xt, Yt)


# # CSV of Test Data Set Prediction

# In[ ]:


# Load the test set.
df_test_raw = pd.read_csv(base_path_data + '/test.csv', sep=",", header=0, encoding='utf-8')

print(df_test_raw.isna().any())
print(df_test_raw.dtypes)

# Show the result.
print_missing_values(df_test_raw, 0)

# print(df_test_raw.iloc[:10,])
df_test_raw.query('Fare != Fare')


# In[ ]:


from sklearn.preprocessing import Imputer

df_tmp = df_test_raw
# Leave this out for the time being.
# df_tmp['Title'] = df_test_raw['Name'].apply(extract_title)

df_tmp.head()

df_tmp = data_prep.impute(df_tmp, df_tmp, column='Age')
df_tmp = data_prep.impute(df_tmp, df_tmp, column='Cabin', strategy='most_frequent')
df_tmp = data_prep.impute(df_tmp, df_tmp, column='Fare', strategy='mean')
df_tmp['Fare'] = df_tmp['Fare_imputed']

# Check the columns of the trained model based on features of X.
print(X.columns)
print(df_tmp.columns)

df_tmp = data_prep.select_features(df_tmp, features_to_keep=['Fare', 'Age_imputed', 'Sex', 'Pclass'])

# TODO: jh 2018/11/02
# Should check that the columns of the test dataset are the same as the X dataset used for training.

print(df_tmp.columns)

Xt_raw = data_prep.get_one_hot(df_tmp, features=['Sex', 'Pclass']) # 'Cabin_imputed', 

# Add scaling.
Xt_scaled = pd.DataFrame(ss.transform(Xt_raw))
Xt_scaled.columns = Xt_raw.columns

Xt_raw = Xt_scaled
print(df_test_raw.shape, Xt_raw.shape)


# In[ ]:


# Show the result.
print_missing_values(Xt, 0)
print_missing_values(Xt_raw, 0)


# In[ ]:


X_test_given_predicted = clf.predict(Xt_raw)

# print(X_test.iloc[:10,])


# In[ ]:


import datetime

df_to_submit = pd.DataFrame(list(zip(df_test_raw.loc[:,'PassengerId'], X_test_given_predicted)))
df_to_submit.columns = ['PassengerId', 'Survived']
df_to_submit.PassengerId = df_to_submit.PassengerId.astype(np.int32)
df_to_submit.Survived = df_to_submit.Survived.astype(np.int32)

today = datetime.datetime.today()
today_s = today.strftime('%Y%m%d-%H%M')

csv_dest = "%s/csv_submission_%s.csv" % (base_path_data, today_s)

# Save the CSV
if who_am_i[0] != 'root':
    df_to_submit.to_csv(csv_dest, index=False)
    print("test prediction save in:\n %s" % csv_dest)

