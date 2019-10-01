#!/usr/bin/env python
# coding: utf-8

# This notebook is my submission for Titanic Kaggle competition.  It references two good texts on machine learning:
# - Hands-On Machine Learning and Deep Learning with Scikit-Learn and TensorFlow by Aurelien Geron
# - Machine Learning Mastery by Jason Brownlee
# 
# I also found the following websites useful:
# - http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
# - http://michelleful.github.io/code-blog/2015/06/20/pipelines/
# 
# It does not focus on EDA or on achieving a great score.   
# 
# The main aim of this notebook to utilise methods that make it easier to change the features selected and to tune the hyperparameters in a efficient way.
# 
# Alot of the code has been taken from the sources given above.  I found them very useful as one of the most frustrating aspects of the competitions is having to fiddle with the features normalising, scaling etc.  After a while it becomes very messy if not organised in a efficient manner.
# 
# I hope that other users can also gain benefit from the methods.
# 

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings('ignore')

#allows printing of all data in cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# ## 1.0 Investigate Data

# In[ ]:


train_data.head()


# In[ ]:


train_data.info()


# The attributes have the following meaning:
# - **Survived**: the target, 0 means the passenger did not survive, while 1 means he/she survived.
# - **Pclass**: passenger class.
# - **Name**, Sex, Age: self-explanatory
# - **SibSp**: how many siblings & spouses of the passenger aboard the Titanic.
# - **Parch**: how many children & parents of the passenger aboard the Titanic.
# - **Ticket**: ticket id
# - **Fare**: price paid (in pounds)
# - **Cabin**: passenger's cabin number
# - **Embarked**: where the passenger embarked the Titanic

# In[ ]:


train_data.describe()
train_data['Age'].mean()


# In[ ]:


#check the target
train_data['Survived'].value_counts()


# In[ ]:


#age has 177 missing data values, calculate a mean value and insert
average_age = train_data['Age'].mean()
std_age = train_data['Age'].std()
count_age = train_data['Age'].isnull().sum()

#generate rand numbers between (mean - std) & (mean = std) of length count_age
random_1 = np.random.randint(average_age - std_age, average_age + std_age,size = count_age)
#replace nan values with calculated random values
train_data['Age'][np.isnan(train_data['Age'])] = random_1
#float not needed, convert to integer
train_data['Age'] = train_data['Age'].astype(int)


average_age = test_data['Age'].mean()
std_age = test_data['Age'].std()
count_age = test_data['Age'].isnull().sum()

#generate rand numbers between (mean - std) & (mean = std) of length count_age
random_1 = np.random.randint(average_age - std_age, average_age + std_age,size = count_age)
#replace nan values with calculated random values
test_data['Age'][np.isnan(test_data['Age'])] = random_1
#float not needed, convert to integer
test_data['Age'] = test_data['Age'].astype(int)


# In[ ]:


train_data['Age'].mean()


# ## 2.0 Process the Data

# In[ ]:


#Try using using a categorical variable for age
train_data['AgeBucket'] = train_data['Age'] // 15 * 15
train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()

test_data['AgeBucket'] = test_data['Age'] // 15 * 15


# In[ ]:


#add relatives on board category
train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
train_data[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean()
train_data.dtypes

test_data["RelativesOnboard"] = test_data["SibSp"] + test_data["Parch"]


# A method used by Geron to allow efficient pipelining of categorical variables:

# In[ ]:


#A method taken from stakoverflow that uses encodes categorical features into numeric.  It is similar to get_dummies
#but allows  

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out


# DataframeSelector to select specific attributes from the DataFrame:

# In[ ]:


#Use to select specific attributes
from sklearn.base import BaseEstimator, TransformerMixin

# A class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


# Pipeline for the numerical attributes:

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")

num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["SibSp", "Parch", "Fare",'RelativesOnboard'])),
        ("imputer", Imputer(strategy="median")),
        ('Scaler', StandardScaler())
    ])


# Imputer for the string categorical columns (the regular Imputer does not work on those):

# In[ ]:


# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent = pd.Series([X[c].value_counts().index[0] for c in X],
                                       index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent)


# Pipeline for the categorical attributes:

# In[ ]:


cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked","AgeBucket"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", CategoricalEncoder(encoding='onehot-dense')),
    ])


# In[ ]:


cat_pipeline.fit_transform(train_data)


# Join the numerical and categorical pipelines:

# In[ ]:


from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# Preprocessing pipeline that takes the raw data and outputs numerical input features that we can feed to any Machine Learning model we want.

# In[ ]:


X_train = preprocess_pipeline.fit_transform(train_data)
X_train


# Get the labels:

# In[ ]:


y_train = train_data['Survived']

validation_size = 0.2
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=validation_size, 
                                                                random_state=seed)


# In[ ]:


X_train.shape
X_test.shape
y_train.shape
y_test.shape


# ## 3.0 Evaluate non-ensemble Baseline methods  
# 

# In[ ]:


#evaluation - baselines
num_folds = 10
seed = 7
scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s %f %f " % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[ ]:


# compare algorithms
fig = plt.figure()
fig.suptitle('Comparison of non-ensemble methods')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show();


# ## 3.1 Tune the best non-ensemble methods

# In[ ]:


neighbors = [1, 3, 5, 7, 9, 15, 19, 21]
param_grid = dict(n_neighbors=neighbors)
model = KNeighborsClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


# Tune scaled SVM
c_values = [1,1.3,1.5,1.7]
kernel_values = ['rbf']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


svm_clf = SVC(C=1.3,kernel='rbf')
svm_clf.fit(X_train,y_train)


# Try on the test data and make submission

# In[ ]:


X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
scores.mean()


# In[ ]:


output = pd.DataFrame({ 'PassengerId' : test_data['PassengerId'], 'Survived': y_pred })


# In[ ]:


output.to_csv('submission.csv', index=False)


# ## 4.0 Evaluate ensemble methods

# In[ ]:


ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))
results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show();


# ## 4.1 Evaluate Ensemble method

# In[ ]:


# Tune scaled GBM
num_trees = [10,50,100,150,200,250,300]
#kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(n_estimators=num_trees)
#param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200),'n_estimators':[100,200,300]}                 
model = GradientBoostingClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


GB_clf = GradientBoostingClassifier(max_depth=9,n_estimators = 150,min_samples_split=400)
GB_clf.fit(X_train,y_train)
X_test = preprocess_pipeline.transform(test_data)
y_pred = GB_clf.predict(X_test)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(GB_clf, X_train, y_train, cv=10)
scores.mean()
output = pd.DataFrame({ 'PassengerId' : test_data['PassengerId'], 'Survived': y_pred })
output.to_csv('submission.csv', index=False)

