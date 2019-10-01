#!/usr/bin/env python
# coding: utf-8

# # Reusable data transformations using scikit-learn pipelines and hyperparameter optimization

# This kernel illustrates the use of [scikit-learn Pipelines](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) for data transformation. Custom transformers written here are easily reusable for other projects and they also enable including any data transformation parameters in the hyperparameter optimization.

# First import the required libraries:

# In[ ]:


import pandas as pd, sklearn, numpy as np, os
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, make_scorer


# Load the datasets. Note that for simplicity, we'll keep the label field `Survived` in the `train` and `test` dataframes until actually starting transformations and predictions.

# In[ ]:


data_folder = '../input'
train = pd.read_csv(os.path.join(data_folder, 'train.csv'))
test = pd.read_csv(os.path.join(data_folder, 'test.csv'))
n_train, m_train = train.shape


# Take a peek at the columns:

# In[ ]:


train.head()


# `train.info()` shows that many values in `Age` and `Cabin` columns are missing:

# In[ ]:


train.info()


# To get more insights, check statistical summary for numerical attributes using `train.describe()`:

# In[ ]:


train.describe()


# Important things to note here are that 
# - The dataset is slightly skewed as only 38% survived, which could be important to take into account in cross-validation
# - Some values in the `Fare` column are zero
# - As noted above, many `Age`s are missing.

# As there are lots of data exploration kernels out there, I will skip any further data exploration here and focus on the actual data transformation pipelines. Data exploration shows that `Age` is an important indicator for survival and the missing values should be imputed. The data processing steps carried out below are:
# 1. Split the dataset into features and labels. To simplify the kernel, we drop `Ticket`, `Cabin`, `Embarked`, and of course `PassengerId` from the feature set.
# 2. Extract the title from the `Name` field and use it as a feature to impute the missing values. For example, title `Mr.` says that the person in question was not a child.
# 3. One-hot encode all categorical attributes
# 4. Convert the dataframe to numpy matrix.

# First we split the dataset into features and labels and give them common names `X_train` and `y_train`. Note that these are still dataframes, conversion to NumPy is done just before feeding the inputs to prediction algorithms.A Also note that we split the given training set into training and validation sets. The validation set works as a hold-out set that can be used for estimating the generalization error after hyperparameter optimization. The algorithm used in the final submission use, of course, all training data available.

# In[ ]:


from sklearn.model_selection import train_test_split

def drop_unused_columns(df):
    return df.drop(['PassengerId', 'Cabin', 'Ticket', 'Embarked'], axis=1)

def to_features_and_labels(df):
    y = df['Survived'].values
    X = drop_unused_columns(df)
    X = X.drop('Survived', axis=1)
    return X, y

X_train_val, y_train_val = to_features_and_labels(train) # All data with labels, to be split into train and val
X_test = drop_unused_columns(test)

# Split the available training data into training set (used for choosing the best model) 
# and validation set (used for estimating the generalization error, could also be called "hold-out" set)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42)
X_train.head()


# First write a [scikit-learn transformer](http://scikit-learn.org/stable/data_transforms.html) for converting name to title. You may think that there's a lot of overhead involved in writing such classes (and you're right), but the transformer classes are highly reusable and therefore save a lot of time in future projects.

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameColumnMapper(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, mapping_func, new_column_name=None):
        self.column_name = column_name
        self.mapping_func = mapping_func
        self.new_column_name = new_column_name if new_column_name is not None else self.column_name
    def fit(self, X, y=None):
        # Nothing to do here
        return self
    def transform(self, X):
        transformed_column = X.transform({self.column_name: self.mapping_func})
        Y = X.copy()
        Y = Y.assign(**{self.new_column_name: transformed_column})
        if self.column_name != self.new_column_name:
            Y = Y.drop(self.column_name, axis=1)
        return Y

# Return a lambda function that extracts title from the full name, this allows instantiating the pattern only once
def extract_title():
    import re
    pattern = re.compile(', (\w*)')
    return lambda name: pattern.search(name).group(1)

# Example usage and output 
df = DataFrameColumnMapper(column_name='Name', mapping_func=extract_title(), new_column_name='Title').fit_transform(X_train)
df.head()


# Let's take a look at the transformed names:

# In[ ]:


df['Title'].value_counts()[1:10]


# It seems that only the, say, five most frequent would be useful for imputing ages, so let us write a transformer that transforms the less frequent fields of a categorical attribute as "Other". The number of classes to keep is included as a constructor argument that can be optimized using cross-validation.

# In[ ]:


class CategoricalTruncator(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, n_values_to_keep=5):
        self.column_name = column_name
        self.n_values_to_keep = n_values_to_keep
        self.values = None
    def fit(self, X, y=None):
        # Here we must ensure that the test set is transformed similarly in the later phase and that the same values are kept
        self.values = list(X[self.column_name].value_counts()[:self.n_values_to_keep].keys())
        return self
    def transform(self, X):
        transform = lambda x: x if x in self.values else 'Other'
        y = X.transform({self.column_name: transform})
        return X.assign(**{self.column_name: y})

# Print title counts
title_counts = CategoricalTruncator('Title', n_values_to_keep=3).fit_transform(df)['Title'].value_counts()
title_counts


# Let us see what we have done so far by putting the transformers together in a pipeline:

# In[ ]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('name_to_title', DataFrameColumnMapper(column_name='Name', mapping_func=extract_title(), new_column_name='Title')),
    ('truncate_titles', CategoricalTruncator('Title', n_values_to_keep=3))
])

df = pipeline.fit_transform(X_train)
df.head(10)


# Now write a generic imputer that uses values in a given column ("Title" in our case) to impute missing values for a numeric column ("Age") with the median value for the group in question.

# In[ ]:


class ImputerByReference(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_impute, column_ref):
        self.column_to_impute = column_to_impute
        self.column_ref = column_ref
        # TODO Allow specifying the aggregation function
        # self.impute_func = np.median if impute_type == 'median' or impute_type is None else np.mean
    def fit(self, X, y=None):
        # Pick columns of interest
        df = X.loc[:, [self.column_to_impute, self.column_ref]]
        # Dictionary containing mean per group
        self.value_per_group = df.groupby(self.column_ref).median().to_dict()[self.column_to_impute]
        return self
    def transform(self, X):
        def transform(row):
            row_copy = row.copy()
            if pd.isnull(row_copy.at[self.column_to_impute]):
                row_copy.at[self.column_to_impute] = self.value_per_group[row_copy.at[self.column_ref]]
            return row_copy
        return X.apply(transform, axis=1)

# Example output
ImputerByReference('Age', 'Title').fit_transform(df).head(10)


# The full pipeline so far is below. Let us use `.info()` to check that no values are missing from the transformed data:

# In[ ]:


pipeline = Pipeline([
    ('name_to_title', DataFrameColumnMapper(column_name='Name', mapping_func=extract_title(), new_column_name='Title')),
    ('truncate_titles', CategoricalTruncator('Title', n_values_to_keep=3)),
    ('impute_ages_by_title', ImputerByReference('Age', 'Title'))
])

df = pipeline.fit_transform(X_train)
df.info()


# Looks good! Now we one-hot encode all categorical attributes using again a generic transformer. Note that the one-hot encoding would get us into trouble if we were encoding columns with many different values (like column `Ticket`), but we do not worry about that here. The first step is convert all categorical attributes to numerical ones by factorizing to integers (`Mrs.` is 0, `Mr.` is 1, for example.):

# In[ ]:


class CategoricalToOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    def fit(self, X, y=None):
        # Pick all categorical attributes if no columns to transform were specified
        if self.columns is None:
            self.columns = X.select_dtypes(exclude='number')
        
        # Keep track of which categorical attributes are assigned to which integer. This is important 
        # when transforming the test set.
        mappings = {}
        
        for col in self.columns:
            labels, uniques = X.loc[:, col].factorize() # Assigns unique integers for all categories
            int_and_cat = list(enumerate(uniques))
            cat_and_int = [(x[1], x[0]) for x in int_and_cat]
            mappings[col] = {'int_to_cat': dict(int_and_cat), 'cat_to_int': dict(cat_and_int)}
    
        self.mappings = mappings
        return self

    def transform(self, X):
        Y = X.copy()
        for col in self.columns:
            transformed_col = Y.loc[:, col].transform(lambda x: self.mappings[col]['cat_to_int'][x])
            for key, val in self.mappings[col]['cat_to_int'].items():
                one_hot = (transformed_col == val) + 0 # Cast boolean to int by adding zero
                Y = Y.assign(**{'{}_{}'.format(col, key): one_hot})
            Y = Y.drop(col, axis=1)
        return Y
    
# Example output    
CategoricalToOneHotEncoder().fit_transform(df).head()   


# Note that we could drop either `Sex_male` or `Sex_female` without losing any data, but we'll leave that for now. Now that all values are imputed and all columns are numerical, we finally define a transformer for converting the DataFrame to a NumPy matrix and build the  full data preparation pipeline. We also include `MinMaxScaler` as last preprocessing step as some algorithms are sensitive to variations in scale. We also add a simple imputer, as test set has one missing `Fare` value.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

class DataFrameToValuesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        # Remember the order of attributes before converting to NumPy
        self.attribute_order = list(X)
        return self
    def transform(self, X):
        return X.loc[:, self.attribute_order].values

def build_preprocessing_pipeline():
    return Pipeline([
        ('name_to_title', DataFrameColumnMapper(column_name='Name', mapping_func=extract_title(), new_column_name='Title')),
        ('truncate_titles', CategoricalTruncator(column_name='Title', n_values_to_keep=3)),
        ('impute_ages_by_title', ImputerByReference(column_to_impute='Age', column_ref='Title')),
        ('encode_categorical_onehot', CategoricalToOneHotEncoder()),
        ('encode_pclass_onehot', CategoricalToOneHotEncoder(columns=['Pclass'])),
        ('to_numpy', DataFrameToValuesTransformer()),
        ('imputer', SimpleImputer(strategy='median')), # Test set has one missing fare
        ('scaler', MinMaxScaler())
    ])

X_train_prepared = build_preprocessing_pipeline().fit_transform(X_train)
print('Prepared training data: {} samples, {} features'.format(*X_train_prepared.shape))


# Before moving to trying different algorithms and optimizing hyperparameters, we define a few helper functions that are hopefully self-explanatory.

# In[ ]:


def build_pipeline(classifier=None):
    preprocessing_pipeline = build_preprocessing_pipeline()
    return Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('classifier', classifier) # Expected to be filled by grid search
    ])


def build_grid_search(pipeline, param_grid):
    return GridSearchCV(pipeline, param_grid, cv=5, return_train_score=True, refit='accuracy',
                        scoring={ 'accuracy': make_scorer(accuracy_score),
                                  'precision': make_scorer(precision_score)
                                },
                        verbose=1)

def pretty_cv_results(cv_results, 
                      sort_by='rank_test_accuracy',
                      sort_ascending=True,
                      n_rows=5):
    df = pd.DataFrame(cv_results)
    cols_of_interest = [key for key in df.keys() if key.startswith('param_') 
                        or key.startswith('mean_train') 
                        or key.startswith('mean_test_')
                        or key.startswith('rank')]
    return df.loc[:, cols_of_interest].sort_values(by=sort_by, ascending=sort_ascending).head(n_rows)

def run_grid_search(grid_search):
    grid_search.fit(X_train, y_train)
    print('Best test score accuracy is:', grid_search.best_score_)
    return pretty_cv_results(grid_search.cv_results_)


# Trying different algorithms is now straightforward. Choose the parameters to vary and run the grid search with cross validation to find both the best preprocessing pipeline and classifier.

# ## [Logistic classifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier)

# In[ ]:


param_grid = [
    { 'preprocessing__truncate_titles__n_values_to_keep': [3, 4, 5],
      'classifier': [SGDClassifier(loss='log', tol=None, random_state=42)],
      'classifier__alpha': np.logspace(-5, -3, 3),
      'classifier__penalty': ['l2'],
      'classifier__max_iter': [20],
    }
]
log_grid_search = build_grid_search(pipeline=build_pipeline(), param_grid=param_grid)
linear_cv = run_grid_search(grid_search=log_grid_search)


# In[ ]:


linear_cv


# ## [Random forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

# In[ ]:


param_grid = [
    { 'preprocessing__truncate_titles__n_values_to_keep': [5],
      'classifier': [RandomForestClassifier(random_state=42)],
      'classifier__n_estimators': [10, 30, 100],
      'classifier__max_features': range(4, 14, 3)
    }
]
rf_grid_search = build_grid_search(pipeline=build_pipeline(), param_grid=param_grid)
rf_cv_results = run_grid_search(grid_search=rf_grid_search)
rf_cv_results


# ## [SVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

# In[ ]:


param_grid = [
    { 
        'preprocessing__truncate_titles__n_values_to_keep': [5],
        'classifier': [ SVC(random_state=42, probability=True) ], # Probability to use in voting later
        'classifier__C': np.logspace(-1, 1, 3),
        'classifier__kernel': ['linear', 'poly', 'rbf'],
        'classifier__gamma': ['auto', 'scale']
    }
]


svm_grid_search = build_grid_search(pipeline=build_pipeline(), param_grid=param_grid)
svm_cv_results = run_grid_search(grid_search=svm_grid_search)


# In[ ]:


svm_cv_results


# ## [Gaussian process classifier](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier)

# In[ ]:


from sklearn.gaussian_process.kernels import RBF, Matern

param_grid = [
    { 
        'preprocessing__truncate_titles__n_values_to_keep': [5],
        'classifier': [ GaussianProcessClassifier() ], 
        'classifier__kernel': [1.0*RBF(1.0), 1.0*Matern(1.0)]
    }
]

gp_grid_search = build_grid_search(pipeline=build_pipeline(), param_grid=param_grid)
gp_cv_results = run_grid_search(grid_search=gp_grid_search)


# In[ ]:


gp_cv_results


# ## [AdaBoost](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier)

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

param_grid = [
    { 
        'preprocessing__truncate_titles__n_values_to_keep': [5],
        'classifier': [ AdaBoostClassifier(random_state=42) ],
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': np.logspace(-1, 1, 3),
        'classifier__base_estimator': [
            DecisionTreeClassifier(max_depth=1),
            DecisionTreeClassifier(max_depth=2)
        ],
        # 'classifier__base_estimator__max_depth': [1, 2]
    }
]

ada_grid_search = build_grid_search(pipeline=build_pipeline(), param_grid=param_grid)
ada_cv_results = run_grid_search(grid_search=ada_grid_search)


# In[ ]:


ada_cv_results


# ## [Gradient boosting](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

# In[ ]:


param_grid = [
    { 
        'preprocessing__truncate_titles__n_values_to_keep': [5],
        'classifier': [ GradientBoostingClassifier(random_state=42) ],
        'classifier__loss': ['deviance'],
        'classifier__n_estimators': [50, 100],
        'classifier__max_features': [7, 13],
        'classifier__max_depth': [3, 5],
        'classifier__min_samples_leaf': [1],
        'classifier__min_samples_split': [2]
    }
]

gb_grid_search = build_grid_search(pipeline=build_pipeline(), param_grid=param_grid)
gb_cv_results = run_grid_search(grid_search=gb_grid_search)


# In[ ]:


gb_cv_results


# In[ ]:


gb_grid_search.best_estimator_.score(X_val, y_val)


# ## Voting classifier
# Create a voting classifier from the best estimators and check the generalization accuracy for heldout data `X_val`

# In[ ]:


voting_estimators = [
    # ('logistic', log_grid_search),
    # ('rf', rf_grid_search),
    ('svc', svm_grid_search),
    ('gp', gp_grid_search),
    # ('ada', ada_grid_search),
    ('gb', gb_grid_search),
]

estimators_with_names = [(name, grid_search.best_estimator_) for name, grid_search in voting_estimators]

voting_classifier = VotingClassifier(estimators=estimators_with_names,
                                     voting='soft')

voting_classifier.fit(X_train, y_train)
voting_classifier.score(X_val, y_val)
# cross_val_score(voting_classifier, X_train_val, y_train_val, cv=5)


# ## Train voting classifier with all data available

# In[ ]:


voting_classifier.fit(X_train_val, y_train_val)


# ## Prepare the submission

# In[ ]:


def get_predictions(estimator):
    predictions = estimator.predict(X_test)
    indices = test.loc[:, 'PassengerId']
    as_dict = [{'PassengerId': index, 'Survived': prediction} for index, prediction in zip(indices, predictions)]
    return pd.DataFrame.from_dict(as_dict)

predictions = get_predictions(voting_classifier)


# In[ ]:


submission_folder = '.'
dest_file = os.path.join(submission_folder, 'submission.csv')
predictions.to_csv(dest_file, index=False)

