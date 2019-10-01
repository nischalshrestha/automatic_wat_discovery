#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


from patsy import dmatrices, build_design_matrices
import numpy as np
import pandas as pd
import sklearn.pipeline as skpipe
import sklearn.preprocessing as skpreprocess
import sklearn.linear_model as linmod
import sklearn.svm as svm
import sklearn.model_selection as crossval

def read_data():
    return pd.read_csv('../input/train.csv', index_col=0),            pd.read_csv('../input/test.csv', index_col=0)
    
    
def preprocess_data(train, test):
    # Pull out the numeric cols from the test set (to avoid selecting the target variable)
    ncol = test.select_dtypes(include=[np.number]).columns
    # Impute
    imputer = skpreprocess.Imputer(strategy='median')
    train.ix[:, ncol] = imputer.fit_transform(train.ix[:, ncol])
    test.ix[:, ncol] = imputer.transform(test.ix[:, ncol])
    return make_design_matrix(train, test)
    
    
def make_design_matrix(train, test):
    formula = """Survived ~ C(Pclass, Treatment) +
                            C(Sex, Treatment) +
                            is_child(Age) +
                            Age +
                            np.log1p(Fare) +
                            (SibSp + Parch) +
                            num_tickets(Cabin) +
                            Parch
              """
    
    # Make sure we run the stateful transformation on the test set
    y, X = dmatrices(formula, train, return_type='dataframe', NA_action='raise')
    X_test = build_design_matrices([X.design_info], test, return_type='dataframe', NA_action='raise')[0]
    return X, y.Survived, X_test


def is_child(ages):
    return ages.map(lambda x: 1 if x < 18 else 0)

def num_tickets(cabin):
    return cabin.map(lambda x: len(x.split(' ')) if isinstance(x, str) else 0)


# In[ ]:


train, test = read_data()
X, y, X_test = preprocess_data(train, test)


# In[ ]:


model = skpipe.Pipeline([('scaler', skpreprocess.StandardScaler()),
                         ('svm', svm.SVC())])


# In[ ]:


import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

def run_model_diagnostics(X, y, X_test, model):
    # Cross validation score
    cv_scores = crossval.cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print('CV Score: {:.5f} (+/- {:.5f})'.format(cv_scores.mean(), cv_scores.std()))
    
    
# Run model diagnostics
run_model_diagnostics(X, y, X_test, model)


# In[ ]:


# Output!
model.fit(X, y)
preds = pd.DataFrame({'PassengerId': test.index,
                      'Survived': model.predict(X_test)})
preds.Survived = preds.Survived.astype(int)
preds.to_csv('output.csv', index=False)


# In[ ]:




