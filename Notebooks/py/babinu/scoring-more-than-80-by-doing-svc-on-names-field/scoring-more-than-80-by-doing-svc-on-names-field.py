#!/usr/bin/env python
# coding: utf-8

# # Basic Overview
# 
# The idea here is to implement SVC algorithm on names field and see if that helps us in prediction survivals accurately.
# 
# Comments/criticisms/appreciations are greatly accepted and appreciated.
# 
# Source of data : https://www.kaggle.com/c/titanic/data

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV


# In[ ]:


# Make sure that unnecessary  warnings are avoided.
# Thanks to https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)


# In[ ]:


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# ### Using SVC

# In[ ]:


# We use a pipeline to make things easire
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
svc_clf = Pipeline([('vect', TfidfVectorizer()),
                    ('transformer', TfidfTransformer()),
                    ('classify', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, 
                                               random_state=0, max_iter=5, tol=None))])


# #### Training on some portion of input data and using the other portion as a test set

# In[ ]:


svc_clf.fit(train_data['Name'][:700], train_data['Survived'][:700])


# In[ ]:


predictions = svc_clf.predict(train_data['Name'][700:]) 


# In[ ]:


survived_or_not = train_data['Survived'][700:]


# In[ ]:


np.mean(predictions == survived_or_not)


# Comment : The results look encouraging. Let us find optimal parameters using GridSearch

# ### Finding optimal parameters using GridSearch

# In[ ]:


parameters = {'vect__ngram_range' : [(1, 1), (2, 2), (3 , 3)],
              'transformer__use_idf' : (True, False),
              'classify__alpha' : (1e-2, 1e-3),
              }


# In[ ]:


gs_clf = GridSearchCV(svc_clf, parameters, n_jobs=-1)


# In[ ]:


gs_clf.fit(train_data['Name'], train_data['Survived'])


# In[ ]:


cv_result = pd.DataFrame(gs_clf.cv_results_)


# In[ ]:


gs_clf.best_params_


# 
# ### Training the best model on entire training set
# 
# After that, we use the same to generate predictions on the test set provided by kaggle as well.

# In[ ]:


# We use a pipeline to make things easire
from sklearn.linear_model import SGDClassifier
best_model_svc = Pipeline([('vect', TfidfVectorizer()),
                           ('transformer', TfidfTransformer(use_idf=False)),
                           ('classify', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, 
                                                      random_state=0, max_iter=5, tol=None))])


# In[ ]:


best_model_svc.fit(train_data['Name'], train_data['Survived'])


# In[ ]:


# Generate out of sample predictions
predictions = best_model_svc.predict(test_data['Name'])


# In[ ]:


test_data['Predictions'] = predictions


# In[ ]:


kaggle_data = test_data[['PassengerId', 'Predictions']].copy()
kaggle_data.rename(columns={'Predictions' : 'Survived'}, inplace=True)
kaggle_data.sort_values(by=['PassengerId']).to_csv('kaggle_out_svc_names.csv', index=False)

