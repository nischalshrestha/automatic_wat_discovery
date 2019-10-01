#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machines and Neural Networks 
# 
# [Iaroslav Shcherbatyi](http://iaroslav-ai.github.io/), [ED3S 2018](http://iss.uni-saarland.de/de/ds-summerschool/)

# # Synopsis
# 
# Kernel Support Vector Machines are a popular class of learning algorithms. While such algorithms have strong theoretical underpinning and empirically perform well, they do not scale well to "large" datasets, due to computational and memory requirements. In contrast, Artificial Neural Networks scale better, and can be applied to datasets of size in millions. However, training procedure of ANN is less rigorous, and requires setting more parameters than that of SVM. In this notebook, we will look at examples on how both learning algorithms can be used. 
# 
# A few general aspects of machine learning will also be discussed, such data preprocessing, and scaling of feature ranges, which often leads to improved outcomes.
# 
# Note: if you are viewing this notebook on Kaggle, you can download the notebook by clicking the cloud with arrow pointing down in the upper panel, and the necessary data from the panel to the right. 

# # SVM usage in sklearn
# 
# SVM learning algorithm is available in `scikit-learn`, and can be used similar as the other algorithms you have seen.  For this we will use a "Titanic" dataset, which contains records of which passengers survived Titanic crash, and their features.

# In[ ]:


import numpy as np
import pandas as pd
titanic = pd.read_csv('../input/train.csv')

# some preprocessing is applied for simplicity
titanic = titanic[['Sex', 'Pclass', 'Age', 'SibSp', 'Fare', 'Survived']]  # use subset of columns
titanic = titanic.dropna()  # drop rows with missing values

display(titanic.head())


# For some initial prototyping, we will use only numerical coumns, which can be provided directly into the SVM.

# In[ ]:


# use only numerical values
Xy = titanic[['Pclass', 'Age', 'SibSp', 'Fare', 'Survived']].values

# separate inputs and outputs
X = Xy[:, :-1]
y = Xy[:, -1]


# Now, lets use the Kernel SVM for prediction of whether a person would survive Titanic crash or not. A few possibilities are to be considered. 
# 
# * Firstly, the Gaussian Kernel used by default does not account for different ranges of features. Scaling feature values to standard range can be done using `StandardScaler` class from `sklearn`. 
# * It is useful to keep the whole model, including scaling class, in a single variable. This can be done using `make_pipeline` function, which can "chain" multiple data transformation classes and one final learning algorithm. This typically reduces this size of codebase, and makes it easier to share model as a single variable [serialized](https://docs.python.org/3/library/pickle.html) to a file.
# * It is important to not use testing set for hyperparameter (`C`, `gamma`) selection, as it can lead to overfitting; Validation set should be used instead. Furthermore, it is good to avoid bias due to dataset split into training and validation partitions. One way is to use [cross-validation](https://en.wikipedia.org/wiki/Cross-validation).

# In[ ]:


# using Kernel SVM is easy in sklearn
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# do the usual splitting into training / testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
for C in [0.01, 0.1, 1, 10, 100]:
    for gamma in [0.01, 0.1, 1.0, 10, 100]:
        score = 0
        # Task: fill in here proper training and scoring of SVC.        
        print(C, gamma, score)


# # Simple interface to hyperparameter grid search 
# 
# Above loop can be simplified through the `GridSearchCV` class from `sklearn`. This class performs search for parameters of the model which result in highest cross-validation score, and fits the model to the dataset with such best parameters.

# In[ ]:


# using Kernel SVM is easy in sklearn
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# do the usual splitting into training / testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# two arguments are necessary: sklearn "estimator" class,
# and range of parameters for this estimator. 
model = GridSearchCV(
    estimator=make_pipeline(StandardScaler(), SVC()),
    param_grid={  # format: lowercase_estimator_name__param_name
        'svc__C': [0.01, 1, 100], 
        'svc__gamma': [0.01, 1, 100],
    },
    cv=5,  # number of validation folds
    n_jobs=1  # number of parallel jobs
)
# Q: how does the grid search scale computationally?

model.fit(X_train, y_train)
print(model.score(X_test, y_test))

eg_inputs = X_test[:5]
print(eg_inputs)
print(model.predict(eg_inputs))


# Check out [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) for efficient parameter selection; For instance, `sklearn` compatible `BayesSearchCV` class.

# In[ ]:


from skopt import BayesSearchCV

# include below until https://github.com/scikit-optimize/scikit-optimize/issues/718 is resolved
class BayesSearchCV(BayesSearchCV):
    def _run_search(self, x): raise BaseException('Use newer skopt')


# In[ ]:


model = BayesSearchCV(
    estimator=make_pipeline(StandardScaler(), SVC()),
    search_spaces={
        'svc__C': (0.01, 100.0, 'log-uniform'),  # specify ranges instead of discrete values
        'svc__gamma': (0.01, 100.0, 'log-uniform'),
    },
    cv=5,
    n_iter=16,  # fixed number of parameter configuration trials!
    n_jobs=4,  # it runs evaluations in parallel too!
    verbose=1
)

model.fit(X_train, y_train)


# In[ ]:


print(model.best_params_)
print(model.score(X_test, y_test))


# # ANN in Keras
# 
# Keras is a popular python package, that provides hight level apis to work with ANN. It is based on a popular `TensorFlow` library, and is supported by Google.

# In[ ]:


from keras.layers import Input, Dense, LeakyReLU, Softmax
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

# define single input
inp = Input(shape=(4,))
h = inp

# Task: add a layer with 1 neuron
h = Dense(64)(h)  # linear transformation
h = LeakyReLU()(h)  # activation
h = Dense(2)(h)  # final linear layer
h = Softmax()(h)  # softmax activation

# create an ANN model definition
model = Model(inputs=[inp], outputs=[h])

# Task: set learning rate to 100.0, 0.0000001
# this creates a C program that is called from python
model.compile(Adam(), sparse_categorical_crossentropy, ['accuracy'])

# fit the model
model.fit(X_train, y_train, epochs=10)

# evaluate the model
loss, score = model.evaluate(X_test, y_test)
print(score)


# It is also possible to use Keras models as part of your `sklearn` pipeline. This can be achieved with `keras` sklearn wrappers.

# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def make_net(n_neurons=128):
    """Defines architecture of ANN and compiles it."""
    inp = Input(shape=(4,))
    h = Dense(n_neurons)(inp)  # linear transformation
    h = LeakyReLU()(h)  # activation
    h = Dense(2)(h)  # final linear layer
    h = Softmax()(h)  # softmax activation

    model = Model(inputs=[inp], outputs=[h])
    model.compile(Adam(), sparse_categorical_crossentropy, ['accuracy'])
    return model    

# can be used as part of pipeline, and in *SearchCV
# Task: wrap in pipeline, and add scaling of feature ranges
model = KerasClassifier(make_net, n_neurons=256)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)


# In[ ]:


# Obtain example outputs. Remember, columns are:
# Pclass, Age, SibSp, Fare
# Task: what leads to increase of survival likelihood?
my_input= np.array([
    [3, 22.0, 1, 7.2500],
    [3, 20.0, 0, 10.0],
])
print(model.predict_proba(my_input))

