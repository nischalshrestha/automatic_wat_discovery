#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **RRC**: First of all, load our dataset.

# In[ ]:


traindf = pd.read_csv("../input/train.csv")


# **RRC**: And see what we have at the tail

# In[ ]:


traindf.tail(10)


# ## **Preprocessing**

# ## Identification of missings
# 

# In[ ]:


#traindf.describe()
traindf.isnull().sum()


# **RRC**: Surprise! Age feature has nulls. Let's work on that. 

# **WARNING**: number 0 is not a missing value. Also, " " sometimes is not a missing value. It depends on the context! Big mistake impute 'null' values as 0. **Reminder** see number 0 as any other value that could bias a prediction. 

# * Identifying missing values (Typically NaN (Not a Number), null, empty cell in excel spreadsheet, etc.)
# * Eliminating samples or features with missing
# * Imputing missing values

# ## Record dropping

# In[ ]:


traindf.dropna().describe()


# **RRC**: Uh oh! Wait! 183!? That's too much. Why?
# Let's try just removing those features with nulls. 

# ## Feature dropping

# In[ ]:


traindf.dropna(axis=1).describe()


# ## Imputer

# **RRC**: Ok, let's fill (impute) null numerical values in Age. Our strategy is using the 'mean'. Naive, but secure. 

# In[ ]:


traindf['Age'].describe()


# In[ ]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
traindf['Age'] = imp.fit_transform(traindf[['Age']])

traindf['Age'].describe()

#imr = Imputer(missing_values=np.nan, strategy='mean', axis=1)
#traindf['Age'] = imr.fit_transform(traindf['Age']).T

#traindf.describe()


# ## Scaling ordinal features
# 

# **RRC**: Our Fare and Age feautres are completely different. Check Max, quartiles, mins, and distribution are not so homogeneous. Let's fix that. 

# In[ ]:


traindf[['Fare', 'Age']].describe()


# **RRC**: Let's normalize that. 

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
traindfScaled = mms.fit_transform(traindf[['Fare', 'Age']])
traindf[['Fare', 'Age']] = traindfScaled
traindf[['Fare', 'Age']].describe()


# **RRC**: Check! min and max once scalled. Now, Fare and Age are numbers between 0 and 1. This is nice for a good training. 

# ## Encoding of categorical variables (some examples)

# **RRC**: What would happen if we try to encode the Ticket column? 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(traindf['Ticket'].values)
y


# **RRC**: just kidding. Plenty of values. But a good example ;D

# ## One hot encoding
# 
# **RRC**: Onehot encoding is a nice technique to encode variables. But warning! with this code this does not encode missing values. Try to fix that before you use this. 
# 
# Check variables like Sex and Embarked. Let's one-hot encode them. 

# In[ ]:


traindf.head()


# In[ ]:


traindf = pd.get_dummies(traindf, columns = ['Embarked', 'Sex'])
#Once encoded let's see. 
traindf.head()


# ## Transformations

# In[ ]:


#The most simple one :D
traindf['ratioFareClass'] = traindf['Fare']/traindf['Pclass']

traindf.head()


# ## Partitioning a dataset into separate training and test sets

# In[ ]:


features = ['Age', 'Fare', 'Pclass', 'Embarked_C', 'Embarked_S', 'Embarked_Q', 'Sex_female', 'Sex_male', 'ratioFareClass']
traindf['Survived'].value_counts()


# **RRC**: Target variable is well balanced. Let's split dataset with a test size as 30% stratifying the target class

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(traindf[features],
                                                    traindf["Survived"],
                                                    test_size=0.3,
                                                    stratify=traindf['Survived'])

# test size as a 30%
# stratify on survived


# In[ ]:


y_test.value_counts()


# **RRC**: Check survived=1 is exactly a 30% of the total on y_test
# 
# Good. Let's see our final dataset and some statistics about it. 

# In[ ]:


traindf[features].head()
# Features is the final feature list we are going to use. 


# In[ ]:


traindf.describe()


# **RRC**: It seems everything is Ok! so we can start playing with our data training different models. 

# ## Training 

# ### Logistic Regression Classifier

# In[ ]:


#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X_train, y_train)


print("Logistic Regression score (Train): {0:.2}".format(lr.score(X_train, y_train)))
print("Logistic Regression score (Test): {0:.2}".format(lr.score(X_test, y_test)))


# ### KNN Classifier

# In[ ]:


#http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
print("KNN score (Train): {0:.2}".format(neigh.score(X_train, y_train)))
print("KNN score (Test): {0:.2}".format(neigh.score(X_test, y_test)))


# ### Support Vector Machines (Support Vector Classifier)

# In[ ]:


#http://scikit-learn.org/stable/modules/svm.html
from sklearn import svm
svclass = svm.SVC(gamma='scale')
svclass.fit(X_train, y_train) 
print("SVM score (Train): {0:.2}".format(svclass.score(X_train, y_train)))
print("SVM score (Test): {0:.2}".format(svclass.score(X_test, y_test)))


# ### Decision tree

# In[ ]:


#http://scikit-learn.org/stable/modules/tree.html
from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)
print("Decision Tree score (Train): {0:.2}".format(dt.score(X_train, y_train)))
print("Decision Tree score (Test): {0:.2}".format(dt.score(X_test, y_test)))


# ### Random Forest Classifier

# In[ ]:


# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100,
                                criterion='gini',
                                max_depth=5,
                                min_samples_split=10,
                                min_samples_leaf=5,
                                random_state=0)
X_train.head()
forest.fit(X_train, y_train)
print("Random Forest score (Train): {0:.2}".format(forest.score(X_train, y_train)))
print("Random Forest score (Test): {0:.2}".format(forest.score(X_test, y_test)))


# ### Model Evaluation

# In[ ]:


model = forest


# In[ ]:


#WAAAAARNING: not all models have the "feature_importances_" functions
plt.bar(np.arange(len(features)), model.feature_importances_)
plt.xticks(np.arange(len(features)), features, rotation='vertical', ha='left')
plt.tight_layout()


# In[ ]:


X_test

model = forest
# This is an example! Also a bad practise :D
#AGE, SEX, AGE_NAN, PClass, FARE
testcase = np.array([[25, 12, 3, 0, 1, 0, 1, 0, 1]])
prediction = model.predict(testcase)[0]
pproba = model.predict_proba(testcase)[0]
print("Prediction for test case: %s (perish -> %.2f, surv -> %.2f)" %
      ('PERISH' if prediction == 0 else 'SURVIVED!', pproba[0], pproba[1]))


# ## Some regression examples

# ### Ex 1: Diabetes dataset

# In[ ]:


# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
# -----------------------------------------------
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# ### Ex 2: Polynomial interpolation

# In[ ]:


#http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html
# Author: Mathieu Blondel
#         Jake Vanderplas
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x)


# generate points used to plot
x_plot = np.linspace(0, 10, 100)

# generate points and keep a subset of them
x = np.linspace(0, 10, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:20])
y = f(x)

# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

colors = ['teal', 'yellowgreen', 'gold']
lw = 2
plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=lw,
         label="ground truth")
plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")

for count, degree in enumerate([3, 4, 5]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
             label="degree %d" % degree)

plt.legend(loc='lower left')

plt.show()


# ### Ex 3: Isotonic Regression

# In[ ]:


#http://scikit-learn.org/stable/auto_examples/plot_isotonic_regression.html#sphx-glr-auto-examples-plot-isotonic-regression-py
# Author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state

# Data is genereted randomly
n = 100
x = np.arange(n)
rs = check_random_state(0)
y = rs.randint(-50, 50, size=(n,)) + 50. * np.log1p(np.arange(n))

# #############################################################################
# Fit IsotonicRegression and LinearRegression models

ir = IsotonicRegression()

y_ = ir.fit_transform(x, y)

lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression

# #############################################################################
# Plot result

segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(np.full(n, 0.5))

fig = plt.figure()
plt.plot(x, y, 'r.', markersize=12)
plt.plot(x, y_, 'g.-', markersize=12)
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.gca().add_collection(lc)
plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
plt.title('Isotonic regression')
plt.show()


# In[ ]:




