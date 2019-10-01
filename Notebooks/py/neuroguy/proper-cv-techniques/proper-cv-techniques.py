#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Let's explore the Titanic dataset in a semi-rigorous manner that gives a true measure
# of accuracy.  The emphasis here will be on proper cross-validation technique and the
# avoidance of data leakage, which is a very common mis-step for beginners.

df_train = pd.read_csv('../input/train.csv', header = 0) # Read in the training set
df_test = pd.read_csv('../input/test.csv', header = 0)   # Read in the test set

print("There are %d samples and %d features in the training set" %(df_train.shape[0], df_train.shape[1]))
print("There are %d Survived samples and %d Perished samples" % 
      (sum(df_train["Survived"]==0), sum(df_train["Survived"]==1)))
print("\nHere are the null counts for each feature")
print(df_train.isnull().sum(axis=0))

df_train.head()


# So we loaded in the data and can start planning the clean up.  There are several other kernels addressing exploring the data thoroughly, so we will focus on a quick and easy cleaning process.
# 
# Here are some things to note right away:
# 1.  The data is skewed.  We will need to account for this in our classifiers later.
# 2.  There are some likely unimportant columns.  PassengerId is assumed to be just a scalar identifier not correlated with the task in any way.  So that one has to go.  
# 3.   We *may* say the same thing as 1 about a couple other columns as well.  What about Name?  Is that predictive?  Probably, but how do we parse that into a meaningful variable(s)?  Perhaps some kind of tokenization and stemming??  Probably leave this out for now.  Ticket number?  This likely correlates with Pclass and Embarked.  We can leave this out for now as well.
# 4.  Some features have order and magnitude is meaningful.  Age can be left as is.  Fair as well.  What about SibSp and Parch?  Seems pretty safe to leave as is, although other kernels combine these into one 'family' variable.  Should not classifiers be able to explore these variables as one combined or the two original equally?
# 5.  Some features are categorical, such that classes are discrete and while there may be some kind of order, the magnitude between orders is unclear.  Should Pclass, Sex, Cabin, and Embarked be turned into dummy variables?  Embarked should for sure.
# 6.  There are some NaN values that will have to be imputed.

# In[ ]:


from sklearn import preprocessing

# Let's sort the columns properly
y = df_train["Survived"].as_matrix()  # Our ground truth labels

# These variables we will assume are good as is and the magnitude has meaning (although we are not sure
# exactly the scale of the axis of each of these)
X_num = df_train[["Age", "SibSp", "Parch", "Fare", "Pclass"]]

# What about the categorical features?  Which are we going to keep?
X_cat = df_train.select_dtypes(include=[object])

# Let's just drop the troublesome columns for now - Cabin contains a lot of NaN and possible
# clerical errors.  Embarked contains only 2 NaN so we can keep that.  Let's assume
# Name and Ticket are not important for now.
X_cat = X_cat.drop("Cabin", 1)
#X_cat = X_cat.drop("Embarked", 1)
X_cat = X_cat.drop("Name", 1)
X_cat = X_cat.drop("Ticket", 1)

# Because Embarked only has two missing values.
null_cols = X_cat[X_cat.isnull().any(axis=1)].index.values

# Columns 61 and 829 contain null Embarked columns
print(df_train.iloc[null_cols])

# Temporarily fill these in - Because there are only 2 missing values, we will accept a tiny bit of data
# leakage here and just fill in a value of 'C' randomly.  This makes the LabelEncoding and OneHotEncoding
# easier.  TODO: Add these step to our pipelines.
X_cat = X_cat.fillna('C')

le = preprocessing.LabelEncoder()

X_le = X_cat.apply(le.fit_transform)

enc = preprocessing.OneHotEncoder()

enc.fit(X_le)

X_enc = enc.transform(X_le)

# Append sex to the other variables
X = np.c_[X_num.as_matrix(), X_enc.toarray()]

# Repeat for test data
Xt_num = df_test[["Age", "SibSp", "Parch", "Fare", "Pclass"]]
Xt_cat = df_test.select_dtypes(include=[object])

# Let's just drop the troublesome columns for now - Cabin contains a lot of NaN and possible
# clerical errors.  Embarked contains only 2 NaN so let's come back to that one.  Let's assume
# Name and Ticket are not important for now.
Xt_cat = Xt_cat.drop("Cabin", 1)
#Xt_cat = Xt_cat.drop("Embarked", 1)
Xt_cat = Xt_cat.drop("Name", 1)
Xt_cat = Xt_cat.drop("Ticket", 1)

Xt_le = Xt_cat.apply(le.fit_transform)
enc.fit(Xt_le)

Xt_enc = enc.transform(Xt_le)

# Append sex to the other variables
Xt = np.c_[Xt_num.as_matrix(), Xt_enc.toarray()]

print(X.shape)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

X_name = df_train["Name"]

cv = CountVectorizer();

X_dict = cv.fit_transform(X_name).toarray()

# Looks like 1509 unique words in the "Name" feature.  We will have to be more intelligent about this.
#cv.get_feature_names()
X_counts=X_dict.sum(axis=0)
X_freq=np.argsort(X_counts)[::-1]
top = X_freq[:100]

tokes = np.asarray(cv.get_feature_names())

# Looking at the top 100, we can pick out some interesting common features that may be important
tokes[top]


# The question here is are we causing data leakage by looking at the Name data this way?  Actually, probably yes.  This is likely very minor, but if we think about unseen data, perhaps it would not have this level of "Mr" and "Mrs".   So we could one-hot encode the most intuitive words, BUT, by doing so we are optimistically biasing our cross validation scores in theory.
# 
# However, in this case we can probably be fairly safe in picking titles as one-hot encoded features.  The point of this rigourous analysis is not that we cannot do certain pre-processing, but that at each step we should be asking ourselves to what level this pre-processing is likely to affect our scores.  Using our knowledge of the field as a whole (in this case the use of titles in the culture of the time) can inform our decisions.
# 
# So, we will encode titles, but not names, but realizing that there is a very small possibility of data leakage that we have thought about.  Think of analogies to this case in your more real world examples.

# In[ ]:


# Pull out columsn for titles. mr, miss, mrs, master, jr, and dr.

named_cols = ['mr', 'miss', 'mrs', 'master', 'jr', 'dr', 'rev']

title_cols = [cv.vocabulary_[x] for x in named_cols]

X_dict = X_dict[:, title_cols]

# Append name to the other variables
X = np.c_[X, X_dict]


# In[ ]:


# 6 new variables added
X.shape


# In[ ]:


# Do the same thing for the test set
X_name = df_test["Name"]

cv = CountVectorizer();

X_dict = cv.fit_transform(X_name).toarray()

# Looks like 1509 unique words in the "Name" feature.  We will have to be more intelligent about this.
#cv.get_feature_names()
X_counts=X_dict.sum(axis=0)
X_freq=np.argsort(X_counts)[::-1]
top = X_freq[:100]

tokes = np.asarray(cv.get_feature_names())

title_cols = [cv.vocabulary_[x] for x in named_cols]

X_dict = X_dict[:, title_cols]

# Append name to the other variables
Xt = np.c_[Xt, X_dict]

Xt.shape


# In[ ]:


# Data visualization - I'm not going to explore the distributions of the data as this is covered
# extensively in other kernels.  But, let's reduce the dimensionality of the data and check for
# natural distance in the data we have created.
#
# This will require some pre-processing: Impute NaN values and normalize.  PCA expects equal
# weighting of features (TODO: I am not sure about TSNE requirements on features)
#
# TSNE will be the default dimensionality reducer. 
# TSNE generally gives nice results.  Perplexity can be manipulated (between 5-50) for
# better visualization, but generally does not affect results.
# Note that it can be computationally expensive so PCA can be used instead, although the results
# generally will not be as nice.

from sklearn import manifold
from sklearn.decomposition import PCA

perplexity=10
n_components=2

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
Xi = imp.fit(X).transform(X)

pca = PCA(n_components=2)

scalar = preprocessing.StandardScaler()
scalar.fit(Xi)

X_t= scalar.transform(Xi)

tsne = manifold.TSNE(n_components=n_components, init='random',
                         random_state=0, perplexity=perplexity)


#Y = pca.fit_transform(X_t)
Y = tsne.fit_transform(X_t)


# In[ ]:


# Plot our visualization results
# Plot the data in 3 dimensions
import matplotlib.pyplot as plt

classes = ['Survived', 'Perished']

fig = plt.figure(figsize=(10,10))

plt.scatter(Y[y==0, 0], Y[y==0, 1], c="b", s=100, alpha = 0.5)
plt.scatter(Y[y==1, 0], Y[y==1, 1], c="orange", alpha = 0.5, s=100, edgecolors = 'black')

plt.legend(classes)

plt.show()


# There is defintitely some nice structure to this data and natural distances.  There are distinct subgroups and this would be worth further exploration I'm sure, but there is enough here for standard classifers to work with.  On the other hand, given our very basic data cleaning, we can see some clear overlap in classes and so we should not set our hopes too high on this first pass.
# 
# **Proper Cross Validation for building a model of Titanic survival prediction**
# 
# This is the meat of this kernal.  We want to estimate what accuracy we can expect on the test set, where we do not know the ground truth labels.  We want a *realistic* score however. In fact, we would rather have a slightly pessimistic score, over a very optimistic score.  If we structure our cross-validation properly, we should get a very close estimate of what we should expect on unseen data.
# 
# Preventing an overly-optimistic score requires careful thoughts about potential sources of data leakage.  Data leakage occurs where steps in your model building process essentially transfer knowledge about the ground truth labels of training set data to, assumed pure representation of unseen data, test sets.  It is very easy to cause this leakage without realizing you are doing it.  *ANY* step along your way from raw data -> model should be suspect.  You should always ask yourself if what you are doing has any abilty to optimisitically bias your hold out test sets during cross validation.
# 
# There are several sources of data leakage and they have varying degrees of influence on your overall score, but here are some main areas that are often overlooked:
# 
# 1.  Pre-processing.  Almost all data will require some pre-processing.  Let's think about what that usually involves.  First we want to deal with missing data.  That will require imputation based on the filled in data if we want to use certain classifiers such as SVMs..  But, if we impute the missing value from the entire training set, we are transferring knowledge about the entire dataset to that missing value, and hence to that sample.  That could positively bias any sample with a missing value.  If the data is skewed, that could positively influence those samples if the samples are from the same class as the positively skewed class.   It will not matter then if you cross validate, as these sample will end up in your test sets, and they carry with them information about samples in your training sets.  This is a more minor concern, but we could definitely get unlucky here and create real optimisitic bias in our model built on our entire training data. The same problems occur with other common pre-processing techniques, such as: Feature Selection using techniques such as Decision Trees or Mutual Information, or Dimensionality Reduction using something like PCA or LDA.
# 
# 2.  Oversampling minority classes to ameliorate the skew in your data.  This is generally done by an algorithm such as SMOTE, or by replication of classes in the minority class.  This is technically still pre-processing, but you should be able to see why this is a much more serious source of data leakage.  If you are replicating data, even if it is randomly generated based on existing data, you are introducing samples that *heavily* carry information about other samples in that class.  If you try this on heavily skewed data, you will of course instantly get amazing scores.... that mean absolutely nothing in the real world of unseen data.
# 
# 3.  Hyperparameter Tuning.  Of course we want to separate our data into folds, so that we can have a cross validation set to test our built models.  However, we also want to tune our model's hyperparameters.  Most classifer models of interest require some hyperparameter tuning, especially for regularization.  So, we will need *ANOTHER* cross validation, inside our first cross validation, to prevent the test set data from leaking into our hyperparameter tuning.  Can you see why?
# 
# If we want to grid search the parameters of our model, we will need to cross validate.  We hold out a test set, we try one set of parameters on the grid, and we score.  We repeat that for our K-folds, and then move onto the next point on the grid.  In the end we get a cross validated score of each grid point.  Then we pick the best grid point and we are good to go, right?  *NO*  We are definitely not good to go.  Because our hyper parameter search involved the entire training set on each grid point, samples are in and out of the training fold on each grid point and the choice of hyperparameters is being tuned and evaluated with the same data.
# 
# Hence, we need to nest two folds of cross validation.  The outer CV holds out a test fold and passes on K-1 training folds for model building .  The inner CV operates on the K-1 folds training set from the outer CV.  The inner CV hold out another test set and passes on another K-1 training folds to the model tuning grid search in turn as described above.  This way we are getting a realistic view of completely naive data on each outer CV fold.
# 
# The downside to all of this of course is that, from preprocessing to hyper parameter tuning, we are always calculating on less that the full data.  Such is life.  This gives us a slighly more pessimistic score than we would expect to realize on unseen data when we ultimately end up creating our final model on the entire training set.  Yep, that is right.  Once we are done evaluating our model, we will of course refit our model to the entire dataset to maximize our score on unseen data.  We just have no way of actually knowing how much better this score will be than the one we just estimated using our nested CV.
# 
# Fortunately, sklearn makes all of this extremely easy to implement in practice.  So let's do that now.  See [http://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html](http://), who I credit with the basic structure of the following code.

# In[ ]:


# Set up a nested cross-validation pipeline on our cleaned Titanic dataset.  Note that I was
# careful to say cleaned, and not pre-processed.  We have done nothing to the data other than
# inspect it, throw away clearly clerical or overly sparse data, and one-hot encode categorical 
# data.  Nothing about the structure/distribution of the data has been used to help clean the
# data thus far.  We did run some pre-processing for visualization processes, but then threw away
# that analysis.  X contains our clean feature data.  y contains our target data.
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Logistic Regression
from sklearn.linear_model import LogisticRegression 

# MLP
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline as Pipeline_SK
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Set up a stratified CV for both outer and inner CVs.  Stratified because we want to test on
# the real distribution of our skewed data - Just going to use 3 for speed.  Increase this
# for more realistic results
inner_cv = StratifiedKFold(n_splits=5, shuffle=True)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True)

# Standard svm with a rbf kernel and balanced weights to account for the skewed data.  We pick
# a regularization constant of 9, but we will tune this.
svm = SVC(kernel='rbf', class_weight='balanced', C=9, random_state=1, probability=True)
lr = LogisticRegression(class_weight='balanced', C=9, random_state=1)
gnb = GaussianNB()
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.1)

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)

# Add this lda to the pipe if you like.
lda = LDA(solver='eigen', shrinkage='auto')

p_grid_svm = {"svm__C": [0.05, 0.1, 0.2, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]}
p_grid_lr = {"lr__C": [0.05, 0.1, 0.2, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]}
p_grid_gnb = {}
p_grid_mlp = {"mlp__hidden_layer_sizes": [(100,), (50,), (100, 50,), (50, 20,)],
             "mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2]}

# We are setting up separate pipes and grids for each classifier, but you can do this on the Voting
# Classifier as well and just index over one grid as v_svm__svm__C, v_lr__lr__C, etc...

# Parameter tuning - inner CV - SVM
pipe = Pipeline_SK([('imp', imp), ('standardscalar', preprocessing.StandardScaler()), ('svm', svm)])
clf_svm = GridSearchCV(estimator=pipe, param_grid=p_grid_svm, cv=inner_cv, scoring='accuracy')

# Parameter tuning - inner CV - LR
pipe = Pipeline_SK([('imp', imp), ('standardscalar', preprocessing.StandardScaler()), ('lr', lr)])
clf_lr = GridSearchCV(estimator=pipe, param_grid=p_grid_lr, cv=inner_cv, scoring='accuracy')

# Parameter tuning - inner CV - MLP
pipe = Pipeline_SK([('imp', imp), ('standardscalar', preprocessing.StandardScaler()), ('mlp', mlp)])
clf_mlp = GridSearchCV(estimator=pipe, param_grid=p_grid_mlp, cv=inner_cv, scoring='accuracy')

# Parameter tuning - inner CV - GB - No Parameters to tune so we won't use the inner loop
pipe = Pipeline_SK([('imp', imp), ('standardscalar', preprocessing.StandardScaler()), ('lda', lda), ('gnb', gnb)])
clf_gnb=pipe

# Set up a voting classifier on the probability outputs of our 4 sample classifiers.  This should 
# generally give us a few more percent.  A decision tree would be a good addition to this.
eclf2 = VotingClassifier(estimators=[('v_svm', clf_svm), ('v_lr', clf_lr), ('v_mlp', clf_mlp), ('v_gnb', clf_gnb)], voting='soft')

# Measure scores - outer CV
ns_vote = cross_val_score(eclf2, X=X, y=y, cv=outer_cv, scoring='accuracy')
ypred_vote = cross_val_predict(eclf2, X, y, cv=outer_cv)


# In[ ]:


from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score)

nested_score=ns_vote
ypred=ypred_vote

print(nested_score)

print("Accuracy score to expect from an unseen dataset: %0.03f (+/- %0.03f)" % (nested_score.mean(), nested_score.std()))

print("Accuracy score to expect from an unseen dataset: %0.03f %%" % (100*accuracy_score(y, ypred)))


cm = confusion_matrix(y, ypred)

print(classification_report(y, ypred, target_names=classes))


# Well, 82% with a <1% variation.  Not bad and about in line with what I have read about this dataset.  The nested scores look quite stable.  Precision and recall on the Perished class are not amazing.  We definitely skipped some steps that could give a few extra %.  Namely:
# 
# 1)  Cabin category.  How do we deal with missing values in categorical data without introducing bias?   Do the few filled in columns of cabin help and if so how do we properly encode them while filling in the NaNs in a meaningful way?
# 
# 2)  Some more complicated analysis of name?  There are more complicated family relationships for sure.
# 
# 3)  Ticket?  Again, ticket number likely correlates very highly with class.
# 
# 4)  Other classifiers.  We can try the ever popular XGBoost.  We can try deeper grid searches.  We can try stacking.  We can likely eek out a few more % in this data, but I have read that it maxes out around 84% using exhaustive searches with the techniques presented here.  The TSNE evaluation appears to agree with this assessment.  Feature engineering will likely be needed to push this data past 84%, if it is possible at all.
# 
# That being said, let's build our final model.
# 
# 

# In[ ]:


clf_svm.fit(X, y)


# In[ ]:


# And predict
ypred_test = clf_svm.predict(Xt)

ypred_test


# In[ ]:


# And save in the proper format
# One column each of PassengerId	Survived
data = pd.DataFrame(df_test['PassengerId'])

data['Survived'] = ypred_test

data.to_csv('submission_svm.csv', index=False)

