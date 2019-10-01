#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_test, 
                             y = y_test.astype(int), cv = 10, scoring = 'precision')
print("Accuracy mean " + str(accuracies.mean()))
print("Accuracy std " + str(accuracies.std()))


# std seems high
# 
# Before changing algorithm, let's try to work on features
# 
# *Feature selection* using RFE (recursive feature elimination)
#  
# 

# In[ ]:


from sklearn.feature_selection import RFE 

rfe = RFE(classifier, 6)
rfe = rfe.fit(X_test, y_test.astype(int))
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)


# Hello everybody,
# 
# this is my first notebook/competition and I hope to have feedbacks about what I'm doing (especially wrong things).
# 
# I haven't seen other submissions, as I want to start from scratch and see what I can find
# 
# I'm very fascinated by ML and I'm eager to learn as much as possible 
# 
# Ok, let's start!
# 
# Besides the results, what I'll like to do is to establish a correct general workflow helping to work with all datasets
# 
# The steps:
# 
# 
# 
# 1) Inspect the data to have a first guess of features, relations, instances quality and draw some graph helping to visualize them
# 
# 2) Do some preprocessing (get rid of nan, categorical feature encoding, feature scaling - if necessary)
# 
# 3) Further analysis
# 
# 4) Build a baseline classifier (Logistic Regression in this case) just to have a starting point
# 
# 5) Do features selection and engineering to improve results
# 
# 6) Repeat from step 2 with another approach (algorithm, features, etc) until complete satisfaction :)

# In[ ]:


# Importing some libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the train dataset from file
dataset = pd.read_csv('../input/train.csv')

#Some info about it
dataset.info()

dataset.isnull().sum()

dataset.describe()


# Let's see what we have 
# 
# PassengerId: meta
#  
# Survived: target 
# 
# Pclass: feature (seems important, based on position probably) 
# 
# Name: meta
#  
# Sex: feature (not sure how can impact on surviving an iceberg hit :)) 
# 
# Age: feature (maybe target related) 
# 
# Sibsp, Parch: (seem important, an event happening to all the people in a group) 
# 
# Fare: maybe related to class 
# 
# Ticket, Cabin, Embarked: not related, just meta
# 
# 
# Rows number seems ok respect the features 
# 
# Age is missing on 20% data, we'll see how to deal it

# In[ ]:



# Let's explore the data visually against the target

survived_pclass = pd.crosstab([dataset.Pclass], dataset.Survived.astype(bool))
survived_pclass.plot(kind='bar', stacked=False, color=['red','blue'], grid=False)

survived_sex = pd.crosstab([dataset.Sex], dataset.Survived.astype(bool))
survived_sex.plot(kind='bar', stacked=False, color=['red','blue'], grid=False)

survived_sibsp = pd.crosstab([dataset.SibSp], dataset.Survived.astype(bool))
survived_sibsp.plot(kind='bar', stacked=False, color=['red','blue'], grid=False)

survived_parch = pd.crosstab([dataset.Parch], dataset.Survived.astype(bool))
survived_parch.plot(kind='bar', stacked=False, color=['red','blue'], grid=False)

plt.show()


# So male, with 3rd class and alone is the victim type
# High SibSp too seems very deadly :(
# 
# Ok, time to preprocess for further analysis

# In[ ]:


#get all relevant columns
workingDataset = dataset.iloc[:, [1,2,4,5,6,7,9]]

# get rid of age nan rows (first approach)
workingDataset = workingDataset[np.isfinite(workingDataset['Age'])]

# feature/target selection

workingData = workingDataset.values
X = workingData[:, 1:]
y = workingData[:, 0]

# encoding feature (sex)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# avoid dummy trap
X = X[:, 1:]

from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
sc = StandardScaler()
preprocessedData = sc.fit_transform(X)
# rebuild feature's dataframe with normalized data for graphs purpose
preprocessedDataset = DataFrame(data=preprocessedData)
preprocessedDataset.columns = ['Sex','Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

preprocessedDataset.describe()


# In[ ]:


def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

colors = np.where(dataset.Survived == 1, 'blue', 'red')
plt.scatter(x=rand_jitter(dataset.Parch), y=rand_jitter(dataset.SibSp), c = colors)
plt.xlabel('Parch')
plt.ylabel('SibSp')



# In[ ]:


plt.scatter(x=rand_jitter(preprocessedDataset.Age), y=rand_jitter(preprocessedDataset.Fare), c = colors)
plt.xlabel('Age')
plt.ylabel('Fare')


# In[ ]:


plt.boxplot(preprocessedData)
plt.xlabel("Attribute Index")
plt.ylabel(("Quartile Ranges - Normalized "))


# In[ ]:


#parallel coordinates
nRows = len(preprocessedDataset.index)
nCols = len(preprocessedDataset.columns)


nDataCol = nCols
for i in range(nRows):
   #assign color based on "1" or "0" labels
   if y[i] == 1:   #survived
      pcolor = "blue"
   else:
      pcolor = "red"
   #plot rows of data as if they were series data
   dataRow = preprocessedDataset.iloc[i,0:nDataCol] 
   dataRow.plot(color=pcolor, alpha=0.5)

plt.xlabel("Attribute Index")
plt.ylabel(("Attribute Values"))
plt.show()


# 
# Low correlation betwen features
# Fare with some outliers, age should be ok...let's have confirmation with probplots

# In[ ]:


import scipy.stats as stats
import pylab

col = 5 
colData = []
for row in X:
   colData.append(float(row[col]))

stats.probplot(colData, dist="norm", plot=pylab)
pylab.show()

col = 2 
colData = []
for row in X:
   colData.append(float(row[col]))

stats.probplot(colData, dist="norm", plot=pylab)
pylab.show()


# In[ ]:


corMat = DataFrame(preprocessedDataset.corr())

#visualize correlations using heatmap
plt.pcolor(corMat)
plt.show()


# Correlation is low
# 
# Time to build baseline classifier with Logistic Regression and simple split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(preprocessedData, y, 
                                                    test_size = 0.25, random_state = 0)

y_test = y_test.astype(int)
y_train = y_train.astype(int)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import seaborn as sn
sn.heatmap(cm, annot=True)


# mmm I'm sure can be better...
# 
# Let's check the accuracy doing k-fold cross validation

# In[ ]:


from sklearn.model_selection import cross_val_score

accuracy = cross_val_score(estimator = classifier, X = X_test, 
                             y = y_test, cv = 10, scoring = 'accuracy')

print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))


# std seems high
# 
# Before changing algorithm, let's try to work on features
# 
# *Feature selection* using RFE (recursive feature elimination)
#  
# 

# In[ ]:


from sklearn.feature_selection import RFE

rfe = RFE(classifier, 6)
rfe = rfe.fit(X_test, y_test)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)


# Feature engineering using PCA
# 
# (but should not work given the result of RFE)

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_pca, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_pca)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sn.heatmap(cm, annot=True)

accuracy = cross_val_score(estimator = classifier, X = X_test_pca, 
                             y = y_test, cv = 10, scoring = 'accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))




# In[ ]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_test_pca, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red',  'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# Let's try LDA

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_lda, y_train)

# Predicting the Test set results
y_pred_lda = classifier.predict(X_test_lda)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_lda)
sn.heatmap(cm, annot=True)

accuracy = cross_val_score(estimator = classifier, X = X_test_lda, 
                             y = y_test, cv = 10, scoring = 'accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))


# ok, let's finish with kernel-pca using not linear approach

# In[ ]:


from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 5, kernel = 'rbf')
X_train_kpca = kpca.fit_transform(X_train)
X_test_kpca = kpca.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_kpca, y_train)

# Predicting the Test set results
y_pred_kpca = classifier.predict(X_test_kpca)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_kpca)
sn.heatmap(cm, annot=True)

accuracy = cross_val_score(estimator = classifier, X = X_test_kpca, 
                             y = y_test, cv = 10, scoring = 'accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))

