#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this notebook, we will try to optimize logistic regression to predict survival rate on the Titanic.
# 
# Logistic regression has two main parameters that can be tuned:
# 
#  - lambda: the regularization parameter
#  - the polynomial degree
# 
# For more information, please refer to is a wonderful course on coursera.org which explains Logistic Regression very well: Machine Learning by Pr Andrew Ng.
# 
# Of course, comments on this work are more than welcome!

# # Reading data

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import re as re
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})
original_test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
test = original_test.copy(deep=True)
full_data = [train, test]

print (train.info())


# # Preparing data for resolution
# 
# In this notebook, we will not get into the details of feature engineering. We will make some classic processing on the features. There may be several feature optimization that could still be done, but it is not the purpose of this notebook.
# 
# ## Family size categories
# Let's create LargeFamily (2 classes: 1 to 4, 5 and up)

# In[ ]:


for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['LargeFamily'] = dataset['FamilySize'].apply(lambda r: 0 if r<=4 else 1)


# 
# 
# ## Names ##
# inside this feature we can find the title of people.

# In[ ]:


def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

print(pd.crosstab(train['Title'], train['Sex']))


#  so we have titles. let's categorize it and check the title impact on survival rate.

# In[ ]:


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


# # Other data
# Now let's clean all other fields and map our features into numerical values.

# In[ ]:


for dataset in full_data:   
    # Fill missing values in Embarked with most frequent port 'S'
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
    # Fill missing values in Fare with median
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    # Fill missing values in age with random data based on mean and standard variation
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
        
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
    
# Feature Selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',                 'Parch', 'FamilySize']
train = train.drop(drop_elements, axis = 1)

test  = test.drop(drop_elements, axis = 1)

print (train.head(10))


# Good! now we have a clean dataset.

# Now let's find which classifier works better on each dataset. 

# # Classifier Comparison #
# 
# ## Logistic regression ##
# 
# Let's try different values for:
# 
#  - lambda: the regularization term
#  - the degree of the polynomial combination of features

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

n_splits = 10
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=0)

X = train.values[0::, 1::]
y = train.values[0::, 0]

log_cols = ["lambda", "Poly Degree", "Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)

acc_dict = {}

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    for lambd in [0.0001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100]:
        for poly_degree in range(1,4):
            
            # Create polynomial features
            poly = PolynomialFeatures(poly_degree)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.fit_transform(X_test)

            clf = LogisticRegression(C=1/lambd)        
            clf.fit(X_train_poly, y_train)
            train_predictions = clf.predict(X_test_poly)
            acc = accuracy_score(y_test, train_predictions)
            if lambd in acc_dict:
                if poly_degree in acc_dict[lambd]:
                    acc_dict[lambd][poly_degree] += acc
                else:
                    acc_dict[lambd][poly_degree] = acc
            else:
                acc_dict[lambd] = {}
                acc_dict[lambd][poly_degree] = acc


for lambd in acc_dict:
    for poly_degree in acc_dict[lambd]:
        acc_value = acc_dict[lambd][poly_degree] / n_splits
        log_entry = pd.DataFrame([[lambd, poly_degree, acc_value]], columns=log_cols)
        log = log.append(log_entry)

#print ('Classifier Accuracy')
#print (log)
#print ()

plt.figure()

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

heatmap_data = log.pivot("lambda", "Poly Degree", "Accuracy")
ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f')


# Great!
# 
# We can see that purely linear Logistic Regression is not optimal. It has too much bias, and that we can significantly improve results by choosing a degree 2 or more polynomial logistic regression.

# ## Support Vector Machine ##
# Let's do the same job for Support Vector Machine classification.
# For support Vector Machine, we do not need to use polynomial features. But we have another parameter that can be tuned: the gamma of the kernel function.
# 
# So let's try different values for:
# 
#  - lambda: the regularization term
#  - gamma: parameter of the kernel function
# 
# Please note that for SVM, by convention, we use C=1/lambda. But here we will stick with lambda, to be coherent with logistic regression.

# In[ ]:


from sklearn.svm import SVC

log_cols = ["lambda", "gamma", "Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)

acc_dict = {}

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    for lambd in [0.0001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100]:
        for gamma in [1.0E-3, 1.0E-2, 1.0E-1, 1.0, 10.0, 1.0E3]:
            
            clf = SVC(probability=True, C=1/lambd, gamma=gamma)        
            clf.fit(X_train, y_train)
            train_predictions = clf.predict(X_test)
            acc = accuracy_score(y_test, train_predictions)
            if lambd in acc_dict:
                if gamma in acc_dict[lambd]:
                    acc_dict[lambd][gamma] += acc
                else:
                    acc_dict[lambd][gamma] = acc
            else:
                acc_dict[lambd] = {}
                acc_dict[lambd][gamma] = acc


for lambd in acc_dict:
    for gamma in acc_dict[lambd]:
        acc_value = acc_dict[lambd][gamma] / n_splits
        log_entry = pd.DataFrame([[lambd, gamma, acc_value]], columns=log_cols)
        log = log.append(log_entry)

#print ('Classifier Accuracy')
#print (log)
#print ()

plt.figure()

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

heatmap_data = log.pivot("lambda", "gamma", "Accuracy")
ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f')


# We can see that gamma=0.1 is a good choice, and that the model doesn't seem to need much regularization.
# 
# # Prediction with SVC #
# Let's use SVC with parameters lambda=0.3 and gamma=0.1 to predict our data.
# Please note that optimal parameters can vary each time you run this notebook. This is because the training and cross-validation sets are randomly choosen.

# In[12]:


# Use candidate classifier
lambd = 0.3
gamma = 0.1
candidate_classifier = SVC(probability=True, C=1/lambd, gamma=gamma)   

# Create polynomial features
X_train = train.values[0::, 1::]
y_train = train.values[0::, 0]
X_test = test.values

candidate_classifier.fit(X_train, y_train)
surv_pred = candidate_classifier.predict(X_test)

submit = pd.DataFrame({'PassengerId' : original_test.loc[:,'PassengerId'],
                       'Survived': surv_pred.T})
submit.to_csv("../working/submit_svc_gamma01.csv", index=False)


# # Prediction with logistic regression #
# Let's use logistic regression with lambda=0.03 and poly_degree=3 to predict our data.

# In[ ]:


# Use candidate classifier
lambd = 0.03
poly_degree = 3
candidate_classifier = LogisticRegression(C=1/lambd)

# Create polynomial features
X = train.values[0::, 1::]
y = train.values[0::, 0]
poly = PolynomialFeatures(poly_degree)
X_train_poly = poly.fit_transform(X)
X_test_poly = poly.fit_transform(test.values)

candidate_classifier.fit(X_train_poly, y)
surv_pred = candidate_classifier.predict(X_test_poly)

submit = pd.DataFrame({'PassengerId' : original_test.loc[:,'PassengerId'],
                       'Survived': surv_pred.T})
submit.to_csv("../working/submit_logistic_poly_3.csv", index=False)


# # Check submission file #

# In[ ]:


submit.head()


# In[ ]:


submit.shape


# # Conclusion #
# On our dataset, we can improve significantly logistic regression by using polynomial combination of features. For SVM, the optimal values are close to the default values, so there is no great improvement to expect from optimization of sigma and lambda.
