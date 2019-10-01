#!/usr/bin/env python
# coding: utf-8

# # Another Titanic Kernel
# Just as the title says, part of my learning process. Obtained scores are indicative, but reasonable constant across iterations and changes.

# ## Import required modules

# In[ ]:


import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, linear_model, neighbors, tree, ensemble,                     neural_network, svm, gaussian_process, naive_bayes
import scipy.stats as st
import pickle


# ## Load data files

# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# ## Basic data exploration. 

# In[ ]:


train.head(3)


# In[ ]:


basic_expl=pd.concat([train.dtypes, train.isnull().sum(), test.isnull().sum()], axis=1)
basic_expl.columns=['type', 'NaN_train', 'NaN_test']
basic_expl.T


# ## Basic data exploration. Results.
# Attributes considered of little value for prediction:
# -  **PassengerId**: arbittary sequential number, has no significance.  
# 
# For the other variables, we'll perform the following actions:
# -  **Age**: We need to fill the NaN values somehow and cluster/bin. 
# -  **Cabin**: Need to make binary. 0 if no cabin (NaN), 1 if passenger has cabin. Should be highly correlated to Pclass and Fare.
# -  **Embarked**: C = Cherbourg = 0, Q = Queenstown = 1, S = Southampton = 2. Use mode of same Pclass if NaN.
# -  **Fare**: Only 1 NaN value, we fill with the mean/mode of the same Pclass group.
# -  **Name**: name titles could imply (social) class, so weĺl extract the titles from there and code them numerically. 
# -  **Sex**: Need to make binary. 0 for male, 1 for female.  
# -  **Ticket**: Repeated tickets could indicate a family group, and family groups (or at least individual members of those groups) could have different chances of survival. Single=0, Family=1

# In[ ]:


# Define a working copy of the original datasets
train_filtered = train.copy()
test_filtered = test.copy()


# In[ ]:


# Code all varibales into numeric features
# Cabin: NaN=0, Other=1
train_filtered['Cabin'] = np.where(train_filtered['Cabin'].isnull(), 0, 1)
test_filtered['Cabin'] = np.where(test_filtered['Cabin'].isnull(), 0, 1)

# Embarked: C = Cherbourg = 0, Q = Queenstown = 1, S = Southampton = 2
train_filtered['Embarked'] = np.where(train_filtered['Embarked']=='C', 0, np.where(train_filtered['Embarked']=='Q',1,2))
test_filtered['Embarked'] = np.where(test_filtered['Embarked']=='C', 0, np.where(test_filtered['Embarked']=='Q',1,2))

# Sex: male=0, female=1
train_filtered['Sex'] = np.where(train_filtered['Sex']=='male', 0, 1)
test_filtered['Sex'] = np.where(test_filtered['Sex']=='male', 0, 1)

# Ticket: Single=0, Family=1
train_filtered.Ticket = np.where(train_filtered.Ticket.duplicated(), 1,0)
test_filtered.Ticket = np.where(test_filtered.Ticket.duplicated(), 1,0)


# In[ ]:


# Check our assumption about the cabins holds true (fare of cabin holders is higher)
label_0 = 'Has no cabin, average fare %s' %train_filtered.Fare[train_filtered.Cabin == 0].mean()
label_1 = 'Has cabin, average fare %s' %train_filtered.Fare[train_filtered.Cabin == 1].mean()
train_filtered.Fare[train_filtered.Cabin == 0].plot(label=label_0)
train_filtered.Fare[train_filtered.Cabin == 1].plot(label=label_1)
plt.ylabel('Fare')
plt.xlabel('RowId')
plt.legend()
plt.show()


# In[ ]:


# Check variable distributions before performing any changes (changes should not significantly impact PDFs)
scatter_matrix(train_filtered, figsize=(10,10))
plt.show()


# In[ ]:


# Check variable correlations before performing any changes (changes should not significantly impact CORR)
cor = train_filtered.corr()
plt.figure(figsize=(12,9))
plt.pcolor(cor, cmap='Spectral')
plt.yticks(np.arange(0.5, len(cor.index), 1), cor.index)
plt.xticks(np.arange(0.5, len(cor.columns), 1), cor.columns)
plt.colorbar()
plt.show()


# In[ ]:


# What variables have the highest correlation with survival?
c = ['r' if x<0 else 'g' for x in cor.Survived.drop('Survived').values]
for i, bar in enumerate(cor.Survived.drop('Survived')):
    plt.bar(i, bar, color=c[i])
plt.xticks(range(len(c)), cor.Survived.drop('Survived').index, rotation='vertical')
plt.title('Variable correlation w/ Survival')
plt.ylabel('Correlation')
plt.show()


# NOTE: Lack of correlation doesn't mean the variable is not significant for the classification problem. For example, 'Age' doesn't show strong correlation to survival because increasing/decreasing age doesn't ncessarily imply increasing/decreasing chances of survival (no correlation), but some age groups (infants) clearly have better survival probability.

# In[ ]:


# Fill NaN by using the average value of the same Pclass
test_filtered.Fare = test_filtered.Fare.fillna(test_filtered.groupby('Pclass').Fare.transform('mean'))
train_filtered.Age = train_filtered.Age.fillna(train_filtered.groupby('Pclass').Age.transform('mean'))
test_filtered.Age = test_filtered.Age.fillna(test_filtered.groupby('Pclass').Age.transform('mean'))

# Fill NaN by using the mode value of the same Pclass
# test_filtered.Fare = test_filtered.groupby('Pclass').Fare.apply(lambda x: x.fillna(x.mode()[0]))
# train_filtered.Age = train_filtered.groupby('Pclass').Age.apply(lambda x: x.fillna(x.mode()[0]))
# test_filtered.Age = test_filtered.groupby('Pclass').Age.apply(lambda x: x.fillna(x.mode()[0]))


# In[ ]:


# Check Age distribution has not been modified when filling NaNs
plt.figure(figsize=(8,4))
plt.subplot(2,2,1)
train.Age.hist(grid=0, edgecolor='black', bins=20)
plt.title('Original Age Histogram')
plt.subplot(2,2,2)
train[train.Survived==1].Age.hist(grid=0, edgecolor='black', bins=20)
train[train.Survived==0].Age.hist(grid=0, edgecolor='black', bins=20, alpha=0.5)
plt.title('Original Age Histogram (by Survival)')
plt.subplot(2,2,3)
train_filtered.Age.hist(grid=0, edgecolor='black', bins=20)
plt.title('Modified Age Histogram')
plt.subplot(2,2,4)
train_filtered[train_filtered.Survived==1].Age.hist(grid=0, edgecolor='black', bins=20)
train_filtered[train_filtered.Survived==0].Age.hist(grid=0, edgecolor='black', bins=20, alpha=0.5)
plt.title('Modified Age Histogram (by Survival)')
plt.tight_layout()
plt.show()


# <font size=4 color=red> This filling methodology doesn't seem to be appropriate, as it changes the 'Age' probability distribution significantly, and specially skews the non-survival distro.</font>
# <font size=4 color=blue> Instead, we will sample the original 'Age' probability distribution and populate the missing values with those -random-samples. <font size=2> We should use something like Pclass to steer the sampling instead of just sampling randomly, but will skip that step for this quick exercise. </font>

# In[ ]:


# Restore the original 'Age' data.
train_filtered.Age = train.Age
test_filtered.Age = test.Age

# Model the original Age distributions
train_age_hist = np.histogram(train.Age.dropna(), bins=20)
test_age_hist = np.histogram(test.Age.dropna(), bins=20)
train_age_hist_dist = st.rv_histogram(train_age_hist)
test_age_hist_dist = st.rv_histogram(test_age_hist)

# Fill NaN by using random samples from the original data
train_filtered.Age = train_filtered.Age.apply(lambda x: train_age_hist_dist.rvs() if np.isnan(x) else x)
test_filtered.Age = test_filtered.Age.apply(lambda x: test_age_hist_dist.rvs() if np.isnan(x) else x)


# In[ ]:


# Check Age distribution has not been modified when filling NaNs
plt.figure(figsize=(8,4))
plt.subplot(2,2,1)
train.Age.hist(grid=0, edgecolor='black', bins=20)
plt.title('Original Age Histogram')
plt.subplot(2,2,2)
train[train.Survived==1].Age.hist(grid=0, edgecolor='black', bins=20)
train[train.Survived==0].Age.hist(grid=0, edgecolor='black', bins=20, alpha=0.5)
plt.title('Original Age Histogram (by Survival)')
plt.subplot(2,2,3)
train_filtered.Age.hist(grid=0, edgecolor='black', bins=20)
plt.title('Modified Age Histogram')
plt.subplot(2,2,4)
train_filtered[train_filtered.Survived==1].Age.hist(grid=0, edgecolor='black', bins=20)
train_filtered[train_filtered.Survived==0].Age.hist(grid=0, edgecolor='black', bins=20, alpha=0.5)
plt.title('Modified Age Histogram (by Survival)')
plt.tight_layout()
plt.show()


# <font size=3 color=gree> Still not correct (used random sampling), but much better in not biasing the origina data.</font>

# In[ ]:


# Group/classify 'Age' into discrete bins
train_filtered.Age = pd.cut(train_filtered.Age, 5, labels=False)
test_filtered.Age = pd.cut(test_filtered.Age, 5, labels=False)


# In[ ]:


# Check title types
pd.concat([train_filtered,test_filtered]).Name.str.extract(r',\s*([^\.]*)\s*\.', expand=False).value_counts()


# In[ ]:


# Extract title from Names and code into numbers. Lump the lower count values into 'Other'
train_filtered['Title'] = train.Name.str.extract(r',\s*([^\.]*)\s*\.', expand=False)
test_filtered['Title'] = test.Name.str.extract(r',\s*([^\.]*)\s*\.', expand=False)

# Mr=0, Miss=1, Mrs=2, Master=3, Other=4
titles_dict={'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3}
train_filtered.Title = train_filtered.Title.map(titles_dict) # if not in titles_dict, nan will be applied
test_filtered.Title = test_filtered.Title.map(titles_dict) # if not in titles_dict, nan will be applied
train_filtered.Title.fillna(4, inplace=True)
test_filtered.Title.fillna(4, inplace=True)


# In[ ]:


# SibSp and Parch seem to have little impact on their own.
# Let's try to create a combined "Family" variable and see
train_filtered['Family'] = 0
test_filtered['Family'] = 0
train_filtered.Family.loc[(train_filtered.SibSp!=0) | (train_filtered.Parch!=0)] = 1
test_filtered.Family.loc[(test_filtered.SibSp!=0) | (test_filtered.Parch!=0)] = 1


# In[ ]:


# Let's repeat the check now..."Family" variable has way more correlation to survival now!cor = train_filtered.corr()
c = ['r' if x<0 else 'g' for x in cor.Survived.drop('Survived').values]
for i, bar in enumerate(cor.Survived.drop('Survived')):
    plt.bar(i, bar, color=c[i])
plt.xticks(range(len(c)), cor.Survived.drop('Survived').index, rotation='vertical')
plt.title('Variable correlation w/ Survival')
plt.ylabel('Correlation')
plt.show()


# In[ ]:


basic_expl=pd.concat([train_filtered.dtypes, train_filtered.isnull().sum(), test_filtered.isnull().sum()], axis=1)
basic_expl.columns=['type', 'NaN_train', 'NaN_test']
basic_expl.T


# ## Select final sets

# In[ ]:


model_features = ['Age', 'Cabin', 'Embarked', 'Family', 'Fare', 'Pclass', 'Sex',  'Ticket', 'Title']


# In[ ]:


cor = train_filtered[model_features].join(train_filtered.Survived).corr()
c = ['r' if x<0 else 'g' for x in cor.Survived.drop('Survived').values]
for i, bar in enumerate(cor.Survived.drop('Survived')):
    plt.bar(i, bar, color=c[i])
plt.xticks(range(len(c)), cor.Survived.drop('Survived').index, rotation='vertical')
plt.title('Variable correlation w/ Survival')
plt.ylabel('Correlation')
plt.show()


# In[ ]:


X_train_filtered = train_filtered[model_features]
y_train_filtered = train_filtered['Survived']
test_filtered = test_filtered[model_features]


# ## Split train and test datasets

# In[ ]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train_filtered, y_train_filtered, 
                                                    test_size=0.33, random_state=0)


# ## Transform/Scale/Normalize

# In[ ]:


scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
test_filtered_scaled = scaler.transform(test_filtered)


# ## Models
# Let's just try a few popular classifiers...

# ### 1. Linear Regression
# Training fit 0.4347  
# Testing fit 0.4004  
# Kaggle score

# In[ ]:


# Call model instance
linear = linear_model.LinearRegression()
# Fit model on train data
linear.fit(X_train_scaled , y_train)
# Check fit quality over train set
print ('Training fit score: %s' %linear.score(X_train_scaled, y_train))
# Check fit quality over test set
print ('Testing fit score: %s' %linear.score(X_test_scaled, y_test))


# In[ ]:


# Save the best result
linear_result=linear.predict(test_filtered_scaled)


# In[ ]:


# Check relative features importance (using matching coefficient)
plt.bar(np.arange(len(linear.coef_)), linear.coef_)
plt.xticks(np.arange(len(linear.coef_)), model_features, rotation=45)
plt.xlabel('Coefficients'); plt.ylabel('Coefficient value'); plt.title('Feature importance')
plt.show()


# ### 2. Logistic Regression
# Training fit 0.8305  
# Testing fit 0.8034  
# Kaggle score 

# In[ ]:


# Call model instance
logistic = linear_model.LogisticRegression(random_state=0)
# Fit model on train data
logistic.fit(X_train_scaled , y_train)
# Check fit quality over train set
print ('Training fit score: %s' %logistic.score(X_train_scaled, y_train))
# Check fit quality over test set
print ('Testing fit score: %s' %logistic.score(X_test_scaled, y_test))


# In[ ]:


# # Run parameter optimizer
# param_grid ={'random_state':[0], 'penalty':['l1', 'l2'], 'tol':[1e-3,1e-4, 1e-5, 1e-6], 'C': st.randint(20, 500)
#             ,'fit_intercept':[True, False], 'class_weight': ['balanced', None, {0:0.1, 1:0.9}, {0:0.9, 1:0.1}]}
# opt = RandomizedSearchCV(linear_model.LogisticRegression(), param_grid, scoring='accuracy',n_iter=100)
# opt.fit(X_train_scaled , y_train)
# # Check fit quality over test set
# opt.score(X_test_scaled, y_test)


# In[ ]:


# Save the best result
logistic_result=logistic.predict(test_filtered_scaled)


# In[ ]:


# Check relative features importance (using matching coefficient)
plt.bar(np.arange(len(logistic.coef_[0])), logistic.coef_[0])
plt.xticks(np.arange(len(logistic.coef_[0])), model_features, rotation=45)
plt.xlabel('Coefficients'); plt.ylabel('Coefficient value'); plt.title('Feature importance')
plt.show()


# ### 3. K-Nearest Neighbors
# Training fit 0.8490  
# Testing fit 0.8101  
# Kaggle score 0.7606

# In[ ]:


# Call model instance
knn = neighbors.KNeighborsClassifier()
# Fit model on train data
knn.fit(X_train_scaled , y_train)
# Check fit quality over train set
print ('Training fit score: %s' %knn.score(X_train_scaled, y_train))
# Check fit quality over test set
print ('Testing fit score: %s' %knn.score(X_test_scaled, y_test))


# In[ ]:


# # Run parameter optimizer
# param_grid ={'n_neighbors':range(1,len(model_features)+1)}
# opt = RandomizedSearchCV(neighbors.KNeighborsClassifier(), param_grid, scoring='accuracy', n_iter=8)
# opt.fit(X_train_scaled , y_train)
# # Check fit quality over test set
# opt.score(X_test_scaled, y_test)


# In[ ]:


# Save the best result
knn_result=knn.predict(test_filtered_scaled)


# ### 4. Decision Tree
# Training fit 0.9547  
# Testing fit 0.7966  
# Kaggle score 0.71291

# In[ ]:


# Call model instance
dt = tree.DecisionTreeClassifier(random_state=0)
# Fit model on train data
dt.fit(X_train_scaled , y_train)
# Check fit quality over train set
print ('Training fit score: %s' %dt.score(X_train_scaled, y_train))
# Check fit quality over test set
print ('Testing fit score: %s' %dt.score(X_test_scaled, y_test))


# In[ ]:


# Export decision tree graph
from sklearn import tree
tree.export_graphviz(dt,out_file='tree.dot')


# In[ ]:


# # Run parameter optimizer
# param_grid ={'random_state':[0], 'criterion':['gini', 'entropy'], 'splitter':['best', 'random']
#             ,'class_weight': ['balanced', None, {0:0.1, 1:0.9}, {0:0.9, 1:0.1}]}
# opt = RandomizedSearchCV(tree.DecisionTreeClassifier(), param_grid, scoring='accuracy', n_iter=16)
# opt.fit(X_train_scaled , y_train)
# # Check fit quality over test set
# opt.score(X_test_scaled, y_test)


# In[ ]:


# Save the best result
dt_result=dt.predict(test_filtered_scaled)


# In[ ]:


# Check features importance (Gini importances)
plt.bar(np.arange(len(dt.feature_importances_)), dt.feature_importances_)
plt.xticks(np.arange(len(dt.feature_importances_)), model_features, rotation=45)
plt.xlabel('Coefficients'); plt.ylabel('Coefficient value'); plt.title('Feature importance')
plt.show()


# ### 5. Random Forest
# Training fit 0.9429  
# Testing fit 0.8441  
# Kaggle score 0.7273

# In[ ]:


# Call model instance
rf = ensemble.RandomForestClassifier(random_state=0)
# Fit model on train data
rf.fit(X_train_scaled , y_train)
# Check fit quality over train set
print ('Training fit score: %s' %rf.score(X_train_scaled, y_train))
# Check fit quality over test set
print ('Testing fit score: %s' %rf.score(X_test_scaled, y_test))


# In[ ]:


# # Run parameter optimizer
# param_grid ={'random_state':[0], 'n_estimators': st.randint(1, 500), 'criterion':['gini', 'entropy']
#             ,'bootstrap':[True, False], 'class_weight': ['balanced', None, {0:0.1, 1:0.9}, {0:0.9, 1:0.1}]}
# opt = RandomizedSearchCV(ensemble.RandomForestClassifier(), param_grid, scoring='accuracy', n_iter=100)
# opt.fit(X_train_scaled , y_train)
# # Check fit quality over test set
# opt.score(X_test_scaled, y_test)


# In[ ]:


# Check features importance (Gini importances)
plt.bar(np.arange(len(dt.feature_importances_)), dt.feature_importances_)
plt.xticks(np.arange(len(dt.feature_importances_)), model_features, rotation=45)
plt.xlabel('Coefficients'); plt.ylabel('Coefficient value'); plt.title('Feature importance')
plt.show()


# In[ ]:


# Save the best result
rf_result=rf.predict(test_filtered_scaled)


# ### 6. Neural Network
# Kaggle score 0.78468

# In[ ]:


# Call model instance
nn = neural_network.MLPClassifier(random_state=0, max_iter=1000)
# Fit model on train data
nn.fit(X_train_scaled , y_train)
# Check fit quality over training set
print ('Training fit score: %s' %nn.score(X_train_scaled, y_train))
# Check fit quality over test set
print ('Testing fit score: %s' %nn.score(X_test_scaled, y_test))


# In[ ]:


# # Run parameter optimizer
# param_grid ={'random_state':[0], 'hidden_layer_sizes': (st.randint(1, 500).rvs(),)
#              ,'activation':['identity', 'logistic', 'tanh', 'relu'], 'solver':['lbfgs', 'sgd', 'adam']}
# opt = RandomizedSearchCV(neural_network.MLPClassifier(max_iter=1000), param_grid, scoring='accuracy', n_iter=12)
# opt.fit(X_train_scaled , y_train)
# # Check fit quality over test set
# opt.score(X_test_scaled, y_test)


# In[ ]:


# Save the best result
nn_result=nn.predict(test_filtered_scaled)


# ### 7. Support Vector Machines
# Training fit 0.8406   
# Testing fit 0.8135  
# Kaggle score 0.7799

# In[ ]:


# Call model instance
svms = svm.SVC()
# Fit model on train data
svms.fit(X_train_scaled , y_train)
# Check fit quality over training set
print ('Training fit score: %s' %svms.score(X_train_scaled, y_train))
# Check fit quality over test set
print ('Testing fit score: %s' %svms.score(X_test_scaled, y_test))


# In[ ]:


# # Run parameter optimizer
# param_grid ={'C':[0.5, 1, 1.5], 'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'degree':[3,5,7]
#             ,'probability':[True, False], 'class_weight': ['balanced', None, {0:0.1, 1:0.9}, {0:0.9, 1:0.1}]}
# opt = RandomizedSearchCV(svm.SVC(), param_grid, scoring='accuracy', return_train_score=True, n_iter=50)
# opt.fit(X_train_scaled , y_train)
# # Check fit quality over test set
# print ('Training fit OPT score: %s' %opt.score(X_test_scaled, y_test))


# In[ ]:


# Save the best result
svms_result=svms.predict(test_filtered_scaled)


# ### 8. Bagging Ensemble
# Training fit 0.9463  
# Testing fit 0.8203  
# Kaggle score 0.79904

# In[ ]:


# Call model instance
be = ensemble.BaggingClassifier()
# Fit model on train data
be.fit(X_train_scaled , y_train)
# Check fit quality over train set
print ('Training fit score: %s' %be.score(X_train_scaled, y_train))
# Check fit quality over test set
print ('Testing fit score: %s' %be.score(X_test_scaled, y_test))


# In[ ]:


# # Run parameter optimizer
# param_grid ={'n_estimators': st.randint(1, 500)}
# opt = RandomizedSearchCV(ensemble.BaggingClassifier(), param_grid, scoring='accuracy', return_train_score=True)
# opt.fit(X_train_scaled , y_train)
# # Check fit quality over test set
# print ('Training fit OPT score: %s' %opt.score(X_test_scaled, y_test))


# In[ ]:


# Save the best result
be_result=be.predict(test_filtered_scaled)
# Save the model
pickle.dump(be, open('bagging_ensemble_model', 'wb'))

# some time later...
 
# load the model from disk
loaded_model = pickle.load(open('bagging_ensemble_model', 'rb'))
result = loaded_model.score(X_test_scaled, y_test)
print(result)


# ## Export results csv file

# In[ ]:


best_result = be_result
dfres = pd.DataFrame([test.PassengerId, best_result]).T
dfres.columns=['PassengerId', 'Survived']
dfres.set_index('PassengerId', inplace=True)
dfres.to_csv('results.csv')

