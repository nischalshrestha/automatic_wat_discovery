#!/usr/bin/env python
# coding: utf-8

# # TITANIC SOLUTION
# ### A BEGINNER'S GUIDE
# - Exploratory Data Analysis (EDA) with Visualization
# - Feature Extraction
# - Data Modelling
# - Model Evaluation

# ## Loading Modules

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set() # setting seaborn default for plots


# ## Loading Datasets
# 
# Loading train and test dataset

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ## Looking into the training dataset
# Printing first 5 rows of the train dataset.

# In[ ]:


train.head()


# Below is a brief information about each columns of the dataset:
# 
# 1. **PassengerId:** An unique index for passenger rows. It starts from 1 for first row and increments by 1 for every new rows.
# 
# 2. **Survived:** Shows if the passenger survived or not. 1 stands for survived and 0 stands for not survived.
# 
# 3. **Pclass:** Ticket class. 1 stands for First class ticket. 2 stands for Second class ticket. 3 stands for Third class ticket.
# 
# 4. **Name:** Passenger's name. Name also contain title. "Mr" for man. "Mrs" for woman. "Miss" for girl. "Master" for boy.
# 
# 5. **Sex:** Passenger's sex. It's either Male or Female.
# 
# 6. **Age:** Passenger's age. "NaN" values in this column indicates that the age of that particular passenger has not been recorded.
# 
# 7. **SibSp:** Number of siblings or spouses travelling with each passenger.
# 8. **Parch:** Number of parents of children travelling with each passenger.
# 9. **Ticket:** Ticket number.
# 10. **Fare:** How much money the passenger has paid for the travel journey.
# 11. **Cabin:** Cabin number of the passenger. "NaN" values in this column indicates that the cabin number of that particular passenger has not been recorded.
# 12. **Embarked:** Port from where the particular passenger was embarked/boarded.

# **Total rows and columns**
# 
# We can see that there are 891 rows and 12 columns in our training dataset.

# In[ ]:


train.shape


# **Describing training dataset**
# 
# *describe()* method can show different values like count, mean, standard deviation, etc. of numeric data types.

# In[ ]:


train.describe()


# *describe(include = ['O'])* will show the descriptive statistics of object data types.

# In[ ]:


train.describe(include=['O'])


# This shows that there are duplicate *Ticket number* and *Cabins* shared. The highest number of duplicate ticket number is "CA. 2343". It has been repeated 7 times. Similarly, the highest number of people using the same cabin is 4. They are using cabin number "C23 C25 C27".
# 
# We also see that 644 people were embarked from port "S".
# 
# Among 891 rows, 577 were Male and the rest were Female.

# We use *info()* method to see more information of our train dataset.

# In[ ]:


train.info()


# We can see that *Age* value is missing for many rows. 
# 
# Out of 891 rows, the *Age* value is present only in 714 rows.
# 
# Similarly, *Cabin* values are also missing in many rows. Only 204 out of 891 rows have *Cabin* values.

# In[ ]:


train.isnull().sum()


# There are 177 rows with missing *Age*, 687 rows with missing *Cabin* and 2 rows with missing *Embarked* information.

# ## Looking into the testing dataset
# 
# Test data has 418 rows and 11 columns.
# 
# > Train data rows = 891
# >
# > Test data rows = 418
# >
# > Total rows = 891+418 = 1309
# 
# We can see that around 2/3 of total data is set as Train data and around 1/3 of total data is set as Test data.

# In[ ]:


test.shape


# *Survived* column is not present in Test data.
# We have to train our classifier using the Train data and generate predictions (*Survived*) on Test data.

# In[ ]:


test.head()


# In[ ]:


test.info()


# There are missing entries for *Age* in Test dataset as well.
# 
# Out of 418 rows in Test dataset, only 332 rows have *Age* value.
# 
# *Cabin* values are also missing in many rows. Only 91 rows out ot 418 have values for *Cabin* column.

# In[ ]:


test.isnull().sum()


# There are 86 rows with missing *Age*, 327 rows with missing *Cabin* and 1 row with missing *Fare* information.

# ## Relationship between Features and Survival
# 
# In this section, we analyze relationship between different features with respect to *Survival*. We see how different feature values show different survival chance. We also plot different kinds of diagrams to **visualize** our data and findings.

# In[ ]:


survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]

print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))
print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))
print ("Total: %i"%len(train))


# ### Pclass vs. Survival
# 
# Higher class passengers have better survival chance.

# In[ ]:


train.Pclass.value_counts()


# In[ ]:


train.groupby('Pclass').Survived.value_counts()


# In[ ]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# In[ ]:


#train.groupby('Pclass').Survived.mean().plot(kind='bar')
sns.barplot(x='Pclass', y='Survived', data=train)


# ### Sex vs. Survival
# 
# Females have better survival chance.

# In[ ]:


train.Sex.value_counts()


# In[ ]:


train.groupby('Sex').Survived.value_counts()


# In[ ]:


train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()


# In[ ]:


#train.groupby('Sex').Survived.mean().plot(kind='bar')
sns.barplot(x='Sex', y='Survived', data=train)


# ### Pclass & Sex vs. Survival

# Below, we just find out how many males and females are there in each *Pclass*. We then plot a stacked bar diagram with that information. We found that there are more males among the 3rd Pclass passengers.

# In[ ]:


tab = pd.crosstab(train['Pclass'], train['Sex'])
print (tab)

tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Pclass')
plt.ylabel('Percentage')


# In[ ]:


sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train)


# From the above plot, it can be seen that:
# - Women from 1st and 2nd Pclass have almost 100% survival chance. 
# - Men from 2nd and 3rd Pclass have only around 10% survival chance.

# ### Pclass, Sex & Embarked vs. Survival

# In[ ]:


sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train)


# From the above plot, it can be seen that:
# - Almost all females from Pclass 1 and 2 survived.
# - Females dying were mostly from 3rd Pclass.
# - Males from Pclass 1 only have slightly higher survival chance than Pclass 2 and 3.

# ### Embarked vs. Survived

# In[ ]:


train.Embarked.value_counts()


# In[ ]:


train.groupby('Embarked').Survived.value_counts()


# In[ ]:


train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()


# In[ ]:


#train.groupby('Embarked').Survived.mean().plot(kind='bar')
sns.barplot(x='Embarked', y='Survived', data=train)


# ### Parch vs. Survival

# In[ ]:


train.Parch.value_counts()


# In[ ]:


train.groupby('Parch').Survived.value_counts()


# In[ ]:


train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()


# In[ ]:


#train.groupby('Parch').Survived.mean().plot(kind='bar')
sns.barplot(x='Parch', y='Survived', ci=None, data=train) # ci=None will hide the error bar


# ### SibSp vs. Survival

# In[ ]:


train.SibSp.value_counts()


# In[ ]:


train.groupby('SibSp').Survived.value_counts()


# In[ ]:


train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()


# In[ ]:


#train.groupby('SibSp').Survived.mean().plot(kind='bar')
sns.barplot(x='SibSp', y='Survived', ci=None, data=train) # ci=None will hide the error bar


# ### Age vs. Survival

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train, split=True, ax=ax1)
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, split=True, ax=ax2)
sns.violinplot(x="Sex", y="Age", hue="Survived", data=train, split=True, ax=ax3)


# From *Pclass* violinplot, we can see that:
# - 1st Pclass has very few children as compared to other two classes.
# - 1st Plcass has more old people as compared to other two classes.
# - Almost all children (between age 0 to 10) of 2nd Pclass survived.
# - Most children of 3rd Pclass survived.
# - Younger people of 1st Pclass survived as compared to its older people.
# 
# From *Sex* violinplot, we can see that:
# - Most male children (between age 0 to 14) survived.
# - Females with age between 18 to 40 have better survival chance.

# In[ ]:


total_survived = train[train['Survived']==1]
total_not_survived = train[train['Survived']==0]
male_survived = train[(train['Survived']==1) & (train['Sex']=="male")]
female_survived = train[(train['Survived']==1) & (train['Sex']=="female")]
male_not_survived = train[(train['Survived']==0) & (train['Sex']=="male")]
female_not_survived = train[(train['Survived']==0) & (train['Sex']=="female")]

plt.figure(figsize=[15,5])
plt.subplot(111)
sns.distplot(total_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(total_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Age')

plt.figure(figsize=[15,5])

plt.subplot(121)
sns.distplot(female_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(female_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Female Age')

plt.subplot(122)
sns.distplot(male_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(male_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Male Age')


# From the above figures, we can see that:
# - Combining both male and female, we can see that children with age between 0 to 5 have better chance of survival.
# - Females with age between "18 to 40" and "50 and above" have higher chance of survival.
# - Males with age between 0 to 14 have better chance of survival.

# ### Correlating Features

# Heatmap of Correlation between different features:
# 
# >Positive numbers = Positive correlation, i.e. increase in one feature will increase the other feature & vice-versa.
# >
# >Negative numbers = Negative correlation, i.e. increase in one feature will decrease the other feature & vice-versa.
# 
# In our case, we focus on which features have strong positive or negative correlation with the *Survived* feature.

# In[ ]:


plt.figure(figsize=(15,6))
sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)


# ## Feature Extraction
# 
# In this section, we select the appropriate features to train our classifier. Here, we create new features based on existing features. We also convert categorical features into numeric form.

# ### Name Feature
# 
# Let's first extract titles from *Name* column.

# In[ ]:


train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')


# In[ ]:


train.head()


# As you can see above, we have added a new column named *Title* in the Train dataset with the *Title* present in the particular passenger name.

# In[ ]:


pd.crosstab(train['Title'], train['Sex'])


# The number of passengers with each *Title* is shown above.
# 
# We now replace some less common titles with the name "Other".

# In[ ]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',  	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# After that, we convert the categorical *Title* values into numeric form.

# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


# In[ ]:


train.head()


# ### Sex Feature
# 
# We convert the categorical value of *Sex* into numeric. We represent 0 as female and 1 as male.

# In[ ]:


for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


train.head()


# ### Embarked Feature
# 
# There are empty values for some rows for *Embarked* column. The empty values are represented as "nan" in below list.

# In[ ]:


train.Embarked.unique()


# Let's check the number of passengers for each *Embarked* category.

# In[ ]:


train.Embarked.value_counts()


# We find that category "S" has maximum passengers. Hence, we replace "nan" values with "S".

# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


train.head()


# We now convert the categorical value of *Embarked* into numeric. We represent 0 as S, 1 as C and 2 as Q.

# In[ ]:


for dataset in train_test_data:
    #print(dataset.Embarked.unique())
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[ ]:


train.head()


# ### Age Feature
# 
# We first fill the NULL values of *Age* with a random number between (mean_age - std_age) and (mean_age + std_age). 
# 
# We then create a new column named *AgeBand*. This categorizes age into 5 different age range.

# In[ ]:


for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['AgeBand'] = pd.cut(train['Age'], 5)

print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())


# In[ ]:


train.head()


# Now, we map *Age* according to *AgeBand*.

# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4


# In[ ]:


train.head()


# ### Fare Feature
# 
# Replace missing *Fare* values with the median of *Fare*.

# In[ ]:


for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())


# Create *FareBand*. We divide the *Fare* into 4 category range.

# In[ ]:


train['FareBand'] = pd.qcut(train['Fare'], 4)
print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())


# In[ ]:


train.head()


# Map *Fare* according to *FareBand*

# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[ ]:


train.head()


# ### SibSp & Parch Feature
# 
# Combining *SibSp* & *Parch* feature, we create a new feature named *FamilySize*.

# In[ ]:


for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())


# About data shows that: 
# 
# - Having *FamilySize* upto 4 (from 2 to 4) has better survival chance. 
# - *FamilySize = 1*, i.e. travelling alone has less survival chance.
# - Large *FamilySize* (size of 5 and above) also have less survival chance.

# Let's create a new feature named *IsAlone*. This feature is used to check how is the survival chance while travelling alone as compared to travelling with family.

# In[ ]:


for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


# This shows that travelling alone has only 30% survival chance.

# In[ ]:


train.head(1)


# In[ ]:


test.head(1)


# ## Feature Selection
# 
# We drop unnecessary columns/features and keep only the useful ones for our experiment. Column *PassengerId* is only dropped from Train set because we need *PassengerId* in Test set while creating Submission file to Kaggle.

# In[ ]:


features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# We are done with Feature Selection/Engineering. Now, we are ready to train a classifier with our feature set.

# ## Classification & Accuracy 

# Define training and testing set

# In[ ]:


X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test.drop("PassengerId", axis=1).copy()

X_train.shape, y_train.shape, X_test.shape


# There are many classifying algorithms present. Among them, we choose the following *Classification* algorithms for our problem:
# 
# - Logistic Regression
# - Support Vector Machines (SVC)
# - Linear SVC
# - k-Nearest Neighbor (KNN)
# - Decision Tree
# - Random Forest
# - Naive Bayes (GaussianNB)
# - Perceptron
# - Stochastic Gradient Descent (SGD)
# 
# Here's the training and testing procedure:
# 
# > First, we train these classifiers with our training data. 
# >
# > After that, using the trained classifier, we predict the *Survival* outcome of test data.
# >
# > Finally, we calculate the accuracy score (in percentange) of the trained classifier.
# 
# ***Please note:*** that the accuracy score is generated based on our training dataset.

# In[ ]:


# Importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier


# ### Logistic Regression
# 
# [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression), or logit regression, or logit model is a regression model where the dependent variable (DV) is categorical. This article covers the case of a binary dependent variable—that is, where it can take only two values, "0" and "1", which represent outcomes such as pass/fail, win/lose, alive/dead or healthy/sick. Cases where the dependent variable has more than two outcome categories may be analysed in multinomial logistic regression, or, if the multiple categories are ordered, in ordinal logistic regression.

# In[ ]:


clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)
print (str(acc_log_reg) + ' percent')


# ### Support Vector Machine (SVM)
# 
# [Support Vector Machine (SVM)](https://en.wikipedia.org/wiki/Support_vector_machine) model is a Supervised Learning model used for classification and regression analysis. It is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.
# 
# In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces. Suppose some given data points each belong to one of two classes, and the goal is to decide which class a new data point will be in. In the case of support vector machines, a data point is viewed as a $p$-dimensional vector (a list of $p$ numbers), and we want to know whether we can separate such points with a $(p-1)$-dimensional hyperplane.
# 
# When data are not labeled, supervised learning is not possible, and an unsupervised learning approach is required, which attempts to find natural clustering of the data to groups, and then map new data to these formed groups. The clustering algorithm which provides an improvement to the support vector machines is called **support vector clustering** and is often used in industrial applications either when data are not labeled or when only some data are labeled as a preprocessing for a classification pass.
# 
# In the below code, [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) stands for Support Vector Classification.

# In[ ]:


clf = SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, y_train) * 100, 2)
print (acc_svc)


# ### Linear SVM
# 
# Linear SVM is a SVM model with linear kernel.
# 
# In the below code, [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) stands for Linear Support Vector Classification.

# In[ ]:


clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)
print (acc_linear_svc)


# ### $k$-Nearest Neighbors
# 
# [$k$-nearest neighbors algorithm (k-NN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) is one of the simplest machine learning algorithms and is used for classification and regression. In both cases, the input consists of the $k$ closest training examples in the feature space. The output depends on whether $k$-NN is used for classification or regression:
# 
# - In *$k$-NN classification*, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its $k$ nearest neighbors ($k$ is a positive integer, typically small). If $k = 1$, then the object is simply assigned to the class of that single nearest neighbor.
# 
# 
# - In *$k$-NN regression*, the output is the property value for the object. This value is the average of the values of its $k$ nearest neighbors.

# In[ ]:


clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train) * 100, 2)
print (acc_knn)


# ### Decision Tree
# 
# A [decision tree](https://en.wikipedia.org/wiki/Decision_tree) is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes). The paths from root to leaf represent classification rules.

# In[ ]:


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
print (acc_decision_tree)


# ### Random Forest
# 
# [Random forests](https://en.wikipedia.org/wiki/Random_forest) or **random decision forests** are an **ensemble learning method** for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for *decision trees' habit of overfitting to their training set*.
# 
# [Ensemble methods](https://en.wikipedia.org/wiki/Ensemble_learning) use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.

# In[ ]:


clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print (acc_random_forest)


# ### Gaussian Naive Bayes
# 
# [Naive Bayes classifiers](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
# 
# [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) (alternatively **Bayes' law** or **Bayes' rule**) describes the probability of an event, based on prior knowledge of conditions that might be related to the event. For example, if cancer is related to age, then, using Bayes' theorem, a person's age can be used to more accurately assess the probability that they have cancer, compared to the assessment of the probability of cancer made without knowledge of the person's age.
# 
# Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. It is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable. For example, a fruit may be considered to be an apple if it is red, round, and about 10 cm in diameter. A naive Bayes classifier considers each of these features to contribute independently to the probability that this fruit is an apple, regardless of any possible correlations between the color, roundness, and diameter features.

# In[ ]:


clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred_gnb = clf.predict(X_test)
acc_gnb = round(clf.score(X_train, y_train) * 100, 2)
print (acc_gnb)


# ### Perceptron
# 
# [Perceptron](https://en.wikipedia.org/wiki/Perceptron) is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.

# In[ ]:


clf = Perceptron(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_perceptron = clf.predict(X_test)
acc_perceptron = round(clf.score(X_train, y_train) * 100, 2)
print (acc_perceptron)


# ### Stochastic Gradient Descent (SGD)
# 
# [Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (often shortened in **SGD**), also known as incremental gradient descent, is a stochastic approximation of the gradient descent optimization method for minimizing an objective function that is written as a sum of differentiable functions. In other words, SGD tries to find minima or maxima by iteration.

# In[ ]:


clf = SGDClassifier(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_sgd = clf.predict(X_test)
acc_sgd = round(clf.score(X_train, y_train) * 100, 2)
print (acc_sgd)


# ## Confusion Matrix
# 
# A [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix), also known as an error matrix, is a specific table layout that allows visualization of the performance of an algorithm. Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa). The name stems from the fact that it makes it easy to see if the system is confusing two classes (i.e. commonly mislabelling one as another).
# 
# In predictive analytics, a table of confusion (sometimes also called a confusion matrix), is a table with two rows and two columns that reports the number of false positives, false negatives, true positives, and true negatives. This allows more detailed analysis than mere proportion of correct classifications (accuracy). Accuracy is not a reliable metric for the real performance of a classifier, because it will yield misleading results if the data set is unbalanced (that is, when the numbers of observations in different classes vary greatly). For example, if there were 95 cats and only 5 dogs in the data set, a particular classifier might classify all the observations as cats. The overall accuracy would be 95%, but in more detail the classifier would have a 100% recognition rate for the cat class but a 0% recognition rate for the dog class.
# 
# Here's another guide explaining [Confusion Matrix with example](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/).
# 
# $\begin{matrix} & Predicted Positive & Predicted Negative \\ Actual Positive & TP & FN \\ Actual Negative & FP & TN \end{matrix}$
# 
# In our (Titanic problem) case: 
# 
# >**True Positive:** The classifier predicted *Survived* **and** the passenger actually *Survived*.
# >
# >**True Negative:** The classifier predicted *Not Survived* **and** the passenger actually *Not Survived*.
# >
# >**False Postiive:** The classifier predicted *Survived* **but** the passenger actually *Not Survived*.
# >
# >**False Negative:** The classifier predicted *Not Survived* **but** the passenger actually *Survived*.

# In the example code below, we plot a confusion matrix for the prediction of ***Random Forest Classifier*** on our training dataset. This shows how many entries are correctly and incorrectly predicted by our classifer.

# In[ ]:


from sklearn.metrics import confusion_matrix
import itertools

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest_training_set = clf.predict(X_train)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy: %i %% \n"%acc_random_forest)

class_names = ['Survived', 'Not Survived']

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)
np.set_printoptions(precision=2)

print ('Confusion Matrix in Numbers')
print (cnf_matrix)
print ('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage')
print (cnf_matrix_percent)
print ('')

true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix = pd.DataFrame(cnf_matrix, 
                             index = true_class_names,
                             columns = predicted_class_names)

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(121)
sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True)


# ## Comparing Models
# 
# Let's compare the accuracy score of all the classifier models used above.

# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 
              'Perceptron', 'Stochastic Gradient Decent'],
    
    'Score': [acc_log_reg, acc_svc, acc_linear_svc, 
              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb, 
              acc_perceptron, acc_sgd]
    })

models.sort_values(by='Score', ascending=False)


# From the above table, we can see that *Decision Tree* and *Random Forest* classfiers have the highest accuracy score.
# 
# Among these two, we choose *Random Forest* classifier as it has the ability to limit overfitting as compared to *Decision Tree* classifier.

# ## Create Submission File to Kaggle

# In[ ]:


test.head()


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred_random_forest
    })

# submission.to_csv('submission.csv', index=False)


# ## References
# 
# This notebook is created by learning from the following notebooks:
# 
# - [A Journey through Titanic](https://www.kaggle.com/omarelgabry/a-journey-through-titanic)
# - [Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
# - [Pytanic](https://www.kaggle.com/headsortails/pytanic)
# - [Titanic best working Classifier](https://www.kaggle.com/sinakhorami/titanic-best-working-classifier)
# - [My approach to Titanic competition](https://www.kaggle.com/rafalplis/my-approach-to-titanic-competition)
