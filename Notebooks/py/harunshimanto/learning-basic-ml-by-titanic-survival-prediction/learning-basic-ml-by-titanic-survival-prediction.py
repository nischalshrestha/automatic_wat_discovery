#!/usr/bin/env python
# coding: utf-8

# # &#127909; Intro Of This Notebook:
# I will go through in this notebook the whole process of creating a machine learning model on the famous Titanic dataset, which is used by many people all over the world.

# In[ ]:


get_ipython().run_cell_magic(u'html', u'', u"<style>\n@import url('https://fonts.googleapis.com/css?family=Ewert|Roboto&effect=3d|ice|');\nbody {background-color: gainsboro;} \na {color: #37c9e1; font-family: 'Roboto';} \nh1 {color: #37c9e1; font-family: 'Orbitron'; text-shadow: 4px 4px 4px #aaa;} \nh2, h3 {color: slategray; font-family: 'Orbitron'; text-shadow: 4px 4px 4px #aaa;}\nh4 {color: #818286; font-family: 'Roboto';}\nspan {font-family:'Roboto'; color:black; text-shadow: 5px 5px 5px #aaa;}  \ndiv.output_area pre{font-family:'Roboto'; font-size:110%; color:lightblue;}      \n</style>")


# # &#128220; About RMS Titanic:
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# The RMS(Royal Mail Ship) Titanic was a British passenger liner that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after it collided with an iceberg during its maiden voyage from Southampton to New York City. There were an estimated 2,224 passengers and crew aboard the ship, and more than 1,500 died, making it one of the deadliest commercial peacetime maritime disasters in modern history. The RMS Titanic was the largest ship afloat at the time it entered service and was the second of three Olympic-class ocean liners operated by the White Star Line. The Titanic was built by the Harland and Wolff shipyard in Belfast. Thomas Andrews, her architect, died in the disaster.
# ![Imgur](https://i.imgur.com/l8lxGPm.jpg)
# 

# # &#128250; Short Video on RMS Titanic

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo('9xoqXVjBEF8')


# # &#128209; About the Titanic Problem:
# Using the machine learning tools, we need to analyze the information about the passensgers of RMS Titanic and predict which passenger has survived. This problem has been published by Kaggle and is widely used for learning basic concepts of Machine Learning

# # &#128220; Overview Of Titanic Datasets:
# 
# ### Data Dictionary
# 
# * Age: Age
# 
# * Cabin: Cabin
# 
# * Embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# 
# * Fare: Passenger Fare
# 
# * Name: Name
# 
# * Parch: Number of Parents/Children Aboard
# 
# * Pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# 
# * Sex: Sex
# 
# * Sibsp: Number of Siblings/Spouses Aboard
# 
# * Survived: Survival (0 = No; 1 = Yes)
# 
# * Ticket: Ticket Number
# 
# ### Variable Notes
# * pclass: A proxy for socio-economic status (SES) 1st = Upper 2nd = Middle 3rd = Lower
# 
# * age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# * sibsp: The dataset defines family relations in this way... Sibling = brother, sister, stepbrother, stepsister Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# * parch: The dataset defines family relations in this way... Parent = mother, father Child = daughter, son, stepdaughter, stepson Some children travelled only with a nanny, therefore parch=0 for them.

# # &#128229; Download Titanic Datasets:
# You can Download this Datasets from Our [Machine Learning Home Kaggle.](https://www.kaggle.com) 
# 
# Here is Dataset Download link: &#128071;
# * Train => "https://www.kaggle.com/c/titanic/download/train.csv"
# * Test => "https://www.kaggle.com/c/titanic/download/test.csv"

# # &#128233; Importing the Libraries

# In[ ]:


# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# # &#128187; Load and Read DataSets
# 

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Load and Read Datasets")


# In[ ]:


test['Survived'] = 0
test.head()


# In[ ]:


complete_data = train.append(test, ignore_index=True,sort=False)
complete_data.head()


# In[ ]:


print("No. of Training Data samples: " + str(train.shape[0]))
print("No. of Test Data samples: " + str(test.shape[0]))
print("Complete Data samples: " + str(complete_data.shape[0]))


# #  &#128223; Data Pre-processing 

# ## What is Data Pre-pocessing?
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# Data preprocessing is a data mining technique that involves transforming raw data into an understandable format. Real-world data is often incomplete, inconsistent, and/or lacking in certain behaviors or trends, and is likely to contain many errors. Data preprocessing is a proven method of resolving such issues. Data preprocessing prepares raw data for further processing.
# ![Imgur](https://i.imgur.com/VuYZfho.jpg)

# ## &#128223; 1. Handle Missing Data

# Check for missing values in the columns

# In[ ]:


train.isnull().sum()


# ###  &#128210; Note:
# Around 80% of Cabin's data is missing. So it will not be of much use to train the model.
# 
# Let us replace the missing values for age with median. Though not a best approach to replace missing data, we shall use this method for sake of simplicity.

# In[ ]:


train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())


# Replace missing data for Embarked. Let us use the port where maximum passengers have boarded

# In[ ]:


train.Embarked.value_counts()


# In[ ]:


train ['Embarked'] = train['Embarked'].fillna('S')
train.Embarked.unique()


# In[ ]:


test ['Embarked'] = test['Embarked'].fillna('S')
test.Embarked.unique()


# ## &#128290; 2. Encode categorical feature columns
# Encode the values of the categorical columns -- Sex, Embarked

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


def encode_features(data_set, feature_names):
    for feature_name in feature_names:
        le = LabelEncoder()
        le.fit(data_set[feature_name])
        encoded_column = le.transform(data_set[feature_name])
        data_set[feature_name] = encoded_column
    return data_set


# In[ ]:


features_to_encode = ['Sex', 'Embarked']
train_data = encode_features(train, features_to_encode)
train_data.head(10)


# In[ ]:


features_to_encode = ['Sex', 'Embarked']
test_data = encode_features(test, features_to_encode)
test_data.head(10)


# ## &#128257; 3. Converting Features
# 

# First I thought, we have to delete the ‘Cabin’ variable but then I found something interesting. A cabin number looks like ‘C123’ and the letter refers to the deck. Therefore we’re going to extract these and create a new feature, that contains a persons deck. Afterwords we will convert the feature into a numeric variable. The missing values will be converted to zero. In the picture below you can see the actual decks of the titanic, ranging from A to G.

# In[ ]:


import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train,test]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
# we can now drop the cabin feature
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)


# Converting “Fare” from float to int64, using the “astype()” function pandas provides:

# In[ ]:


data = [train,test]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)


# We will use the Name feature to extract the Titles from the Name, so that we can build a new feature out of that.

# In[ ]:


data = [train,test]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'],axis=1)


# In[ ]:


train['Ticket'].describe()


# Since the Ticket attribute has 929 unique tickets, it will be a bit tricky to convert them into useful categories. So we will drop it from the dataset.

# In[ ]:


train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'],axis=1)


# # &#128203; Exploratory Data Analysis 

# ## &#128505; What is Exploratory data analysis?
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# In statistics, exploratory data analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task.
# 
# You can say that EDA is statisticians way of story telling where you explore data, find patterns and tells insights. Often you have some questions in hand you try to validate those questions by performing EDA. <b>I have one article on [EDA](https://hackernoon.com/overview-of-exploratory-data-analysis-with-python-6213e105b00b)

# In[ ]:


train.info()


# In[ ]:


test.info()


# The training-set has 891 examples and 11 features + the target variable (survived). 2 of the features are floats, 5 are integers and 5 are objects. Below I have listed the features with a short description:

# In[ ]:


train.describe()


# Above we can see that 38% out of the training-set survived the Titanic. We can also see that the passenger ages range from 0.4 to 80. On top of that we can already detect some features, that contain missing values, like the ‘Age’ feature.

# In[ ]:


test.describe()


# In[ ]:


train.head(8)


# In[ ]:


test.head(8)


# From the table above, we can note a few things. First of all, that we need to convert a lot of features into numeric ones later on, so that the machine learning algorithms can process them. Furthermore, we can see that the features have widely different ranges, that we will need to convert into roughly the same scale. We can also spot some more features, that contain missing values (NaN = not a number), that wee need to deal with.Let’s take a more detailed look at what data is actually missing:

# In[ ]:


total = train.isnull().sum().sort_values(ascending=False)
percent1 = train.isnull().sum()/train.isnull().count()*100
percent2 = (round(percent1, 1)).sort_values(ascending=False)
missingdata = pd.concat([total, percent2], axis=1, keys=['Total', '%'])
missingdata.head(5)


# In[ ]:


total = test.isnull().sum().sort_values(ascending=False)
percent1 = test.isnull().sum()/test.isnull().count()*100
percent2 = (round(percent1, 1)).sort_values(ascending=False)
missingdata = pd.concat([total, percent2], axis=1, keys=['Total', '%'])
missingdata.head(5)


# 

# The ‘Cabin’ feature needs further investigation, but it looks like that we might want to drop it from the dataset, since 77 % of it are missing.

# In[ ]:


train.columns.values


# Above you can see the 11 features + the target variable (survived). What features could contribute to a high survival rate ?
# To me it would make sense if everything except ‘PassengerId’, ‘Ticket’ and ‘Name’ would be correlated with a high survival rate.

# # 📈 📉 📊 Data Visualization:

# ## 1. Age and Sex:

# In[ ]:


survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train[train['Sex']==0]
men = train[train['Sex']==1]
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
plt.xkcd()
_ = ax.set_title('Male')


# ### &#128210; Note:
# You can see that men have a high probability of survival when they are between 18 and 30 years old, which is also a little bit true for women but not fully. For women the survival chances are higher between 14 and 40.
# 
# For men the probability of survival is very low between the age of 5 and 18, but that isn't true for women. Another thing to note is that infants also have a little bit higher probability of survival.
# 
# Since there seem to be certain ages, which have increased odds of survival and because I want every feature to be roughly on the same scale, I will create age groups later on.

# ## 2.  Embarked, Pclass and Sex:
# 
# 

# In[ ]:


FacetGrid = sns.FacetGrid(train, row='Embarked', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()
plt.xkcd()


# Embarked seems to be correlated with survival, depending on the gender.
# 
# Women on port Q and on port S have a higher chance of survival. The inverse is true, if they are at port C. Men have a high survival probability if they are on port C, but a low probability if they are on port Q or S.
# 
# Pclass also seems to be correlated with survival. We will generate another plot of it below.

# ## 3. Pclass:

# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=train)
plt.xkcd()


# Here we see clearly, that Pclass is contributing to a persons chance of survival, especially if this person is in class 1. We will create another pclass plot below.

# In[ ]:


grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
plt.xkcd()


# ## 4. SibSp and Parch:
# SibSp and Parch would make more sense as a combined feature, that shows the total number of relatives, a person has on the Titanic. I will create it below and also a feature that sows if someone is not alone.

# In[ ]:


data = [train,test]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
train['not_alone'].value_counts()


# In[ ]:


axes = sns.factorplot('relatives','Survived', 
                      data=train, aspect = 2.5, )
plt.xkcd()


# # &#128295; Building Machine Learning Models
# 

# # 4. Feature Selection
#  drop unnecessary columns/features and keep only the useful ones for our experiment. Column PassengerId is only dropped from Train set because we need PassengerId in Test set while creating Submission file to Kaggle.

# In[ ]:


features_drop = ['SibSp', 'Parch','relatives','Deck']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)
test = test.drop(['Survived'],axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# ### &#128204; Note:
# Now we will train several Machine Learning models and compare their results. Note that because the dataset does not provide labels for their testing-set, we need to use the predictions on the training set to compare the algorithms with each other.

# In[ ]:


X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()


# # 1. Logistic Regression
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# Logistic regression, or logit regression, or logit model is a regression model where the dependent variable (DV) is categorical. This article covers the case of a binary dependent variable—that is, where it can take only two values, "0" and "1", which represent outcomes such as pass/fail, win/lose, alive/dead or healthy/sick. Cases where the dependent variable has more than two outcome categories may be analysed in multinomial logistic regression, or, if the multiple categories are ordered, in ordinal logistic regression.

# In[ ]:


clf = LogisticRegression()
clf.fit(X_train, Y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round( clf.score(X_train, Y_train) * 100, 2)
print (str(acc_log_reg) + ' %')


# # 2.Support Vector Machine (SVM)
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# Support Vector Machine (SVM) model is a Supervised Learning model used for classification and regression analysis. It is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.
# 
# In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces. Suppose some given data points each belong to one of two classes, and the goal is to decide which class a new data point will be in. In the case of support vector machines, a data point is viewed as a  p -dimensional vector (a list of  p  numbers), and we want to know whether we can separate such points with a  (p−1) -dimensional hyperplane.
# 
# When data are not labeled, supervised learning is not possible, and an unsupervised learning approach is required, which attempts to find natural clustering of the data to groups, and then map new data to these formed groups. The clustering algorithm which provides an improvement to the support vector machines is called support vector clustering and is often used in industrial applications either when data are not labeled or when only some data are labeled as a preprocessing for a classification pass.
# 
# In the below code, SVC stands for Support Vector Classification.

# In[ ]:


clf = SVC()
clf.fit(X_train, Y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, Y_train) * 100, 2)
print (str(acc_svc) + '%')


# # 3. Linear SVM
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# Linear SVM is a SVM model with linear kernel.
# 
# In the below code, LinearSVC stands for Linear Support Vector Classification.

# In[ ]:


clf = LinearSVC()
clf.fit(X_train, Y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_linear_svc = round(clf.score(X_train, Y_train) * 100, 2)
print (str(acc_linear_svc) + '%')


# # 4. Stochastic Gradient Descent (SGD)
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# 
# Stochastic gradient descent (often shortened to SGD), also known as incremental gradient descent, is an iterative method for optimizing a differentiable objective function, a stochastic approximation of gradient descent optimization. A recent article implicitly credits Herbert Robbins and Sutton Monro for developing SGD in their 1951 article titled "A Stochastic Approximation Method"; see Stochastic approximation for more information. It is called stochastic because samples are selected randomly (or shuffled) instead of as a single group (as in standard gradient descent) or in the order they appear in the training set.
# Both statistical estimation and machine learning consider the problem of minimizing an objective function that has the form of a sum:
# ![Imgur](https://i.imgur.com/j8zrTZk.png)
# where the parameter  ww which minimizes   Q(w) is to be estimated. Each summand function Qi is typically associated with the i-th observation in the data set (used for training).

# In[ ]:


sgd = linear_model.SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(str(acc_sgd)+'%')


# # 5. Random Forest
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# ![Imgur](https://i.imgur.com/lREy3CV.jpg)
# ![Imgur](https://i.imgur.com/lEuwiKK.jpg)

# In[ ]:


clf = RandomForestClassifier()
clf.fit(X_train, Y_train)

Y_prediction_randomforest = clf.predict(X_test)

clf.score(X_train, Y_train)
acc_random_forest = round(clf.score(X_train, Y_train) * 100, 2)
print(str(acc_random_forest) + '%')


# # 6.*k*-Nearest Neighbors
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# *k* -nearest neighbors algorithm (k-NN) is one of the simplest machine learning algorithms and is used for classification and regression. In both cases, the input consists of the  k  closest training examples in the feature space. The output depends on whether  k -NN is used for classification or regression:
# 
# * In  k -NN classification, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its  k  nearest neighbors ( k  is a positive integer, typically small). If  k=1 , then the object is simply assigned to the class of that single nearest neighbor.
# 
# * In  k -NN regression, the output is the property value for the object. This value is the average of the values of its  k nearest neighbors.

# In[ ]:


clf = KNeighborsClassifier()
clf.fit(X_train, Y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, Y_train) * 100, 2)
print (str(acc_knn)+'%')


# # 7.Decision Tree
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes). The paths from root to leaf represent classification rules.

# In[ ]:


clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, Y_train) * 100, 2)
print (str(acc_decision_tree) + '%')


# # 8. Gaussian Naive Bayes
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
# 
# Bayes' theorem (alternatively Bayes' law or Bayes' rule) describes the probability of an event, based on prior knowledge of conditions that might be related to the event. For example, if cancer is related to age, then, using Bayes' theorem, a person's age can be used to more accurately assess the probability that they have cancer, compared to the assessment of the probability of cancer made without knowledge of the person's age.
# 
# Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. It is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable. For example, a fruit may be considered to be an apple if it is red, round, and about 10 cm in diameter. A naive Bayes classifier considers each of these features to contribute independently to the probability that this fruit is an apple, regardless of any possible correlations between the color, roundness, and diameter features.

# In[ ]:


clf = GaussianNB()
clf.fit(X_train, Y_train)
y_pred_gnb = clf.predict(X_test)
acc_gnb = round(clf.score(X_train, Y_train) * 100, 2)
print (str(acc_gnb) + '%')


# # 9.Confusion Matrix 
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# 
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
# 

# In[ ]:


clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
y_pred_decision_tree_training_set = clf.predict(X_train)
acc_decision_tree = round(clf.score(X_train, Y_train) * 100, 2)
print ("Accuracy: %i %% \n"%acc_decision_tree)

class_names = ['Survived', 'Not Survived']

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_train, y_pred_decision_tree_training_set)
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
plt.xkcd()


# ## Comparing Models
# 
# Let's compare the accuracy score of all the classifier models used above.

# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 'Stochastic Gradient Decent'],
    
    'Score': [acc_log_reg, acc_svc, acc_linear_svc, 
              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb, acc_sgd]
    })

models.sort_values(by='Score', ascending=False)


# From the above table, we can see that *Decision Tree* and *Support Vector Machines* classfiers have the highest accuracy score.
# 
# Among these two, we choose *Support Vector Machines* classifier as it has the ability to limit overfitting as compared to *Decision Tree* classifier.

# # Create Submission File to Kaggle
# 

# In[ ]:


test.head()


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_prediction_randomforest 
    })
submission.to_csv('submission.csv', index=False)


# # References
# This notebook is created by learning from the following notebooks:
# 
# 1.[Predicting the Survival of Titanic Passengers](https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8)
# 
# 2.[A Journey through Titanic](https://www.kaggle.com/omarelgabry/a-journey-through-titanic)
# 
# 3.[Introduction to machine learning in Python with scikit-learn](https://www.dataschool.io/machine-learning-with-scikit-learn/)
# 

# # &#128225; Motivation 
# ## "The level of technical ability you need to show is not lowered, it’s even higher when you don’t have the educational background, but it’s totally possible."— **Dario Amodei, PhD, Researcher at OpenAI**
