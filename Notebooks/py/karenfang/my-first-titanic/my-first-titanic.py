#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# 
# **Problem: **
# 
# > In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# **Some Facts: **
# * On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. The survial rate is 32.46%
# * One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew.
# * Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

# The following steps recomendations are from https://www.kaggle.com/startupsci/titanic-data-science-solutions
# 
# **Data Science Problem Solving Sample Steps: **
# 1. Question or problem definition.
# 2. Acquire training and testing data.
# 3. Wrangle, prepare, cleanse the data.
# 4. Analyze, identify patterns, and explore the data.
# 5. Model, predict and solve the problem.
# 6. Visualize, report, and present the problem solving steps and final solution.
# 7. Supply or submit the results.
# 
# 
# **Workflow goals**
# 
# **Classifying** - We may want to classify or categorize our samples. We may also want to understand the implications or correlation of different classes with our solution goal.
# 
# **Correlating** - One can approach the problem based on available features within the training dataset. Which features within the dataset contribute significantly to our solution goal? Statistically speaking is there a correlation among a feature and solution goal? As the feature values change does the solution state change as well, and visa-versa? This can be tested both for numerical and categorical features in the given dataset. We may also want to determine correlation among features other than survival for subsequent goals and workflow stages. Correlating certain features may help in creating, completing, or correcting features.
# 
# **Converting** - For modeling stage, one needs to prepare the data. Depending on the choice of model algorithm one may require all features to be converted to numerical equivalent values. So for instance converting text categorical values to numeric values.
# 
# **Completing** - Data preparation may also require us to estimate any missing values within a feature. Model algorithms may work best when there are no missing values.
# 
# **Correcting** - We may also analyze the given training dataset for errors or possibly innacurate values within features and try to corrent these values or exclude the samples containing the errors. One way to do this is to detect any outliers among our samples or features. We may also completely discard a feature if it is not contribting to the analysis or may significantly skew the results.
# 
# **Creating** - Can we create new features based on an existing feature or a set of features, such that the new feature follows the correlation, conversion, completeness goals.
# 
# **Charting** - How to select the right visualization plots and charts depending on nature of the data and the solution goals.  We may want to classify or categorize our samples. We may also want to understand the implications or correlation of different classes with our solution goal.
# 

# In[1]:


# step 1 is to import some useful libraries for data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd


# # Getting the data
# 
# 1. Getting file into Pandas DataFrame
# 2. Combine training and test dataset together for some operations analysis

# In[3]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df,test_df]


# # Describing the data
# 
# >we can check the head(), info(), desribe(), columns to sort of get a feel of what the data look like

# In[4]:


train_df.head()


# # Check out the info for the dataset
# 
# 1. Find out which columns are categorical, which are numerical will help us select appropriate plots for visulazation in the future. 
# > Categorical columns: Survived, Sex, Embarked, Ordinal: Pclass
# >
# > Numerical columns: Age, Fare, SibSp, Parch
# 
# 2. Find out which columns have mixed data types: these are candidates for correcting goals. 
#     * Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.
# 
# 3. Find out columsn may contain erros or typos: this is hard to review for a large dataset. Reviewing a few samples from a smaller dataset may just tell us alright, which features may require correcting.
#     * Name column might be one
# 

# In[5]:


train_df.info()

# we can see that there are 12 columns in the training dataset. 
    # Numerical columns: 
        # Continous: Age, Fare
        # Discrete: SibSp, Parch
    # Categorical columns: 
        # Survived, Sex, Embarked, 
        # Ordinal: Pclass


# In[6]:


# check out column names
print(train_df.columns.values)


# In[7]:


train_df.tail()


# * Find out which columns contain blank, null or empty values: 
#     * Age, Cabin, Embarked contain NA in training dataset
#     * Age, Fare, Cabin contain NA in test dataset    

# In[8]:


train_df.isnull().any()


# In[9]:


test_df.isnull().any()


# In[10]:


# see data info

train_df.info()
print('_'*40)
test_df.info()


# In[11]:


train_df.describe()


# The describe() function tells us the following numerical information: 
# 
# *  Total samples are 891, represeting 891/2224 = 40% of the total passengers on board
# *  Survived is a categorical column with 1 or 0
# *  mean of 38% for survived, meaning the sample survival rate is 38% vs. the actual survival rate of 32%
# *  Average age is aboout 30, max age is 80, but less than 25% of them are over 38 years old.
# *  Most passengers ( > 75%) did not travel with parents of children. 
# *  Fares varied significantly from 0 to 512

# In[12]:


train_df.describe(include=['O'])


# The describe(include=['O'] ) tells us the following categorical information: 
# 
# *  Names are all unique, 891 unique value
# *  Sex is unique too, data have more male than female, 577/891 = 65% male
# *  Ticket has duplicate value 1 - 681/891 = 23.5%
# *  Cabin also have many duplicate value. Or many several passengers shared a cabin 
# *  Embarked have 3 possible values, with S be the most frequent one

# # Analyze by pivoting features

# In[13]:


# how Pclass associated with survival?
 # look slike higher class had higher chance of survival rate. 

train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', 
                                                                                      ascending = False)


# In[14]:


# How gender associate with survival?
# female has 74% of survival rate vs. male only has 18%
train_df[['Sex','Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', 
                                                                                  ascending = False)


# In[15]:


# how sibsp associate with survival?
# SibSp = 1 or 2 had the highest survival rate
train_df[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',
                                                                                    ascending = False)


# In[16]:


# how Parent and Child associate with survival?

train_df[['Parch','Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by='Survived',
                                                                                      ascending = False)


# By analyzing the data using pivoting features, we know the following: 
# 
# * **Pclass**: Pclass = 1 has the highest survival rate. Pclass is defintely associated with survived. So included in the model 
# * **Sex**: female is much likely to survive compared to male. Def included Sex. 
# * **SibSp and Parch**: These features have zero correlation for certain values. It may be best to derive a feature or a set of features from these individual features

# # Analyze by visualizing data
# 
# **Correlating numerical features**
# 
# Let us start by understanding correlations between numerical features and our solution goal (Survived).
# 
# A histogram chart is useful for analyzing continous numerical variables like Age where banding or ranges will help identify useful patterns. The histogram can indicate distribution of samples using automatically defined bins or equally ranged bands. This helps us answer questions relating to specific bands (Did infants have better survival rate?)
# 
# 
# Note that x-axis in historgram visualizations represents the count of samples or passengers.

# In[17]:


# import visulization libaries
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[18]:


g = sns.FacetGrid(train_df, col = 'Survived')
g = g.map(plt.hist, 'Age', bins = 20)


# **Age vs. Survived visulization**
# 
# 1.  Infants had high survival rate
# 2.  Oldest passengers (age = 80) survived
# 3.  Age from 15 - 25 did not survived many
# 4.  Most passengers are in 15 - 35 range
# 
# **Decisions: **
# 1. We should consider Age in our model
# 2. We should complete Age for empty cells
# 3. We should band age groups

# In[19]:


grid = sns.FacetGrid(train_df, col = 'Pclass', row = 'Survived')
grid = grid.map(plt.hist, 'Age', alpha = 0.5, bins = 20)
grid.add_legend()


# **Age, Pclass vs. Survived**
# 
# 1. Pclass = 3 has most passengers, but most of them didn't survive. 
# 2. Infants in Pclass 2 and 3 mostly survived. 
# 3. most passengers in PClass = 1 survived. 
# 
# **Decisions:** consider Pclass for model training. 
# 

# In[20]:


grid = sns.FacetGrid(train_df,col='Pclass', hue = 'Survived')
grid.map(plt.hist, 'Age', alpha = 0.5)
grid.add_legend()


# In[21]:


train_df.head()


# **Correlating categorical features**

# In[22]:


grid = sns.FacetGrid(train_df, col='Embarked')
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# **Observations**
# 
# 1. Female has higher survival rate than males, expect in Embarked = C. But embarked C might associated with Pclass
# 
# **Decisions**
# 
# 1. Add Sex to model
# 2. Complete and add Embarked to model training as we remember Embarked has missing value
# 

# **Correlating categorical and numerical features**
# 
# We may also want to correlate categorical features (with non-numeric values) and numeric features. We can consider correlating Embarked (Categorical non-numeric), Sex (Categorical non-numeric), Fare (Numeric continuous), with Survived (Categorical numeric).

# In[23]:


grid = sns.FacetGrid(train_df, row = 'Embarked', col = 'Survived')
grid.map(sns.barplot, 'Sex', 'Fare', alpha = 0.5, ci=None)
grid.add_legend()


# **Observations**: Higher fare paying passengers had better survival rate. 
# 
# **Decisions**: Consider banding fare

# # Wrangle Data
# 
# **Correcting data by dropping features: **
# - Based on our study, we can drop Cabin and Ticket
# - Remember to drop the same features in both training and test set to be consistent. 

# In[24]:


train_df = train_df.drop(['Ticket', 'Cabin'], axis = 1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis = 1)
combine = [train_df, test_df]


# In[25]:


train_df.head()


# **Creating new feature extracting from existing**
# 
# **Name: **We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.
# 
# In the following code we extract Title feature using regular expressions. The RegEx pattern (\w+\.) matches the first word which ends with a dot character within Name feature. The expand=False flag returns a DataFrame.
# 

# In[26]:


for dataset in combine: 
    dataset['Title']= dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

pd.crosstab(train_df['Title'], train_df['Sex'])


# In[27]:


# We can replace many titles with a more common name or classify them as Rare.

for dataset in combine: 
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Dr', 'Lady','Capt', 'Col',
                                                'Don','Jonkheer', 'Major', 'Rev', 'Sir'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values('Survived', 
                                                                                     ascending = False)
                                       


# **Observations: **
#     
# * Most titles band Age groups accurately. For example: Master title has Age mean of 5 years.
# * Survival among Title Age bands varies slightly.
# * Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer).
# 
# **Decisions: **added Title for model training 

# In[28]:


train_df.head()


# In[29]:


grid =sns.FacetGrid(train_df, row = 'Survived', col = 'Title')
grid.map(plt.hist, 'Age')


# In[30]:


# since title is categorical, we would like to convert it to numerical
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}

for dataset in combine: 
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)

train_df.head()


# In[31]:


test_df.head()


# In[32]:


# then we can drop name now
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


# **Converting a categorical feature**
# 
# Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.

# In[33]:


sex_mapping = {'female': 1, 'male': 0}
for dataset in combine: 
    dataset['Sex'] = dataset['Sex'].map(sex_mapping).astype(int)


# In[34]:


train_df.head()


# **Completing a numerical continuous feature**
# 
# Now we should start estimating and completing features with missing or null values. We will first do this for the Age feature.
# 
# We can consider three methods to complete a numerical continuous feature.
# 
# 1. A simple way is to generate random numbers between mean and standard deviation.
# 2. More accurate way of guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender, and Pclass. Guess Age values using median values for Age across sets of Pclass and Gender feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...
# 3. Combine methods 1 and 2. So instead of guessing age values based on median, use random numbers between mean and standard deviation, based on sets of Pclass and Gender combinations.
# 
# Method 1 and 3 will introduce random noise into our models. The results from multiple executions might vary. We will prefer method 2.
# 

# In[35]:


grid = sns.FacetGrid(train_df, row = 'Pclass', col = 'Sex')
grid.map(plt.hist, 'Age', alpha = 0.5, bins = 20)
grid.add_legend()


# In[36]:


guess_ages = np.zeros((2,3))
guess_ages


# In[37]:


for dataset in combine: 
    for i in range(0,2):
        for j in range(0,3): 
            # a list of non NA age by sex and pclass
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1) ]['Age'].dropna()
            
            age_guess = guess_df.median()
            
            # convert random age float to nearest 0.5 age
            guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5
            
    # assign the guess_ages back to dataset
    for i in range(0,2): 
        for j in range(0,3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i ) & (dataset.Pclass == j+1),                         'Age']= guess_ages[i,j]

    dataset['Age'] =dataset['Age'].astype(int)
    
train_df.head()


# create Age band and determine correlations with Survived

# In[38]:


train_df['AgeBand'] = pd.cut(train_df['Age'],5)


# In[39]:


train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index = False).mean().sort_values('AgeBand', 
                                                                                           ascending = True)


# In[40]:


# Let us replace Age with ordinals based on these age bands
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train_df.head()


# In[41]:


test_df.head()


# In[42]:


# then we can now remove ageband

train_df = train_df.drop('AgeBand', axis = 1)
combine = [train_df, test_df]
train_df.head()


# In[43]:


# Create new feature combining existing features
# We can create a new feature for FamilySize which combines Parch and SibSp. 
# This will enable us to drop Parch and SibSp from our datasets.

for dataset in combine: 
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']+1
    
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean().sort_values('Survived', 
                                                                                                 ascending = False)


# In[44]:


train_df.head()


# In[45]:


# create another feature called isalone

for dataset in combine: 
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index = False).mean()


# In[46]:


train_df.head()


# In[47]:


test_df.head()


# In[48]:


# now we can drop SibSp, Parch, FamilySize in favor of IsAlone
train_df = train_df.drop(['SibSp', 'Parch', 'FamilySize'], axis = 1)
test_df = test_df.drop(['SibSp', 'Parch', 'FamilySize'], axis = 1)


# In[49]:


combine = [train_df, test_df]
train_df.head()


# In[50]:


# We can also create an artificial feature combining Pclass and Age.

for dataset in combine: 
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
    
train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head()


# In[51]:


# see which one still has missing value
train_df.isnull().any()


# In[52]:


# noticed that Embarked still has missing value
train_df.describe(include=['O'])

# we can fill in the missing value with the most common occurance


# In[53]:


freq_port = train_df['Embarked'].dropna().mode()[0]
freq_port


# In[54]:


for dataset in combine: 
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)


# In[55]:


train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean()

# Looks like C has the highest survival rate


# **Converting categorical feature to numeric**

# In[56]:


embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
for dataset in combine: 
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping).astype(int)
    
train_df.head()


# In[62]:


test_df.describe()
# we can see Fair is missing one value 
# 417 total count vs. test dataset has 418 count


# **Quick completing and converting a numeric feature**
# 
# We can now complete the Fare feature for single missing value in test dataset using mode to get the value that occurs most frequently for this feature.
# 
# The completion goal achieves desired requirement for model algorithm to operate on non-null values.

# In[66]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace = True)
test_df.head()


# In[68]:


test_df.describe()

# note that Fare now doesn't have any missing value


# In[69]:


# create fareband
train_df['FareBand'] = pd.qcut(train_df['Fare'],q = 4)


# In[72]:


train_df[['FareBand','Survived']].groupby(['FareBand'], as_index = False).mean().sort_values('FareBand', ascending = True)


# In[74]:


test_df.head()


# In[80]:


# convert fareband to oridinal numerical columns
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    


# In[81]:


# we can now drop fare band for train_df
train_df = train_df.drop('FareBand',axis =1)

combine = [train_df,test_df]

train_df.head()


# In[82]:


test_df.head()


# # Model, predict and solve
# 
# Our problem is a classification and regression problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models to a few. These include:
# 
# * Logistic Regression
# * KNN or k-Nearest Neighbors
# * Support Vector Machines
# * Naive Bayes classifier
# * Decision Tree
# * Random Forrest
# * Perceptron
# * Artificial neural network
# * RVM or Relevance Vector Machine

# In[88]:


# import all the model library
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.svm import SVC, LinearSVC # SVM
from sklearn.naive_bayes import GaussianNB #Naive Bayes classifier
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.ensemble import RandomForestClassifier #Random Forrest 
from sklearn.linear_model import Perceptron #Perceptron
from sklearn.linear_model import SGDClassifier 


# In[83]:


# set the train and test dataset
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[93]:


train_df.head()


# In[89]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# We can use Logistic Regression to validate our assumptions and decisions for feature creating and completing goals. This can be done by calculating the coefficient of the features in the decision function.
# 
# Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).

# In[95]:


# check the coefficient of the model
coef_df = pd.DataFrame(train_df.columns.delete(0))
coef_df.columns = ['Feature']
coef_df["Correlation"] = pd.Series(logreg.coef_[0])

coef_df.sort_values(by='Correlation', ascending=False)


# ** Support Vector Machines **
# 
# Next we model using Support Vector Machines which are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training samples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new test samples to one category or the other, making it a non-probabilistic binary linear classifier. 

# In[98]:


# SVC
svc = SVC()
svc.fit(X_train,Y_train)
Y_pred=svc.predict(X_test)
acc_svc = round(svc.score(X_train,Y_train)* 100 ,2)
acc_svc

# we can see that the score is 83.84 > logistic regression score of 80.36


# ** KNN **
# 
# In pattern recognition, the k-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric method used for classification and regression. A sample is classified by a majority vote of its neighbors, with the sample being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.

# In[99]:


knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# ** Naive Bayes Classifiers **
# 
# In machine learning, naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features) in a learning problem. 

# In[101]:


# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train)*100,2)
acc_gaussian

# this the worst score so far


# ** Perceptron **
# 
# The perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not). It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. The algorithm allows for online learning, in that it processes elements in the training set one at a time. 

# In[102]:


# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100,2)
acc_perceptron


# In[104]:


# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train)*100,2)
acc_linear_svc


# In[107]:


# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train)*100,2)
acc_sgd


# ** Decision Tree **
# 
# This model uses a decision tree as a predictive model which maps features (tree branches) to conclusions about the target value (tree leaves). Tree models where the target variable can take a finite set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees.

# In[108]:


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred= decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train)*100,2)
acc_decision_tree


# ** Random Forests**
# 
# The next model Random Forests is one of the most popular. Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees (n_estimators=100) at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

# In[110]:


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train)*100,2)
acc_random_forest


# **Rank all the models**

# In[112]:


models = pd.DataFrame(
    {'Model': ['Logistic Regression','KNN','Support Vector Machines','Naive Bayes','Decision Tree','Random Forrest','Perceptron',
               'Stochastic Gradient Decent','Linear SVC'],
     'Score': [acc_log,acc_knn, acc_svc, acc_gaussian,acc_decision_tree,acc_random_forest,acc_perceptron,acc_sgd, acc_linear_svc]
    }
)

models.sort_values(by='Score',ascending = False)

# we can see the best one are Decision Tree and Random Forrest. Since Randomw Forrest is an ensemble of Decision tree, 
# so it will perform better in the long run


# In[114]:


# we will use the Random Forrest model for submission
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)

my_submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

# export the submission file
my_submission.to_csv('submission.csv', index = False)


# In[ ]:




