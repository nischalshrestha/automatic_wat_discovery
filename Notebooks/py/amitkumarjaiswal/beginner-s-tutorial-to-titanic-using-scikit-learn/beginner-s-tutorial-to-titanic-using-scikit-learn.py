#!/usr/bin/env python
# coding: utf-8

# > *Author: Amit Kumar Jaiswal*
# 
# # Beginner's Tutorial to Titanic Survival Challenge using scikit-learn
# 
# ## Import the libraries
# As the first step all neccessary libraries will be imported; this list will be updated as we are going forward

# In[1]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score


# ## Getting to know your data
# As usual the first step is to get to know the data; how many samples, what are the attributes, what are the missing data.

# In[4]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]


# ## Analyze by describing data
# Pandas also helps describe the datasets answering following questions early in our project.
# ### Which features are available in the dataset?
# Noting the feature names for directly manipulating or analyzing these. These feature names are described on the [Kaggle data page here](https://www.kaggle.com/c/titanic/data).

# In[3]:


print(train_df.columns.values)


# ### Which features are categorical?
# These values classify the samples into sets of similar samples. Within categorical features are the values nominal, ordinal, ratio, or interval based? Among other things this helps us select the appropriate plots for visualization.
# * Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
# 
# ### Which features are numerical?
# Which features are numerical? These values change from sample to sample. Within numerical features are the values discrete, continuous, or timeseries based? Among other things this helps us select the appropriate plots for visualization.
# * Continous: Age, Fare. Discrete: SibSp, Parch.

# In[6]:


# preview the data
train_df.head()


# ### Which features are mixed data types?
# Numerical, alphanumeric data within same feature. These are candidates for correcting goal.
# * Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.
# 
# ### Which features may contain errors or typos?
# This is harder to review for a large dataset, however reviewing a few samples from a smaller dataset may just tell us outright, which features may require correcting.
# * Name feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets, and quotes used for alternative or short names.

# In[7]:


train_df.tail()


# ### Which features contain blank, null or empty values?
# These will require correcting.
# * Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
# * Cabin > Age are incomplete in case of test dataset.
# 
# ### What are the data types for various features?
# Helping us during converting goal.
# * Seven features are integer or floats. Six in case of test dataset.
# * Five features are strings (object).

# In[8]:


train_df.info()
print('_'*40)
test_df.info()


# ### What is the distribution of numerical feature values across the samples?
# This helps us determine, among other early insights, how representative is the training dataset of the actual problem domain.
# * Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).
# * Survived is a categorical feature with 0 or 1 values.
# * Around 38% samples survived representative of the actual survival rate at 32%.
# * Most passengers (> 75%) did not travel with parents or children.
# * Nearly 30% of the passengers had siblings and/or spouse aboard.
# * Fares varied significantly with few passengers (<1%) paying as high as $512.
# * Few elderly passengers (<1%) within age range 65-80.

# In[9]:


train_df.describe()


# ### What is the distribution of categorical features?
# * Names are unique across the dataset (count=unique=891)
# * Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
# * Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
# * Embarked takes three possible values. S port used by most passengers (top=S)
# * Ticket feature has high ratio (22%) of duplicate values (unique=681).

# In[10]:


train_df.describe(include=['O'])


# ## Analyze by pivoting features
# To confirm some of our observations and assumptions, we can quickly analyze our feature correlations by pivoting features against each other. We can only do so at this stage for features which do not have any empty values. It also makes sense doing so only for features which are categorical (Sex), ordinal (Pclass) or discrete (SibSp, Parch) type.
# * **Pclass** We observe significant correlation (>0.5) among Pclass=1 and Survived (classifying #3). We decide to include this feature in our model.
# * **Sex** We confirm the observation during problem definition that Sex=female had very high survival rate at 74% (classifying #1).
# * **SibSp and Parch** These features have zero correlation for certain values. It may be best to derive a feature or a set of features from these individual features (creating #1).

# In[11]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[12]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[13]:


train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[14]:


train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ## Analyze by visualizing data
# Now we can continue confirming some of our assumptions using visualizations for analyzing the data.
# ### Correlating numerical features
# Let us start by understanding correlations between numerical features and our solution goal (Survived).
# A histogram chart is useful for analyzing continous numerical variables like Age where banding or ranges will help identify useful patterns. The histogram can indicate distribution of samples using automatically defined bins or equally ranged bands. This helps us answer questions relating to specific bands (Did infants have better survival rate?)
# Note that x-axis in historgram visualizations represents the count of samples or passengers.
# 
# **Observations:**
# * Infants (Age <=4) had high survival rate.
# * Oldest passengers (Age = 80) survived.
# * Large number of 15-25 year olds did not survive.
# * Most passengers are in 15-35 age range.
# 
# **Decisions:**
# 
# This simple analysis confirms our assumptions as decisions for subsequent workflow stages.
# * We should consider Age (our assumption classifying #2) in our model training.
# * Complete the Age feature for null values (completing #1).
# * We should band age groups (creating #3).

# In[15]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# ### Correlating numerical and ordinal features
# We can combine multiple features for identifying correlations using a single plot. This can be done with numerical and categorical features which have numeric values.
# **Observations:**
# * Pclass=3 had most passengers, however most did not survive. Confirms our classifying assumption #2.
# * Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies our classifying assumption #2.
# * Most passengers in Pclass=1 survived. Confirms our classifying assumption #3.
# * Pclass varies in terms of Age distribution of passengers.
# 
# **Decisions:**
# * Consider Pclass for model training.

# In[16]:


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# ### Correlating categorical features
# Now we can correlate categorical features with our solution goal.
# 
# **Observations:**
# * Female passengers had much better survival rate than males. Confirms classifying (#1).
# * Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived.
# * Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports. Completing (#2).
# * Ports of embarkation have varying survival rates for Pclass=3 and among male passengers. Correlating (#1).
# 
# **Decisions:**
# * Add Sex feature to model training.
# * Complete and add Embarked feature to model training.

# In[17]:


grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# ### Correlating categorical and numerical features
# We may also want to correlate categorical features (with non-numeric values) and numeric features. We can consider correlating Embarked (Categorical non-numeric), Sex (Categorical non-numeric), Fare (Numeric continuous), with Survived (Categorical numeric).
# 
# **Observations:**
# * Higher fare paying passengers had better survival. Confirms our assumption for creating (#4) fare ranges.
# * Port of embarkation correlates with survival rates. Confirms correlating (#1) and completing (#2).
# 
# **Decisions:**
# * Consider banding Fare feature.

# In[18]:


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# ### Wrangle data
# We have collected several assumptions and decisions regarding our datasets and solution requirements. So far we did not have to change a single feature or value to arrive at these. Let us now execute our decisions and assumptions for correcting, creating, and completing goals.
# 
# ### Correcting by dropping features
# This is a good starting goal to execute. By dropping features we are dealing with fewer data points. Speeds up our notebook and eases the analysis.
# Based on our assumptions and decisions we want to drop the Cabin (correcting #2) and Ticket (correcting #1) features.
# 
# Note that where applicable we perform operations on both training and testing datasets together to stay consistent.

# In[19]:


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape


# ### Creating new feature extracting from existing
# We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.
# 
# In the following code we extract Title feature using regular expressions. The RegEx pattern (\w+\.) matches the first word which ends with a dot character within Name feature. The expand=False flag returns a DataFrame.
# 
# **Observations:**
# 
# When we plot Title, Age, and Survived, we note the following observations.
# * Most titles band Age groups accurately. For example: Master title has Age mean of 5 years.
# * Survival among Title Age bands varies slightly.
# * Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer).
# 
# **Decision:**
# * We decide to retain the new Title feature for model training.

# In[20]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])


# We can replace many titles with a more common name or classify them as Rare.

# In[21]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Jonkheer', 'Dona'], 'Lady')
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Don', 'Major', 'Sir'], 'Sir')
    
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# We can convert the categorical titles to ordinal.

# In[22]:


title_mapping = {"Col": 1, "Dr": 2, "Lady": 3, "Master": 4, "Miss": 5, "Mr": 6, "Mrs": 7, "Rev": 8, "Sir": 9}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()


# Now we can safely drop the Name feature from training and testing datasets. We also do not need the `PassengerId` feature in the training dataset.

# In[23]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


# ### Converting a categorical feature
# Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.
# 
# Let us start by converting Sex feature to a new feature called Gender where female=1 and male=0.

# In[24]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# ### Completing a numerical continuous feature
# Now we should start estimating and completing features with missing or null values. We will first do this for the Age feature.
# 
# We can consider three methods to complete a numerical continuous feature.
# 
# * A simple way is to generate random numbers between mean and [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation).
# * More accurate way of guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender, and Pclass. Guess Age values using [median](https://en.wikipedia.org/wiki/Median) values for Age across sets of Pclass and Gender feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...
# * Combine methods 1 and 2. So instead of guessing age values based on median, use random numbers between mean and standard deviation, based on sets of Pclass and Gender combinations.
# 
# Method 1 and 3 will introduce random noise into our models. The results from multiple executions might vary. We will prefer method 2.

# In[25]:


grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# Let us start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations.

# In[26]:


guess_ages = np.zeros((2,3))
guess_ages


# Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.

# In[27]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


# Let us create Age bands and determine correlations with Survived.

# In[28]:


train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# Let us replace Age with ordinals based on these bands.

# In[29]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()


# We can not remove the AgeBand feature.

# In[30]:


train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()


# ### Create new feature combining existing features
# We can create a new feature for FamilySize which combines Parch and SibSp. This will enable us to drop Parch and SibSp from our datasets.

# In[31]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# We can create another feature called IsAlone.

# In[32]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone.

# In[33]:


train_df = train_df.drop(['Parch',], axis=1)
test_df = test_df.drop(['Parch'], axis=1)
combine = [train_df, test_df]

train_df.head()


# We can also create an artificial feature combining Pclass and Age.

# In[34]:


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# ### Completing a categorical feature
# Embarked feature takes S, Q, C values based on port of embarkation. Our training dataset has two missing values. We simply fill these with the most common occurance.

# In[35]:


freq_port = train_df.Embarked.dropna().mode()[0]
freq_port


# In[36]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ### Converting categorical feature to numeric
# We can now convert the EmbarkedFill feature by creating a new numeric Port feature.

# In[37]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# ## Quick completing and converting a numeric feature
# We can now complete the Fare feature for single missing value in test dataset using mode to get the value that occurs most frequently for this feature. We do this in a single line of code.
# 
# Note that we are not creating an intermediate new feature or doing any further analysis for correlation to guess missing feature as we are replacing only a single value. The completion goal achieves desired requirement for model algorithm to operate on non-null values.
# 
# We may also want round off the fare to two decimals as it represents currency.

# In[38]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


# We can not create FareBand.

# In[39]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# Convert the Fare feature to ordinal values based on the FareBand.

# In[40]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


# In[41]:


# Test dataset
test_df.head(10)


# ## Model, predict and solve
# Now we are ready to train a model and predict the required solution. There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Our problem is a classification and regression problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. 
# With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models to a few. These include:
# * Logistic Regression
# * KNN or k-Nearest Neighbors
# * Support Vector Machines
# * Naive Bayes classifier
# * Decision Tree
# * Random Forrest
# * Perceptron
# * Artificial neural network
# * RVM or Relevance Vector Machine

# In[42]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.head(10)


# In[43]:


Y_train.head(10)


# In[44]:


X_test.head(10)


# Logistic Regression is a useful model to run early in the workflow. Logistic regression measures the relationship between the categorical dependent variable (feature) and one or more independent variables (features) by estimating probabilities using a logistic function, which is the cumulative logistic distribution. 
# Reference [Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression).
# 
# Note the confidence score generated by the model based on our training dataset.

# In[45]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# We can use Logistic Regression to validate our assumptions and decisions for feature creating and completing goals. This can be done by calculating the coefficient of the features in the decision function.
# 
# Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).
# * Sex is highest positivie coefficient, implying as the Sex value increases (male: 0 to female: 1), the probability of Survived=1 increases the most.
# * Inversely as Pclass increases, probability of Survived=1 decreases the most.
# * This way Age*Class is a good artificial feature to model as it has second highest negative correlation with Survived.
# * So is Title as second highest positive correlation.

# In[46]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# Next we model using Support Vector Machines which are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training samples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new test samples to one category or the other, making it a non-probabilistic binary linear classifier. Reference [Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine).
# 
# Note that the model generates a confidence score which is higher than Logistics Regression model.

# In[48]:


# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In pattern recognition, the k-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric method used for classification and regression. A sample is classified by a majority vote of its neighbors, with the sample being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. Reference [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).
# 
# KNN confidence score is better than Logistics Regression but worse than SVM.

# In[49]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In machine learning, naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features) in a learning problem. Reference [Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier).
# 
# The model generated confidence score is the lowest among the models evaluated so far.

# In[50]:


# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# The perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not). It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. The algorithm allows for online learning, in that it processes elements in the training set one at a time. 
# Reference [Wikipedia](https://en.wikipedia.org/wiki/Perceptron).

# In[51]:


# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# This model uses a decision tree as a predictive model which maps features (tree branches) to conclusions about the target value (tree leaves). Tree models where the target variable can take a finite set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning).
# 
# The model confidence score is the highest among models evaluated so far.

# In[52]:


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# The next model Random Forests is one of the most popular. Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees (n_estimators=100) at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Random_forest).
# 
# The model confidence score is the highest among models evaluated so far. We decide to use this model's output (Y_pred) for creating our competition submission of results.

# In[53]:


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[54]:


# Gradient Boosting
grad_boost = GradientBoostingClassifier(n_estimators = 100)
grad_boost.fit(X_train, Y_train)
Y_pred = grad_boost.predict(X_test)
grad_boost.score(X_train, Y_train)
acc_grad_boost = round(grad_boost.score(X_train, Y_train) * 100, 2)
acc_grad_boost


# In[55]:


# RidgeClassifierCV
Ridge= RidgeClassifierCV()
Ridge.fit(X_train, Y_train)
Y_pred = Ridge.predict(X_test)
acc_Ridge= round(Ridge.score(X_train, Y_train) * 100, 2)
acc_Ridge


# ## Model evaluation
# We can now rank our evaluation of all the models to choose the best one for our problem.

# In[56]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron', 'Grad boost','Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, acc_perceptron, acc_grad_boost, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)


# ## Optimizing Model
# Lets try to do a grid search on a few hyperparameters to get the best results from the Random Forest and from the gradient boosting to try to improve the overall accuracy

# In[57]:


# Split the training set into a development and an evaluation sets
from sklearn.model_selection import train_test_split
X_dev, X_eval, y_dev, y_eval = train_test_split(X_train,
                                                Y_train,
                                                test_size=0.2,
                                                random_state=42)

X_test.head(10)


# In[58]:


# 1. Random Forest
import time
dict_clf = {}


paramgrid = {
    'n_estimators':      [100, 150, 200, 250, 300, 400, 500],
    'criterion':         ['gini', 'entropy'],
    'max_features':      ['auto', 'log2'],
    'min_samples_leaf':  list(range(2, 8))
}
GS = GridSearchCV(RandomForestClassifier(random_state=77),
                  paramgrid,
                  cv=4)

# Fit the data and record time taking to train
t0 = time.time()
GS.fit(X_dev, y_dev)
t = time.time() - t0

# Store best parameters, score and estimator
best_clf = GS.best_estimator_
best_params = GS.best_params_
best_score = GS.best_score_

name = 'RF'


# In[59]:


best_clf.fit(X_dev, y_dev)
acc_eval = accuracy_score(y_eval, best_clf.predict(X_eval))

dict_clf[name] = {
    'best_par': best_params,
    'best_clf': best_clf,
    'best_score': best_score,
    'score_eval': acc_eval,
    'fit_time': t,
}

acc_eval


# In[60]:


# 2. GradientBoosting
paramgrid = {
    'n_estimators':      [100, 150, 200, 250, 300, 400, 500],
    'max_features':      ['auto', 'log2'],
    'min_samples_leaf':  list(range(2, 7)),
    'loss' :             ['deviance', 'exponential'],
    'learning_rate':     [0.025, 0.05, 0.075, 0.1],
}
GS = GridSearchCV(GradientBoostingClassifier(random_state=77),
                  paramgrid,
                  cv=4)

# Fit the data and record time taking to train
t0 = time.time()
GS.fit(X_dev, y_dev)
t = time.time() - t0

# Store best parameters, score and estimator
best_clf = GS.best_estimator_
best_params = GS.best_params_
best_score = GS.best_score_

name = 'GB'
best_clf.fit(X_dev, y_dev)
acc_eval = accuracy_score(y_eval, best_clf.predict(X_eval))

dict_clf[name] = {
    'best_par': best_params,
    'best_clf': best_clf,
    'best_score': best_score,
    'score_eval': acc_eval,
    'fit_time': t,
}


# In[61]:


acc_eval


# In[62]:


for clf in dict_clf.keys():
    print("{0} classifier:\n\t- Best score = {1:.2%}".format(clf, dict_clf[clf]['best_score']))
    print("\t- Score on evaluation set = {0:.2%}".format(dict_clf[clf]['score_eval']))
    print("\t- Fitting time = {0:.1f} min".format(round(dict_clf[clf]['fit_time']/60, 1)))
    print("\t- Best parameters:")
    for par in sorted(dict_clf[clf]['best_par'].keys()):
        print("\t\t* {0}: {1}".format(par, dict_clf[clf]['best_par'][par]))


# ## Ensembling
# Now we include a soft voting of the top 3-5 (stacking) trying to improve further more the results

# In[63]:


from sklearn.ensemble import VotingClassifier

estimators = [('RF', dict_clf['RF']['best_clf']),
              ('GB', dict_clf['GB']['best_clf']),
              ('KNN', knn), ('svc', svc), ('trees', decision_tree)]

# Instantiate the VotingClassifier using hard voting
voter = VotingClassifier(estimators=estimators, voting='hard')
voter.fit(X_train, Y_train)

Y_pred = voter.predict(X_test).astype(int)


# In[64]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)


# ## Conclusion
# At this moment the model is achieving 78% accuracy being almost top 30%. However this is not a good result at all. Knowing that we've gone all the way using cross validation, ensembles and stacking it seems that we are a bit off.
# 
# * Apply feature engeneering to achieve higher score (objective of around 82%).
