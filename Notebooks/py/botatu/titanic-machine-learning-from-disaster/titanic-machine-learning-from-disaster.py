#!/usr/bin/env python
# coding: utf-8

# ## How To Start with Supervised Learning
# 
# As you might already know, a good way to approach supervised learning is the following:
# 
# * Perform an Exploratory Data Analysis (EDA) on your data set;
# * Build a quick and dirty model, or a baseline model, which can serve as a comparison against later models that you will build;
# * Iterate this process. You will do more EDA and build another model;
# * Engineer features: take the features that you already have and combine them or extract more information from them to eventually come to the last point, which is
# * Get a model that performs better.
# 
# We shall be performing all the above steps to get a good model that predicts the survival rate of people on Titanic.

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

get_ipython().magic(u'matplotlib inline')
sns.set()


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

df_train.head()


# In[ ]:


df_test.head()


# Use DataFrame .info() method to check out data types, missing values and more (of df_train).

# In[ ]:


df_train.info()


# In this case, you see that there are only 714 non-null values for the 'Age' column in a DataFrame with 891 rows. This means that are are 177 null or missing values.
# 
# Also, use the DataFrame .describe() method to check out summary statistics of numeric columns (of df_train).

# In[ ]:


df_train.describe()


# ### Exploratory Data Analysis (EDA)

# In[ ]:


sns.countplot(x='Survived', data=df_train)


# **Take-away**: in the training set, less people survived than didn't. Let's then build a first model that predicts that nobody survived.

# In[ ]:


df_test['Survived'] = 0
my_submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': df_test['Survived']})
my_submission.to_csv('submission.csv', index=False)


# In[ ]:


my_submission.head()


# Submit the results to Kaggle and you should get an accuracy of **62.7%**. Not too bad for the first iteration.

# ### EDA on Feature Variables
# 
# Now that you have made a quick-and-dirty model, it's time to reiterate: let's do some more Exploratory Data Analysis and build another model soon!
# 
# * You can use seaborn to build a bar plot of the Titanic dataset feature 'Sex' (of df_train).
# * Also, use seaborn to build bar plots of the Titanic dataset feature 'Survived' split (faceted) over the feature 'Sex'.

# In[ ]:


sns.countplot(x='Sex', data=df_train)


# In[ ]:


sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train)


# **Take-away**: Women were more likely to survive than men.

# In[ ]:


df_train.groupby(['Sex']).Survived.sum()


# In[ ]:


print(df_train[df_train['Sex'] == 'female'].Survived.sum()/df_train[df_train['Sex'] == 'female'].Survived.count())
print(df_train[df_train['Sex'] == 'male'].Survived.sum()/df_train[df_train['Sex'] == 'male'].Survived.count())


# 74% of women survived, while 19% of men survived.
# 
# Let's now build a second model and predict that all women survived and all men didn't. Once again, this is an unrealistic model, but it will provide a baseline against which to compare future models.
# 
# Create a column 'Survived' for df_test that encodes the above prediction.
# Save 'PassengerId' and 'Survived' columns of df_test to a .csv and submit to Kaggle.

# In[ ]:


df_test['Survived'] = df_test.Sex == 'female'
df_test['Survived'] = df_test.Survived.apply(lambda x: int(x))
df_test.head()


# In[ ]:


my_submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': df_test['Survived']})
my_submission.to_csv('submission.csv', index=False)


# The accuracy on Kaggle for the above submission will be around **76.6%**
# 
# ### Explore Your Data More!
# * Use seaborn to build bar plots of the Titanic dataset feature 'Survived' split (faceted) over the feature 'Pclass'.

# In[ ]:


sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train)


# **Take-away**: Passengers that travelled in first class were more likely to survive. On the other hand, passengers travelling in third class were more unlikely to survive.
# 
# * Use seaborn to build bar plots of the Titanic dataset feature 'Survived' split (faceted) over the feature 'Embarked'.

# In[ ]:


sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train)


# **Take-away**: Passengers that embarked in Southampton were less likely to survive.
# 
# ### EDA with Numeric Variables
# Use seaborn to plot a histogram of the 'Fare' column of df_train.

# In[ ]:


sns.distplot(df_train.Fare, kde=False)


# **Take-away**: Most passengers paid less than 100 for travelling with the Titanic.
# 
# * Use a pandas plotting method to plot the column 'Fare' for each value of 'Survived' on the same plot.

# In[ ]:


df_train.groupby('Survived').Fare.hist(alpha=0.6)


# **Take-away**: It looks as though those that paid more had a higher chance of surviving.
# 
# * Use seaborn to plot a histogram of the 'Age' column of df_train. You'll need to drop null values before doing so.

# In[ ]:


df_train_drop_na = df_train.dropna()
sns.distplot(df_train_drop_na.Age, kde=False)


# Plot a strip plot & a swarm plot of 'Fare' with 'Survived' on the x-axis.

# In[ ]:


sns.stripplot(x='Survived', y='Fare', data=df_train, alpha=0.3, jitter=True)


# In[ ]:


sns.swarmplot(x='Survived', y='Fare', data=df_train)


# **Take-away**: Fare definitely seems to be correlated with survival aboard the Titanic.
# 
# * Use the DataFrame method .describe() to check out summary statistics of 'Fare' as a function of survival.

# In[ ]:


df_train.groupby('Survived').Fare.describe()


# In[ ]:


sns.lmplot(x='Age', y='Fare', hue='Survived', data=df_train, fit_reg=False, scatter_kws={'alpha': 0.5})


# **Take-away**: It looks like those who survived either paid quite a bit for their ticket or they were young.
# 
# Use seaborn to create a pairplot of df_train, colored by 'Survived'. A pairplot is a great way to display most of the information that you have already discovered in a single grid of plots.

# In[ ]:


sns.pairplot(df_train_drop_na, hue='Survived')


# ## Build a machine learning model

# Below, you will drop the target 'Survived' from the training dataset and create a new DataFrame data that consists of training and test sets combined. You do this because you want to preprocess the data a little bit and make sure that any operations that you perform on the training set are also being done on the test data set.
# 
# But first, you'll store the target variable of the training data for safe keeping.

# In[ ]:


# Remove earlier predictions in the test set
df_test = df_test.drop(['Survived'], axis=1)

# Save training set predictions
survived_train = df_train.Survived

# Concatenate training and testing sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test], sort=True)


# In[ ]:


data.info()


# Note that we have the following missing values.
# 
# * Numerical columns:  **Age** and **Fare**
# * Non-numerical columns: **Cabin** and **Embarked**
# 
# Let us start dealing with missing values in the numerical columns. We shall fill in the missing values for the 'Age' and 'Fare' columns, using the median of the of these variables where you know them.
# 
# **Note** that in this case, you use the median because it's perfect for dealing with outliers. In other words, the median is useful to use when the distribution of data is skewed. Other ways to impute the missing values would be to use the mean, which you can find by adding all data points and dividing by the number of data points, or mode, which is the number that occurs the highest number of times.

# In[ ]:


data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

data.info()


# 

# Let us work on the non-numerical columns and convert it to numerical data. We should do this because, machine learning models work on input features that are numerical. 

# In[ ]:


data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data.head()


# Now, let us select the columns ['Sex_male',  'Fare',  'Age', 'Pclass',  'SibSp'] from the DataFrame to build our first machine learning model:

# In[ ]:


data = data[['Sex_male',  'Fare',  'Age', 'Pclass',  'SibSp']]
data.head()


# In[ ]:


data.info()


# ## Build a Decision Tree Classifier
# 
# Before fitting a model to the data, let us split it back into training and test sets:

# In[ ]:


data_train = data.iloc[:891]
data_test = data.iloc[891:]


# We'll use scikit-learn, which requires our data as arrays, not DataFrames so transform them:

# In[ ]:


X = data_train.values
test = data_test.values
y = survived_train.values


# Let us build a Decision Tree Classifier. First create a model with **max_depth=3** and then fit it to the data. Note that the model is named clf, which is short for "Classifier".

# In[ ]:


# Instantiate model and fit to data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)


# The output tells us all about the DecisionTreeClassifier we built. Except for **max_depth**, the rest are default values. Now, let us make predictions on the test set, new column 'Survived' is created to store predictions in it. Save 'PassengerId' and 'Survived' columns of df_test to a .csv and submit to Kaggle.

# In[ ]:


Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred

#Submit results to Kaggle
my_submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': df_test['Survived']})
my_submission.to_csv('submission.csv', index=False)


# The accuracy should be **77.9%**. Now, let us see what the **max_depth** argument was, why we chose it and explore train_test_split. 
# 
# ## Why Choose max_depth=3 ?
# The depth of the tree is known as a hyperparameter, which means a parameter you need to decide before you fit the model to the data. If you choose a larger max_depth, you'll get a more complex decision boundary.
# 
# If your decision boundary is too complex, you can overfit to the data, which means that your model will be describing noise as well as signal.
# 
# If your max_depth is too small, you might be underfitting the data, meaning that your model doesn't contain enough of the signal.
# 
# But how do you tell whether you're overfitting or underfitting?
# 
# Note: this is also referred to as the bias-variance trade-off; you won't go into details on that here, but we just mention it to be complete!
# 
# One way is to hold out a test set from your training data. You can then fit the model to your training data, make predictions on your test set and see how well your prediction does on the test set.
# 
# 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y)


# Now, let us iterate over values of **max_depth** ranging from **1 to 9** and plot the accuracy of the models on training and test sets:

# In[ ]:


# Setup arrays to store train and test accuracies
dep = np.arange(1, 9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

# Loop over differrent values of k (depth)
for i, k in enumerate(dep):
    # Setup a Decision Tree Classifier
    clf = tree.DecisionTreeClassifier(max_depth=k)
    
    # Fit the classifier to the training data
    clf.fit(X_train, y_train)
    
    # Compute accuracy on the training set
    train_accuracy[i] = clf.score(X_train, y_train)
    
    # Compute accuracy on the testing set
    test_accuracy[i] = clf.score(X_test, y_test)
    
# Generate plot
plt.title('clf: Varying depth of tree')
plt.plot(dep, test_accuracy, label='Testing accuracy')
plt.plot(dep, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()


# As we increase the *max_depth*, we are going to fit better and better to the training data because we'll make decisions that describe the training data. The accuracy for the training data will go up and up, but see that this doesn't happen for the test data: we're overfitting.
# 
# So that's why we chose **max_depth=3**.

# ## Feature Engineering
# 
# In this section, we will learn how feature engineering can help us to up our game when building machine learning models in Kaggle: create new columns, transform variables and more!
# 
# Feature engineering is a process where we use domain knowledge of our data to create additional relevant features that increase the predictive power of the learning algorithm and make our machine learning models perform even better.
# 

# ### Why Feature Engineer At All?
# We perform feature engineering to extract more information from our data, so that we can up our game when building models.
# 
# To begin with, let us import fresh set of data and concatenate the training and testing sets:

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test], sort=True)

data.info()


# #### Titanic's Passenger Titles
# Let's check out what this is all about by looking at an example. Let's check out the 'Name' column with the help of the .tail() method, which helps us to see the last five rows of your data:

# In[ ]:


data.Name.tail()


# Suddenly, you see different titles emerging! In other words, this column contains strings or text that contain titles, such as 'Mr', 'Master' and 'Dona'.
# 
# These titles of course give you information on social status, profession, etc., which in the end could tell you something more about survival.
# 
# At first sight, it might seem like a difficult task to separate the names from the titles, but don't panic! Remember, you can easily use regular expressions to extract the title and store it in a new column 'Title':

# In[ ]:


# Extract Title from Name, store in column and plot barplot

data['Title'] =data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x='Title', data=data)
plt.xticks(rotation=45)


# **Note** that this new column 'Title' is actually a new feature for our data set!
# 
# For example, we probably want to replace 'Mlle' and 'Ms' with 'Miss' and 'Mme' by 'Mrs', as these are French titles and ideally, we want all our data to be in one language. Next, we also take a bunch of titles that we can't immediately categorize and put them in a bucket called 'Special'.
# 
# **Tip**: play around with this to see how your algorithm performs as a function of it!
# 
# Next, we view a barplot of the result with the help of the .countplot() method:

# In[ ]:


data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Mrs'})
data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                       'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'], 'Special')


# In[ ]:


sns.countplot('Title', data=data)
plt.xticks(rotation=45)


# This is what our newly engineered feature 'Title' looks like!
# 
# Now, let us make sure that we have a 'Title' column and check out your data again with the .tail() method:

# In[ ]:


data.tail()


# #### Passenger's Cabins
# 
# When we loaded in the data and inspected it, we saw that there are several NaNs or missing values in the 'Cabin' column.
# 
# It is reasonable to presume that those NaNs didn't have a cabin, which could tell us something about 'Survival'. So, let's now create a new column 'Has_Cabin' that encodes this information and tells us whether passengers had a cabin or not.
# 
# Note that we use the .isnull() method in the code chunk below, which will return True if the passenger doesn't have a cabin and False if that's not the case. However, since we want to store the result in a column 'Has_Cabin', we actually want to flip the result: we want to return True if the passenger has a cabin. That's why you use the tilde ~.

# In[ ]:


data['Has_cabin'] = ~data.Cabin.isnull()

# View head of data
data.head()


# What we want to do now is drop a bunch of columns that contain no more useful information (or that we're not sure what to do with). In this case, we're looking at columns such as *['Cabin', 'Name', 'PassengerId', 'Ticket']*, because
# 
# * We already extracted information on whether or not the passenger had a cabin in your newly added 'Has_Cabin' column;
# * Also, we already extracted the titles from the 'Name' column;
# * We also drop the 'PassengerId' and the 'Ticket' columns because these will probably not tell you anything more about the survival of the Titanic passengers.
# 
# **Tip** there might be more information in the 'Cabin' column, but for this tutorial, you assume that there isn't!
# 
# To drop these columns in our actual data DataFrame, make sure to use the inplace argument in the .drop() method and set it to True:

# In[ ]:


data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
data.head()


# We've now **successfully engineered some new features** such as 'Title' and 'Has_Cabin' and made sure that features that don't add any more useful information for our machine learning model are now dropped from our DataFrame!
# 
# Next, we should to deal with deal with missing values, bin our numerical data, and transform all features into numeric variables using .get_dummies() again. Lastly, we'll build our final model for this tutorial. 
# 
# #### Handling Missing Values
# With all of the changes we have made to our original data DataFrame, it's a good idea to figure out if there are any missing values left with .info():

# In[ ]:


data.info()


# The result of the above line of code tells us that we have missing values in 'Age', 'Fare', and 'Embarked'.
# 
# In this case, we see that 'Age' has 1046 non-null values, so that means that we have 263 missing values. Similarly, 'Fare' only has one missing value and 'Embarked' has two missing values.
# 
# Let us impute these missing values with the help of .fillna():
# 
# Note that, once again, you use the median to fill in the 'Age' and 'Fare' columns because it's perfect for dealing with outliers. Other ways to impute missing values would be to use the mean, which you can find by adding all data points and dividing by the number of data points, or mode, which is the number that occurs the highest number of times.
# 
# You fill in the two missing values in the 'Embarked' column with 'S', which stands for Southampton, because this value is the most common one out of all the values that you find in this column.
# 
# **Tip**: you can double check this by doing some more Exploratory Data Analysis!

# In[ ]:


# Impute missing values for Age, Fare, Embarked
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())
data['Embarked'] = data.Embarked.fillna('S')

data.info()


# In[ ]:


data.head()


# #### Bin numerical data
# 
# Next, we should bin the numerical data, because we have a range of ages and fares. However, there might be fluctuations in those numbers that don't reflect patterns in the data, which might be noise. That's why we'll put people that are within a certain range of age or fare in the same bin. We can do this by using the pandas function *qcut()* to bin our numerical data:

# In[ ]:


# Binning numerical columns
data['CatAge'] = pd.qcut(data.Age, q=4, labels=False)
data['CatFare'] = pd.qcut(data.Fare, q=4, labels=False)
data.head()


# In[ ]:


data.drop(['Age', 'Fare'], inplace=True, axis=1)
data.head()


# ## Number of Members in Family Onboard
# 
# The next thingto do is create a new column, which is the number of members in families that were onboard of the Titanic.
# After the creation of this column, we will go ahead and drop the 'SibSp' and 'Parch' columns from your DataFrame:

# In[ ]:


# Create column of number of Family members onboard (Optional)
# data['Fam_size'] = data.Parch + data.SibSp
data.drop(['Parch', 'SibSp'], inplace=True, axis=1)
data.head()


# ## Transform Variables into Numerical Variables
# Now that we have engineered some more features, such as 'Title' and 'Has_Cabin', and have dealt with missing values, binned numerical data, it's time to transform all variables into numeric ones. We do this because machine learning models generally take numeric input.
# 
# As we have done previously, you will use .get_dummies() to do this:

# In[ ]:


# Transform into binary variables
data_dum = pd.get_dummies(data, drop_first=True)
data_dum.head()


# With all of this done, it's time to build our final model!
# 
# ## Building models with the New Data Set!
# 
# As before, we'll first split the data back into training and test sets. Then, we'll transform them into arrays:

# In[ ]:


# Split into test and train sets
data_train = data_dum.iloc[:891]
data_test = data_dum.iloc[891:]

# Transform into arrays for scikit-learn
X = data_train.values
test = data_test.values
y = survived_train.values


# #### Grid Search
# 
# For our dataset, we begin the grid search by splitting the dataset into 5 groups or folds. Then we hold out the first fold as a test set, fit model on the remaining four folds, predict on the test set and compute the metric of interest. Next, hold out the second fold as your test set, fit on the remaining data, predict on the test set and compute the metric of interest. Then similarly with the third, fourth and fifth.
# 
# As a result, we get five values of accuracy, from which you can compute statistics of interest, such as the median and/or mean and 95% confidence intervals.
# 
# You do this for each value of each hyperparameter that you're tuning and choose the set of hyperparameters that performs the best. This is called grid search.
# 
# Let us now use cross validation and grid search to choose the best max_depth for the new feature-engineered dataset:

# In[ ]:


# Set up the hyperparameter grid
dep = np.arange(1, 9)
param_grid = {'max_depth': dep}

# Instantiate a decision tree classifier: clf
clf = tree.DecisionTreeClassifier()

# Instantiate the GridSearchCV object: clf_cv
clf_cv = GridSearchCV(clf, param_grid=param_grid, cv=5)

# Fit the data
clf_cv.fit(X, y)

# Print the tuned parameter and score
print("Tuned Decision Tree Parameters: {}".format(clf_cv.best_params_))
print("Best score is: {}".format(clf_cv.best_score_))


# Let us make predictions on our test set, create a new column 'Survived' and store predictions in it. Save the 'PassengerId' and 'Survived' columns of df_test to a .csv and submit it to Kaggle!

# In[ ]:


Y_pred = clf_cv.predict(test)


# In[ ]:


# Submit to Kaggle 
df_test['Survived'] = Y_pred
my_submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': df_test['Survived']})
my_submission.to_csv('submission.csv', index=False)


# The accuracy score will be **78.9%**
# 
# ## Next steps
# * Try more feature engineering and try some new models out to improve on this score. 
# * Try out feature scaling

# ## References
# 
# [Datacamp - visualization](https://www.datacamp.com/community/tutorials/kaggle-machine-learning-eda)
# 
# [Datacamp - Decision tree](https://www.datacamp.com/community/tutorials/kaggle-tutorial-machine-learning)
# 
# [Datacamp - Feature Engineering](https://www.datacamp.com/community/tutorials/feature-engineering-kaggle)
# 
# [Datacamp - Github](https://github.com/datacamp/datacamp_facebook_live_titanic)
