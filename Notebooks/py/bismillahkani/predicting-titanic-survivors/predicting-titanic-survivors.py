#!/usr/bin/env python
# coding: utf-8

# **Problem:**
# The goal is to predict if a passenger of Titanic wil survivie or not. This a typical Binary classification problem and we will try to solve it using maching learning tools such as Decision Tree. 
# 
# The first few sections of the code is entirely based on https://www.datacamp.com/community/tutorials/kaggle-tutorial-machine-learning
# 
# However, I tried to consoldate the three step approach into a single flow
# 
# In the later sections, I implement other classification models such as Logistic Regression, Random Forest, KNN, etc. 

# **Import all necessary libraries**

# In[ ]:


# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Figures inline and set visualization style
get_ipython().magic(u'matplotlib inline')
sns.set()


# **Import the data and have a look at what it has**

# In[ ]:


# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# View first few lines of training data
df_train.head()


# In[ ]:


# You can also view the data types and missing data
df_train.info()


# As shown above the training data consists of 12 columns. The target variable that we want to predict is Survived. The remaining variables are called as predictor variables. In this case, you see that there are only 714 non-null values for the 'Age' column in a DataFrame with 891 rows. This means that are are 177 null or missing values. We will see later how to deal with missing values.

# In[ ]:


# you can also see the statistical summary of the training data
df_train.describe()


# **Visual Exploratory Data Analysis (EDA)**

# In[ ]:


sns.countplot(x='Survived', data=df_train);


# In[ ]:


sns.countplot(x='Sex', data=df_train);


# In[ ]:


sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train);


# It looks like females are more most likely to survive than male. With this we can use Pandas to calculate how many male and female survived. 

# In[ ]:


df_train.groupby(['Sex']).Survived.sum()


# In[ ]:


# Use pandas to figure out the proportion of women that survived, along with the proportion of men
print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print(df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())


# We see that 74% of female and 18% of male survived the Titanic disaster. 

# In[ ]:


# Use seaborn to build bar plots of the Titanic dataset feature 'Survived' split (faceted) over the feature 'Pclass'
sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train);


# It looks like passengers that travelled in first class were more likely to survive. On the other hand, passengers travelling in third class were more unlikely to survive. 

# In[ ]:


# Use seaborn to build bar plots of the Titanic dataset feature 'Survived' split (faceted) over the feature 'Embarked'
sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train);


# It looks like passengers that embarked in Southampton were less likely to survive. 

# In[ ]:


# Use seaborn to plot a histogram of the 'Fare' column of df_train
sns.distplot(df_train.Fare, kde=False);


# It looks like most passengers paid less than 100 for travelling with the Titanic.

# In[ ]:


# Use a pandas plotting method to plot the column 'Fare' for each value of 'Survived' on the same plot.
df_train.groupby('Survived').Fare.hist(alpha=0.6);


# It looks as though those that paid more had a higher chance of surviving.

# In[ ]:


# Use seaborn to plot a histogram of the 'Age' column of df_train. You'll need to drop null values before doing so
df_train_drop = df_train.dropna()
sns.distplot(df_train_drop.Age, kde=False);


# In[ ]:


# Plot a strip plot & a swarm plot of 'Fare' with 'Survived' on the x-axis
sns.stripplot(x='Survived', y='Fare', data=df_train, alpha=0.3, jitter=True);


# In[ ]:


sns.swarmplot(x='Survived', y='Fare', data=df_train);


# It looks like fare is correlated with survival aboard the Titanic.

# In[ ]:


# Use the DataFrame method .describe() to check out summary statistics of 'Fare' as a function of survival
df_train.groupby('Survived').Fare.describe()


# In[ ]:


# Use seaborn to plot a scatter plot of 'Age' against 'Fare', colored by 'Survived'
sns.lmplot(x='Age', y='Fare', hue='Survived', data=df_train, fit_reg=False, scatter_kws={'alpha':0.5});


# It looks like those who survived either paid quite a bit for their ticket or they were young.

# In[ ]:


# Use seaborn to create a pairplot of df_train, colored by 'Survived'
sns.pairplot(df_train_drop, hue='Survived');


# **Your first machine Learning Model**

# In[ ]:


# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])

# Check out your new DataFrame data using the info() method
data.info()


# There are two numerical variables that have missing values namely 'Age' and 'Fare' columns. In 'Age' column there are only 1046 non-null values for the total of 1309 entries of the dataframe which says that there are 263 missing values. In 'Fare' column there is only 1 missing value. 
# The missing values of 'Age' and 'Fare' column can be imputed using the median values of the variable. Median is a suitable value for imputing as it is less likely to be affected by outliers in the data. Usually it is a good practice to fill the missing numerical values by median. 

# In[ ]:


# Impute missing numerical variables
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

# Check out info of data
data.info()


# In[ ]:


# Encode the data with numbers because most machine learning models might require numerical inputs
# yo can do this using Pandas function get_dummies() which converts the categorical variable into numerical
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data.head()


# get_dummies() creates a new columns for each of the options in 'Sex' so that it creates a new columns for female called 'Sex_female' and new columns for male called 'Sex_male' which encodes if that row was male or female. As drop_first argument in get_dummies() was set as true 'Sex_female' columns was dropped. 
# 
# Now you will select the 'Sex_male', 'Fare' 'Age'. 'Pclass', 'SibSp' columns from your dataframe to build your first machine learning model. 

# In[ ]:


# Select columns and view head
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
data.head()


# In[ ]:


data.info()


# **Build a Decision Tree Classifier**
# What is a decision tree classifier? It is a tree that allows you to classify data points, which are also known as target variables, based on feature variables. 
# For example, this tree below has a root node that forces you to make a first decision, based on the following question: "Was 'Sex_male'" less than 0.5? In other words, was the data point a female. If the answer to this question is True, you can go down to the left and you get 'Survived'. If False, you go down the right and you get 'Dead'.

# In[ ]:


# Before fitting a model to your data, split it back into training and test sets
data_train = data.iloc[:891]
data_test = data.iloc[891:]
# A Scikit requirement transform the dataframes to arrays
X = data_train.values
test = data_test.values
y = survived_train.values


# In[ ]:





# In[ ]:


# build your decision tree classifier with max_depth=3 and then fit it your data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)


# Now, you'll make predictions on your test set, create a new column 'Survived' and store your predictions in it. Save 'PassengerId' and 'Survived' columns of df_test to a .csv and submit to Kaggle.

# In[ ]:


# Make predictions and store in 'Survived' column of df_test
Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('1st_dec_tree.csv', index=False)


# The accuracy of this model as reported by Kaggle is 78%. Congratulations on the first machine learning model!

# **Feature Engineering**
# 
# You perform feature engineering to extract more information from your data, so that you can up your game when building models.
# 
# **Titanic's Passenger Titles**
# This name column contains strings or text that contain titles, such as 'Mr', 'Master' and 'Dona'. 
# These titles of course give you information on social status, profession, etc., which in the end could tell you something more about survival. 
# At first sight, it might seem like a difficult task to separate the names from the titles, but don't panic! Remember, you can easily use regular expressions to extract the title and store it in a new column 'Title' which will be our new feature of the dataset. 

# In[ ]:


# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])

# Extract Title from Name, store in column and plot barplot
data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);


# As you can see that there are several titles in the above plot and there are many that don't occur so often. So, it makes sense to put them in fewer buckets. For example, you probably want to replace 'Mlle' and 'Ms' with 'Miss' and 'Mme' by 'Mrs', as these are French titles and ideally, you want all your data to be in one language. Next, you also take a bunch of titles that you can't immediately categorize and put them in a bucket called 'Special'. Next, you view a barplot of the result with the help of the .countplot() method

# In[ ]:


data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);


# **Passenger's Cabins** When you loaded in the data and inspected it, you saw that there are several NaNs or missing values in the 'Cabin' column. It is reasonable to presume that those NaNs didn't have a cabin, which could tell you something about 'Survival'. So, let's now create a new column 'Has_Cabin' that encodes this information and tells you whether passengers had a cabin or not.

# In[ ]:


# Did they have a Cabin?
data['Has_Cabin'] = ~data.Cabin.isnull()

# Drop columns and view head
data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
data.head()


# In[ ]:


# Impute missing values for Age, Fare, Embarked
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())
data['Embarked'] = data['Embarked'].fillna('S')
data.info()


# **Bin numerical data**  Next, you want to bin the numerical data, because you have a range of ages and fares. However, there might be fluctuations in those numbers that don't reflect patterns in the data, which might be noise. That's why you'll put people that are within a certain range of age or fare in the same bin. You can do this by using the pandas function qcut() to bin your numerical data

# In[ ]:


# Binning numerical columns
data['CatAge'] = pd.qcut(data.Age, q=4, labels=False )
data['CatFare']= pd.qcut(data.Fare, q=4, labels=False)
data.head()


# In[ ]:


data = data.drop(['Age', 'Fare','SibSp','Parch'], axis=1)
data.head()


# In[ ]:


# Transform into binary variables
data_dum = pd.get_dummies(data, drop_first=True)
data_dum.head()


# **Building models with Your New Data Set** 
# We will use the same Decision Tree classifer wit this new data set. This time, we will use GridSearch with CrossValidation to determine the hyperparameter max_depth of decision tree classifier. 

# In[ ]:


# Split into test.train
data_train = data_dum.iloc[:891]
data_test = data_dum.iloc[891:]

# Transform into arrays for scikit-learn
X = data_train.values
test = data_test.values
y = survived_train.values

# Setup the hyperparameter grid
dep = np.arange(1,9)
param_grid = {'max_depth' : dep}

# Instantiate a decision tree classifier: clf
clf = tree.DecisionTreeClassifier()

# Instantiate the GridSearchCV object: clf_cv
clf_cv = GridSearchCV(clf, param_grid=param_grid, cv=5)

# Fit it to the data
clf_cv.fit(X, y)

# Print the tuned parameter and score
print("Tuned Decision Tree Parameters: {}".format(clf_cv.best_params_))
print("Best score is {}".format(clf_cv.best_score_))


# In[ ]:


# Now, you can make predictions on your test set, create a new column 'Survived' and store your predictions in it
Y_pred = clf_cv.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('dec_tree_feat_eng.csv', index=False)


# The accuracy of this model as reported by Kaggle is 78.9%. 

# **Other classification models implemented by me. **
# 
# In the following sections, I implement other classification models. The models will be trained on the feature engineered new dataset. We will use the models straight away with default parameters and also do hyperparameter tuning using GridSearchCV. 

# **Logistic Regression**
# The first model to try is Logistic Regression. 
# For more detials about Logistic Regression http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

# In[ ]:


# import logisitc regression from sklearn
from sklearn.linear_model import LogisticRegression

#instantiate the classifier without any parameters
logreg = LogisticRegression()

#fit the data to the classifier
logreg.fit(X,y)

#predict the survivors and submit the results
Y_pred = logreg.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('log_reg_feat_eng.csv', index=False)


# The accuracy of logistic regression model as reported by Kaggle is 77.5% which is less than Decision Tree classifier. 
# 
# We will fine tuner the hyperparameters of Logistic Regression using GridSearchCV. 

# In[ ]:


# create parameter grid for hyperparameter tuning
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg,param_grid,cv=5)

# Fit it to the training data
logreg_cv.fit(X,y)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

#predict the survivors and submit the results
Y_pred = logreg_cv.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('log_reg__feat_eng.csv', index=False)


# The accuracy of logisitic regression with tuned parameters of C and penalty using GridSearch as reported by Kaggle is 79.425% which is better then Decision Tree. 
# 
# **Random Forest Classifier**
# 
# We will try RandomForest classifier. For more details, http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

# In[ ]:


#import random forest classifer
from sklearn.ensemble import RandomForestClassifier

#instantiate RandomForest
rf_clf = RandomForestClassifier()

#fit the data
rf_clf.fit(X,y)

#predict the survivors and submit the results
Y_pred = rf_clf.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('random_forest_feat_eng.csv', index=False)


# The accuracy of Random Forest classifier as reported by Kaggle is 76.5% and not an improvement from the best results so far. We will use hyperparameter tuning of RandomForest using GridSearch. The n_estimators in the hyperparameter for RandomForestClassifier.

# In[ ]:


# create parameter grid for hyperparameter tuning
n_estimators = np.arange(10,50)
params_grid = {'n_estimators':n_estimators}

#instantiate RandomForest
rf_clf = RandomForestClassifier()

# Instantiate the GridSearchCV object
rf_clf_cv = GridSearchCV(rf_clf,params_grid,cv=5)

# Fit it to the training data
rf_clf_cv.fit(X,y)

# Print the optimal parameters and best score
print("Tuned Random Forest Classifier Parameter: {}".format(rf_clf_cv.best_params_))
print("Tuned Random Forest Classifier Accuracy: {}".format(rf_clf_cv.best_score_))

#predict the survivors and submit the results
Y_pred = rf_clf_cv.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('random_forest_cv_feat_eng.csv', index=False)


# The accuracy of Random Forest Classifier with n_estiamtors tunded as reported by Kaggle is 77.511% which is not an improvement.

# **KNN Classifier**
# We will use KNN - K Nearest Neigbor classifier to predict Titanic survivors. For more details, http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

# In[ ]:


#import KNN classifier from sklearn
from sklearn.neighbors import KNeighborsClassifier

#Instantiate KNN classifier
knn = KNeighborsClassifier()

#Fit training data to knn
knn.fit(X,y)

#predict the survivors and submit the results
Y_pred = knn.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('knn_feat_eng.csv', index=False)


# The accuracy of KNN classifier as reported by Kaggle is 77.033% which is not an improvement. 
# 
# We will tune the hyperparameter n_neighbors using GridSearchCV.

# In[ ]:


# create parameter grid for hyperparameter tuning
n_neighbors = np.arange(1,20)
params_grid = {'n_neighbors':n_neighbors}

#instantiate KNN
knn = KNeighborsClassifier()

# Instantiate the GridSearchCV object
knn_cv = GridSearchCV(knn,params_grid,cv=5)

# Fit it to the training data
knn_cv.fit(X,y)

# Print the optimal parameters and best score
print("Tuned KNN Classifier Parameter: {}".format(knn_cv.best_params_))
print("Tuned KNN Classifier Accuracy: {}".format(knn_cv.best_score_))

#predict the survivors and submit the results
Y_pred = knn_cv.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('knn_cv_feat_eng.csv', index=False)


# The accuracy of KNN classifier after hyperparameter tuning is 75.69% which is not an improvement. 
