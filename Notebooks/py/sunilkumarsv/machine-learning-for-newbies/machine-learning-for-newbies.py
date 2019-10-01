#!/usr/bin/env python
# coding: utf-8

# ## Let's learn "Machine Learning"!! ##
# In this notebook, let's cover the basic process of how to make sense out of given data. We have been given a raw data dump and it's upto us to make sense out of the data and come out with a prediction for the question, " what sorts of people were likely to survive?". 
# 
# For solving any kind of problem using machine learning, there are few basic steps which need to be performed. These are:
# 
#  1. Import Data (From now on, let's call it a '**Data Frame**')
#  2. Visualizing the data : In this step, once we visualize the data, we can get an idea of in which direction we should head in order to get to the solution.
#  3. Cleaning up the data and transforming data : In almost all the datasets, we will be having some junk data. By junk, I mean, data which might not be useful for applying to our machine learning algorithm. And thus, it's important to remove this junk from our Data Frame. 
#  4. Encode Data : Ah, encoding is a much required process . This is required to make sure that categorical features have their own weight and thus making the regression model much accurate. (Would be great if you could spend some5-10 minutes time to learn about 'Data Encoding' over the internet). In simple terms, with this activity, we normalize the labels. Encoding coverts each unique string value into a number, so that the data is flexible for various algorithms.
#  5. Split the given data into Train set and Test Set : Train set is used to come up with an algorithm. And then, in order to test the accuracy of the algorith, we use Test Set to test our algorithm. 
#  6. Fitting and tuning the algorithms : Fitting an algorith to the Data Frame. Once you verify your algorithm by applying and checking it's accuracy on Test Set, we can fine tune our algorithm for improving the accuracy. Remember, do not overfit thoguh!
#  7. Cross Validate with KFold : Isn't it great to have a double check on the algorithm. KFold is a technicque which we can use to cross validate on how accurate our algorithm is. KFold basically creates a different buckets each time for Test Set. That means, we will have different Test Set bucket each time. And then, we run our algorithm on each of these buckets and test our algorithm. 
#  8. Predict : Afterall, we wanted to predict something, isn't it? :)
#  9. Publish and share your learning!

# Step 1 : Import Data!
# ---------------------
# 
# We use pandas library to import our data (train.csv, test.csv) into our dataframes named data_train & data_test respectively.
# 
# We use 'sample()' method of pandas to display a sample of the data loaded from the csv file.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')

data_train.sample(3)


# Step 2 : Visualizing the data
# -----------------------------
# 
# As we have finished with loading of the data into our dataframe, let's visualize the data to get some insights.
# Let's look at the ratio (percentage) of how many females survived to that of the males survivied as per each 'Embarked' column.
# 
# We can use 'barplot' provided by searborn for visualizing this. We need to mention the column name for X-Axis, Y-Asis, Hue (the column whic will be the factor for calculating the percentage) and of course we need to specify the data frame.

# In[ ]:


sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train);


# 'pointplot' is another plot provided by seaborn which gives us similar insights as the above plot, but with different visuals.
# 
# Let's use 'pointplot' to visualize the raio of number of females survived to the number of males survived according to the column 'Pclass'

# In[ ]:


sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"]);


# Step 3 : Cleaning up the data and transforming the data
# -------------------------------------------------------
# This step can also be looked as 'beutifying' the data or 'sensifying' the data (make the data more sensible to the reader.
# 
# As a first step, let's fill all 'NaN' values in 'Age' column with some invalid data, say '-0.5'. Once we do that, we can now group the data and name the data. For e.g, any person between the age group 0-5 can be termed as 'Baby', age group of 6-12 as 'Child'...etc.

# In[ ]:


def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

data_train = simplify_ages(data_train)
data_test = simplify_ages(data_test)
data_train.head()


# If you look at the 'Cabin' column closely, we can notice that each Cabin starts with a litter followed by an alphabet. It can be safe to assume that, the starting letter says something about the 'Type' of the cabin and thus, seems important for calculations. 
# With this assumption in mind, we now have to truncate the data in 'Cabin' column to only have the first letter instead of the whole term. And yeah, not to forget filling 'NaN' with a value of 'N' (need to be careful with this step though. There is a possibility that the 'Cabin' column has a value starting with 'N' like N47. In this dataset, we don't have that problem though.)

# In[ ]:


def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

data_train = simplify_cabins(data_train)
data_test = simplify_cabins(data_test)
data_train.head()


# Similar to the grouping we have done for 'Age', we can do for 'Fare' column as well. 
# 
# Grouping of 'Fare' can be challenging, because we don't now what kind of values we need to take as start and end point of each group. In order to get help on this, we can use 'describe()' function to get clarity on the data. 
# 
# After doing a describe, we can see that we can have a grouping from 0-8, 9-15, 16-31...etc

# In[ ]:


data_train.Fare.describe()


# In[ ]:


def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', 'Low', 'Medium', 'High', 'Expensive']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

data_train = simplify_fares(data_train)
data_test = simplify_fares(data_test)
data_train.head()


# I think now our Data Frame is in a good shape and we can not delete some un-necessary (again this is assumed) columns. Like Ticket, Name, Embarked. We are taking this concious decision to delete these columns, because it's quite clear that these factors might not have had an impact on someone's survival.
# 
# Isn't our Data Frame looking neat after this Step of 'Cleaning up the data and transforming the data'?

# In[ ]:


def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

data_train = drop_features(data_train)
data_test = drop_features(data_test)
data_train.head()


# We are civilised developers...aren't we? So, let's organize our code! :)

# In[ ]:


def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', 'Low', 'Medium', 'High', 'Expensive']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = drop_features(df)
    return df

data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
data_train = transform_features(data_train)
data_test = transform_features(data_test)
data_train.head()


# Let's now have a look at how a barplot looks when we look at the survival stats based on someone's age.

# In[ ]:


sns.barplot(x="Age", y="Survived", hue="Sex", data=data_train);


# Let's now have a look at how a barplot looks when we look at the survival stats based on someone's cabin type.

# In[ ]:


sns.barplot(x="Cabin", y="Survived", hue="Sex", data=data_train);


# Let's now have a look at how a barplot looks when we look at the survival stats based on someone's fare category

# In[ ]:


sns.barplot(x="Fare", y="Survived", hue="Sex", data=data_train);


# Step 4 : Encoding Data
# -----------------
# 
#  We use LabelEncoder to encode our data. Encoding converts each unique string to a number, thus making it easy for the machines to understand the data and thereby helping greatly towards building algorithm.

# In[ ]:


from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
data_train, data_test = encode_features(data_train, data_test)
data_train.head()


# ## Step 5 : Split the given data into Train set and Test Set  ##
# 
# In this step, we create Data Frames for Independant Variables and Dependent variables. And then, we will split the data into Train & Test sets.
# 
# 'PassengerId' columns also doesn't look like it has an impact on the survival and of course, 'Suvived' column is our Dependant variable. Thus, let's drop those columns from our list of the Data Frame of Independent Vaiables.
# And, create a Data Frame with just 'Survived' column for Dependent variable.

# In[ ]:


from sklearn.model_selection import train_test_split

X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']


# Now, let's create Training set and Test set. Let's keep 80% as Training set and 20% as test set. 
# We use 'train_test_split()' method to split our dataframe.

# In[ ]:


num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)


# ## Step 6 : Fitting and tuning the algorithms ##
# This is the core aspect of machine learning and you will get better in this step only with experience (Thus, play a lot with other Datasets available on Kaggle)
# 
# For this session, I will be going with RandomForestClassifier. You are free to choose a classifier of your choice (Naive Bayes, SVM)

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)


# In[ ]:


predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))


# ## Step 7 : Cross Validate with KFold ##
# Let's validate our model with KFold. 

# In[ ]:


from sklearn.cross_validation import KFold

def run_kfold(clf):
    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))     
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome)) 

run_kfold(clf)


# ## Step 8 : Predict ##
# Let's predict the outcome for our Test set

# In[ ]:


ids = data_test['PassengerId']
predictions = clf.predict(data_test.drop('PassengerId', axis=1))


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
# output.to_csv('titanic-predictions.csv', index = False)
output.head()


# ## Step 9 : Hurray...Rejoice and Publish!!! ##
