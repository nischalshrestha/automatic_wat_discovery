#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This notebook serves two purposes: To help me hone my skills as an amateur data scientist, and to help beginners learn the basics of building an accurate machine learning model.  
# 
# As I understand it, there is a process to building an accurate machine learning model:
# >1. Exploratory data analysis
# 2. Cleaning/Parsing Data
# 3. Building the algorithm/Evaluation
# 
# I hope that others comment and suggest ways for me to improve on my skills, and I will post what I have updated. 
# 
# 

# In[ ]:


# Import necessary libraries and data files
import numpy as np 
import pandas as pd 

training_df = pd.read_csv(r"../input/train.csv")
testing_df = pd.read_csv(r"../input/test.csv")


# # Exploratory Data Analysis (EDA)
# Exploratory data analysis is very important since getting familiar with the data leads to seeing what steps you need to take to
# clean up the data. Also this helps decide on what algorithms to use.
# 
# I used to treat machine learning like a black box. I would just plug in data into my algorithm of choice and wonder why my algorithm score wouldn't improve. Exploratory data analysis is a must since you can't expect good results if you can't understand the inputs.

# In[ ]:


# Get some desciptive information on the data
training_df.info() # Noticed so many nulls in the age and cabin fields


# In[ ]:


#Descriptive statistics on the data set
training_df.describe()


# In[ ]:


# To see what the data set looks like
training_df.head() # Name, ticket, and cabin do not seem to be very useful fields


# In[ ]:


# Univariate/Bivariate data analysis
import seaborn as sns
from matplotlib import pyplot
sns.set_style("ticks")

# Investigating Survival by Gender
sns.factorplot(data=training_df, x="Sex", 
                   col="Survived", kind="count", 
                   size=5, palette="deep")


# In[ ]:


# Comparing survival among pclass
sns.factorplot(data=training_df, x="Pclass", 
                   col="Survived", kind="count", 
                   size=5, palette="deep")


# In[ ]:


# Survival among embarkment
sns.factorplot(data=training_df, x="Embarked", 
                   col="Survived", kind="count", 
                   size=5, palette="deep")


# In[ ]:


# Survival with age
sns.factorplot(data=training_df, x="Survived", y="Age",
              kind="box", size=5)


# In[ ]:


# Sibling/Spouse
sns.factorplot(data=training_df, x="Survived", y="SibSp",
              kind="box", size=5)


# In[ ]:


# Parent/child
sns.factorplot(data=training_df, x="Survived", y="Parch",
              kind="box", size=5)


# In[ ]:


# Fare
sns.factorplot(data=training_df, x="Survived", y="Fare",
              kind="box", size=5)


# # Data Parsing/Cleaning <a name="parsing"></a>
# * Dropping useless columns
# * Filling in NaN values with the most occuring value
# * Make new columns for categorical data (One hot encoding)

# In[ ]:


# Must drop useless columns 
def drop_useless_cols(df):
    return df.drop(["Name", "Ticket", "Cabin"], axis=1)

training_df = drop_useless_cols(training_df)
testing_df = drop_useless_cols(testing_df)

training_df.head()


# ## Engineered Features
# 
# I believe that having any family on board could affect survival rate on the titanic. I wish there were a way to separate the children count from the parent count, since I feel that the more siblings you have, the less likely your are to survive. 
# 
# Maybe having any kind of family on board would contribute to a higher chance of survival, so I created a family feature, which consists of the parch feature plus the sibsp feature.

# In[ ]:


# def family_feature(df):
#     df["Family"] = df["Parch"] - df["SibSp"]
#     return df.drop(["Parch","SibSp"], axis=1)

# training_df = family_feature(training_df)
# testing_df = family_feature(testing_df)


# In[ ]:


training_df.head()


# ## Create dummy columns for the Pclass variable

# In[ ]:


# Create new columns for the Pclass categories
# Pclass is categorical data and should have separate columns

def create_pclass_cols(df):
    # Creates new column for each pclass
    df["pclass_1"] = df["Pclass"].apply(lambda x : 1 if x == 1 else 0) 
    df["pclass_2"] = df["Pclass"].apply(lambda x : 1 if x == 1 else 0)
    df["pclass_3"] = df["Pclass"].apply(lambda x : 1 if x == 1 else 0)
    
    return df.drop("Pclass", axis=1)

training_df = create_pclass_cols(training_df)
testing_df = create_pclass_cols(testing_df)

training_df.head()


# ## One Hot Encoding
# There are two NaNs in the Embarked column that must be filled before getting the dummy variables

# In[ ]:


training_df["Embarked"].unique()


# In[ ]:


training_df["Embarked"].mode()[0]


# In[ ]:


def one_hot_encoding(df):
    #Fill the embarked column with most occuring value
    #since imputer doesnt work well with categorical data
    fill_value = df["Embarked"].mode()[0]
    df["Embarked"].fillna(fill_value, inplace=True)
    
    return pd.get_dummies(df)

training_df = one_hot_encoding(training_df)
testing_df = one_hot_encoding(testing_df)


# In[ ]:


training_df.head()


#  # Machine Learning
#  
#  ## Built with Pipeline
#  
#  >### Imputer
#  In the case of missing values (NaNs) I will fill them in with the most frequent value of that column by using sklearn's Imputer.
#  
#  >### Feature Selection
#  To avoid overfitting on noisy data, I will use SelectPercentile to choose the most useful features
#  
#  >### Classifier (Random Forest) 
# 
# >Now that all the data is parsed, I will go ahead and make a classifier using a random forest. First I will need to split up the training data set into two parts: training set and testing set

# In[ ]:


# Split features from labels
# I also get rid of the PassengerId column since it does not provide any useful data
features = training_df.drop(["PassengerId","Survived"], axis=1).values
labels = training_df["Survived"].values

# Split the dataset in two equal parts
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=0)


# In[ ]:


# Import necessary items to set up pipeline
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipeline = Pipeline([("imputer", Imputer(strategy="most_frequent")),
                    ("selection", SelectPercentile()),
                    ("scaler", MinMaxScaler(feature_range=(-1,1))),
                    #("svm", SVC()),
                    ("rf", RandomForestClassifier())
                    ])

parameters = {"selection__percentile":range(10,100,10),
                 "rf__n_estimators":range(100,200,20),
                 "rf__max_features":["sqrt", "log2", "auto"]
             }

# svm_parameters = {"selection__percentile":range(10,100,10),
#                   "svm__C":[1,10,100,1000,10000],
#                   "svm__gamma":[0,0.1,0.001,0.0001,0.000001,"auto"]
#              }

grid = GridSearchCV(pipeline, parameters, cv=5)

grid.fit(X_train,y_train)

print("The best params are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


# In[ ]:


# Validate the score on the test set
grid.score(X_test, y_test)


# # Create Predictions and Upload Submission

# In[ ]:


predictions = grid.predict(testing_df.drop(['PassengerId'], axis=1))

submission = pd.DataFrame({
        "PassengerId": testing_df["PassengerId"],
        "Survived": predictions
    })

submission.to_csv('prediction_submission.csv', index=False)


# In[ ]:


submission.head()

