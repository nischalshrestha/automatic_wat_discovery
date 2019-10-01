#!/usr/bin/env python
# coding: utf-8

# The purpose of this kernel is to reinforce my understanding of using the logistic regression, k nearest neighbors and support vector machine to make predictions on a binary response. In addition to that, I'd like to get practice of completing an end to end machine learning project. This kernel contains the following sections:
# 1. Data Cleaning
# 2. Exploratory Data Analysis
# 3. Feature Engineering
# 4. Model training and selection using a training and validation data set.
# 5. Submitting predictions.

# In[ ]:


#Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # import seaborn
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import warnings
warnings.filterwarnings('ignore')


# # 1. Data Cleaning

# In[ ]:


titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')


# In[ ]:


titanic_train.head(5)


# ## Some observations about the head of the data frame
# * Looks like there are 11 features and one response variable "survived".
# * PassengerID could represent the index of the data frame
# * Pclass looks like it could be a categorical variable
# * Cabin appears to have some missing data.

# In[ ]:


titanic_train.info()


# Looks like the following columns have some missing values:
# 
# * Age
# * Cabin - which aligns with the previous observation
# * Embarked

# In[ ]:


titanic_test.info()


# The test data set had missing values for the following columns
# * Age
# * Cabin - which aligns with the previous observation
# * Fare

# In[ ]:


titanic_train.describe()


# In[ ]:


titanic_train.describe(include=['O'])


# A couple things here:
# * No repeated names
# * 3 different values for embarked

# # Data Cleaning
# In order to clean this dataset, I'd like to make sure that each column is free from NaN values and is of the correct type. As noted previously, the age, embarked, and cabin columns are all missing values.

# Let's take a look at the age column

# In[ ]:


print("Age broken down by P-class")
titanic_train.groupby('Pclass').mean()[['Age']]


# I'm going to impute the age column based on the average age per passenger determined by the Pclass column for both the training and testing data sets because both of these columns contain missing data and less than 25% of the column is missing data. 

# In[ ]:


titanic_train.loc[titanic_train.Age.isnull(), 'Age'] = titanic_train.groupby('Pclass')['Age'].transform('mean')
titanic_test.loc[titanic_test.Age.isnull(), 'Age'] = titanic_test.groupby('Pclass')['Age'].transform('mean')


# Check out rows 5 and 17 to ensure age of ~25 got inputed for age in row 5 and ~29 was inputted for age in row 17. Looks good, and checking .info() method there are no missing values for age column.

# In[ ]:


titanic_train.iloc[[5, 17]]


# Due to the large number of missing entires for the cabin column in both the training and testing dataset, I'm going to drop it from both.

# In[ ]:


titanic_train = titanic_train.drop('Cabin', axis=1)
titanic_test = titanic_test.drop('Cabin', axis=1)


# Also because Embarked is only missing two entries from the training dataset and fare is only missing one entry from the test dataset I'm just going to impute these values with the mode and median value for each column respectively.

# In[ ]:


titanic_train['Embarked'].fillna(titanic_train['Embarked'].mode()[0], inplace=True)
titanic_test['Fare'].fillna(titanic_test['Fare'].median(), inplace = True)


# Ensure all columns have no null values

# In[ ]:


print('Training Data Null Values')
print(titanic_train.isnull().sum())
print("-" * 30)
print('Test Data Null Values')
print(titanic_test.isnull().sum())


# Looks like all columns are cleaned

# ## Exploratory Data Analysis

# In[ ]:


titanic_train.head()


# Because the goal is to predict the Survived column I want to take a look at the class balance in that column

# In[ ]:


sns.countplot(x='Survived', data=titanic_train)


# There is a class imbalance meaning that more people did not survive the titanic than did survive it in our training dataset.

# Want to look at how the price of tickets bought varied by the age of the people on board.

# In[ ]:


sns.boxplot(x = 'Survived', y = 'Fare', data = titanic_train)


# In[ ]:


titanic_train.groupby('Survived').mean()[['Fare']]


# Looks like the median ticket price is larger for those who survived. Average ticket price is much higher but is likely due to the outlier. Want to investigate this outlier. Look below and see that three individuals purchased tickets at a fare of $512. Money must have not been a problem for these folks!

# In[ ]:


titanic_train.loc[titanic_train['Fare'] > 500, :]


# In[ ]:


titanic_no_500s = titanic_train.loc[titanic_train['Fare'] < 500, :]
sns.boxplot(x = 'Survived', y = 'Fare', data = titanic_no_500s, palette = 'RdBu_r')
titanic_no_500s.groupby('Survived').mean()[['Fare']]


# With the fare's of 500+ removed, the boxplots are more readable. The mean and median are definitely higher for those who survived and will include as a feaure for model training.

# Now I want to take a look at the effect of male vs female passengers

# In[ ]:


sns.countplot(x = 'Sex', data = titanic_train, hue = 'Survived')


# Looking at this chart more male a larger proportion of male passengers didn't survive when compared to female. Will consider this as an important feature for model training and building.

# Let's take a look at the Age column

# In[ ]:


hist = sns.distplot(titanic_train['Age'], color='b', bins=30, kde=False)
hist.set(xlim=(0, 100), title = "Distribution of Passenger Age's")


# In[ ]:


titanic_train.Age.describe()


# In[ ]:


age_box = sns.boxplot(y = 'Age', x = 'Survived',data = titanic_train, palette='coolwarm')
age_box.set(title='Boxplot of Age')


# Based on the description and histogram our passengers are roughly normally distributed with a mean of 29 and median of 26 years of age respectively. Looking at the boxplots of ages of passengers who did and didn't survive the distributions look relatively similar. Based on this I'm debating including the age column in model training.

# Embarked Column

# In[ ]:


titanic_train.groupby(['Embarked']).count()


# In[ ]:


sns.countplot(x = 'Embarked', hue = 'Survived', data=titanic_train)


# Looks like people who boarded from S were more likely to not survive than those who didn't board at S

# PClass column

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass', data=titanic_train, palette = 'rainbow')


# Looks like a majority of those who didn't survive were in the 3rd P-class. Would definitely be worth including as a feature in the model.

# ## Feature Engineering

# First step is to make copies of each dataframe

# In[ ]:


#Make copies of both dataframes.
traindf = titanic_train.copy()
testdf = titanic_test.copy()


# Next I'm going to put the copied dataframes into a list so I can perform the same actions to both dataframes.

# In[ ]:


#Create list of both data frames to apply similar functions to.
all_data = [traindf, testdf]


# ### Drop Name and Ticket Columns

# In[ ]:


#Drop name and ticket columns
for dat in all_data:
    dat.drop(['Name', 'Ticket'], axis=1, inplace=True)


# ### Bin Fare Column
# Next I'm going to bin the fare column based on the summary statistics for that column

# In[ ]:


traindf.describe()['Fare']


# Looks like some good cutoff points will be 0, 8, 15, 31, and 515 to include the max fare value of 512.

# In[ ]:


#Perform operation on both frames
for dat in all_data:
    
    #Create bins to separate fares
    bins = (0, 8, 15, 31, 515)

    #Assign group names to bins
    group_names = ['Fare_Group_1', 'Fare_Group_2', 'Fare_Group_3', 'Fare_Group_4']

    #Bin the Fare column based on bins
    categories = pd.cut(dat.Fare, bins, labels=group_names)
    
    #Assign bins to column
    dat['Fare'] = categories


# ### Bin Age Column

# In[ ]:


traindf.describe()['Age']


# Am going to try binning by every 15 years.

# In[ ]:


#Perform operation on both frames
for dat in all_data:
    
    #Create bins to separate fares
    bins = (0, 15, 30, 45, 60, 75, 90)

    #Assign group names to bins
    group_names = ['Child', 'Young Adult', 'Adult', 'Experienced', 'Senior', 'Elderly']

    #Bin the Fare column based on bins
    categories = pd.cut(dat.Age, bins, labels=group_names)
    
    #Assign bins to column
    dat['Age'] = categories


# In[ ]:


traindf.head()


# ### Create Family Size Feature. SibSp + Parch

# In[ ]:


for dat in all_data:
    dat['Fam_Size'] = dat['SibSp'] + dat['Parch']


# ### Use one hot encoding to code categorical variables.

# In[ ]:


traindf = pd.get_dummies(traindf)
traindf.head()


# In[ ]:


testdf = pd.get_dummies(testdf)
testdf.head()


# # Machine Learning
# In order to predict whether a passenger survived the titainc or not, a classification machine learning algorithm will be needed. I've decided for this kernel to try the following methods:
# * Logistic Regression
# * Support Vector Machine
# * K Nearest Neighbors
# 
# The steps I'm going to take to find the best model are outlined below
# 1. Split data into training, validation, and test sets
# 2. Train and fit each model to training data
# 3. Test each model on validation data
# 4. Pick model with highest prediction accuracy on validation set.
# 5. Use model from step 4 on test dataset.

# In[ ]:


#Import libraries
from sklearn.metrics import confusion_matrix #confusion matrix
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.ensemble import RandomForestClassifier #Random Forest Classifier
from sklearn.svm import SVC #Support Vector Machine
from sklearn.preprocessing import StandardScaler #For scaling data
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.model_selection import train_test_split #Split data into training and validation sets.
from sklearn.metrics import accuracy_score  #Accuracy Score


# ### 1. Split data into training and validation sets
# Because we already have the test dataset provided to us, all we need to do is split the training dataset into a training and validation set.

# In[ ]:


#Split data into training and validation set
X = traindf.drop(columns=['PassengerId', 'Survived'], axis=1)
y = traindf['Survived']

#Note they are labeled as test sets but I'm treating them as validation data sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# ### 2. Train and fit each model to train, test on validaiton data.
# I will do this for each model listed above. The dataframe below will hold the validation results.

# In[ ]:


results = pd.DataFrame(columns=['Validation'], index=['Logistic Regression', 'Support Vector Machine', 'KNN', 'Random Forest'])


# ### Logistic Regression
# First create function to train, fit, and test logistic regression model on validation data

# In[ ]:


def log_reg(X_train, X_test, y_train, y_test):
    #Create logmodel object
    logmodel = LogisticRegression(C=.01)

    #fit logistic regression model
    logmodel.fit(X_train, y_train)

    #Make predictions on validation data
    predictions = logmodel.predict(X_test)
    
    #Print Statistics
    print(accuracy_score(y_test, predictions))
    
    #Return predictions
    return accuracy_score(y_test, predictions)


# In[ ]:


#Get prediction accuracy for model.
LR_preds = log_reg(X_train, X_test, y_train, y_test)

#Add to dataframe.
results.loc['Logistic Regression', 'Validation'] = LR_preds


# ### Support Vector Machine
# First create function to train, fit, and test support vector machine model on. For SVM we will need to scale the input features.
# 

# In[ ]:


def svm(X_train, X_test, y_train, y_test):
    
    #Scale data
    #scaler = StandardScaler()
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    
    #Create list of c values to try
    c_vals = list(range(1, 100))
    
    #Accuracy list
    accuracy = [0 for i in range(99)]
    
    #Loop through c_values
    for i, c in enumerate(c_vals):
        #Create support vector machine object
        svc_model = SVC(C=c)
        
        #fit support vector machine model
        svc_model.fit(X_train, y_train)
        
        #Make predictions
        predictions = svc_model.predict(X_test)
        
        #add accuracy score to accuracy list
        accuracy[i] = accuracy_score(y_test, predictions)
    
    print("Best C Value:", c_vals[accuracy.index(max(accuracy))])
    print(accuracy)
    print("Prediction Accuracy: ", max(accuracy))
    
    return max(accuracy)
        
        


# In[ ]:


#Get support vector machine results
svm_preds = svm(X_train, X_test, y_train, y_test)

#Add to dataframe.
results.loc['Support Vector Machine', 'Validation'] = svm_preds
results.head()


# ### K Nearest Neighbors
# First create function to train, fit, and test K Nearest Neighbors model on. For KNN we will need to scale the input features.

# In[ ]:


def knn(X_train, X_test, y_train, y_test):
    
    #Scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    #Create list of c values to try
    ks = [i + 1 for i in range(20)]
    
    #Accuracy list
    accuracy = [0 for i in range(20)]
    
    #Loop through c_values
    for i, k in enumerate(ks):
        #Create support vector machine object
        knn = KNeighborsClassifier(n_neighbors = k)
        
        #fit support vector machine model
        knn.fit(X_train, y_train)
        
        #Make predictions
        predictions = knn.predict(X_test)
        
        #add accuracy score to accuracy list
        accuracy[i] = accuracy_score(y_test, predictions)
    
    print(ks)
    print(accuracy)
    print("Best k Value:", ks[accuracy.index(max(accuracy))])
    
    print("Prediction Accuracy: ", max(accuracy))
    
    return max(accuracy)


# In[ ]:


knn_preds = knn(X_train, X_test, y_train, y_test)
results.loc['KNN', 'Validation'] = knn_preds
results.head()


# ![](http://)Use SVM with C = 1 to make predictions on testing data.

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


scaler = StandardScaler()
scaler.fit(X)
test_feats = testdf.drop('PassengerId', axis=1)
X = scaler.transform(X)
test_feats = scaler.transform(test_feats)
pca = PCA(n_components = 4)
pca.fit(X)
x_train_pca = pca.transform(X)
x_test_pca = pca.transform(test_feats)
svc_model = SVC(C = 1)
svc_model.fit(x_train_pca, y)
svm_predictions = svc_model.predict(x_test_pca)
output = pd.DataFrame({ 'PassengerId' : testdf['PassengerId'], 'Survived': svm_predictions })
output.to_csv('titanic-predictions-svm-pca.csv', index=False)
output


# In[ ]:


#svc_model = SVC(C = 1)
#svc_model.fit(X, y)
#svm_predictions = svc_model.predict(test_feats)
#output = pd.DataFrame({ 'PassengerId' : testdf['PassengerId'], 'Survived': svm_predictions })
#output.to_csv('titanic-predictions-svm.csv', index=False)


# Conclusion: This model resulted in 78.947% accuracy which ranks in the top 1/2 of submissions on the kaggle leaderboard. As this was intended to be a 
# simple notebook to reinforce learning concepts I'm pretty happy with this result. As I continue to improve my feature engineering skills and understand the workings of more advanced machine learning models I will update this kernel to try and improve upon the body of work that is here.
# 
# If you  made it this far, thanks for reading! Any feedback is appreciated :)
