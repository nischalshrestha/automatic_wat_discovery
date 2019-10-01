#!/usr/bin/env python
# coding: utf-8

# # Project - Who Survived the Titanic?
# 
# 1.Import Python Packages
# 
# 2.Getting the Data and Dictionary
# 
# 3.Dataset analysis
# 
# 4.Dataset Cleaning & Transformations/engineering 
#          - Drop attributes, fill values, Categorize, convert to numeric, add new attributes.
# 
# 5.Visualization and data Analysis.
# 
# 6.Predictive Model
# 
# 7.Summary

# ### Step 1: Import Python Packages

# In[ ]:


import pandas as pd
from pandas import Series,DataFrame
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
from IPython.display import Image

from sklearn.neighbors import KNeighborsClassifier


# ### Step 2: Getting the Data and Data Dictionary

# In[ ]:


# Download the dataset from kaggle, the following link for reference:
# Kaggle website: https://www.kaggle.com/c/titanic/data, file “train.csv”

# Data file path on local drive
zs_train_csv_input="../input/train.csv"
zs_train_csv_input


# In[ ]:


# Read the data file and create a dataframe
titanic_df = pd.read_csv(zs_train_csv_input)

# Let's see a preview of the data
titanic_df.head()


# ###### Metadata

# In[ ]:


# Get overall info for the dataset
titanic_df.info()


# ###### Analysis:
# The train data set has 891 records(Passenger Ids)
# 
# Two attributes (Age,Fare) are floats: 5 attributes (PassengerId, Survived, Pclass,SibSp, Parch) are integers and 5 attributes (Name,Sex,Ticket, Cabin and Embraked) are objects. Noticed missing info in the Age,Cabin,Embarked as the respective records are less than < 891 records of the dataset

# ###### Data dictionary from Kaggle
# 
#     Variable	Definition	    Key
#     survival	Survival	    0 = No, 1 = Yes
#     pclass	    Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
#     sex	        Sex	
#     Age	        Age             in years	
#     sibsp	    # of siblings / spouses aboard the Titanic	
#     parch	    # of parents / children aboard the Titanic	
#     ticket	    Ticket number	
#     fare	    Passenger fare	
#     cabin	    Cabin number	
#     embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
# 
# Variable Notes
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# ### Step 3: Dataset Analysis

# In[ ]:


# Descriptive statistics
titanic_df.describe()


# #### Data Analysis:
# 
# Min and max values: This can give us an idea about the range of values and is helpful to detect outliers. However, this is not standalone measure, needs Mean (central tendency of data distribution) and Standard deviation to assess the outliners. All atttribute Min/max values seems to be reasonable, except Age and Fare.
# 
# Mean: shows the central tendency of the distribution.
# 
# Standard deviation: quantifies the amount of variation from the central tendency of the data distribution. Ignore PassengerId.
# 
# 
# Count: # number of records, helps to identify the missing data. Here, Noticed missing info in the Age attribute.
# 
# Lets analyze missing data. 

# In[ ]:


# Null value records per attribute
total_missing = titanic_df.isnull().sum()
total_missing_sort =total_missing.sort_values(ascending=False)
total_missing_sort


# In[ ]:


#total number of records of the dataset.
total_count = titanic_df.isnull().count()
total_count


# ###### Data Analysis
# 
# The Embarked feature has only 2 missing values, which can easily be filled. It will be much more tricky, to deal with the 'Age' feature, which has 177 missing values. The 'Cabin' feature needs further investigation, but it looks like that we might want to drop it from the dataset, since 77 % of it are missing.

# In[ ]:


# Columns in the dataset

titanic_df.columns.values


# 'PassengerId', 'Ticket' and 'Name' 'Cabin'- not every helpful of the model in determining correlation of high survival rate.  
# 
# Lets analyze the other attributes, helpful for predictive model - 
# 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch',  'Fare', 'Embarked'
# 
# 
# 
# ##### Categorical Attributes:
# 
# A.Nominal:'Sex', 'Embarked' 'Survived'
# 
# B.Ordinal: 'Pclass' 
# 
# Note: 'Survived' is not taken in the model because it's the output variable.
# 
# 
# 
# 
# ##### Numerical Attributes:
# 
# C.Continous: Age, Fare.
# 
# D.Discrete: SibSp, Parch.
# 
# Note: 'SibSp' could be grouped with 'Parch' to create a 'Relative' feature. 
# 
# 

# ### Step 4: Dataset Cleaning & Transformations/engineering

# Objective:
# To prepare clean dataset with list of attributes that can correlate with Survival
# 
# 
# 
# Assumptions:
# 1. Women (Sex=female) were more likely to have survived.
# 2. Children (Age<?) were more likely to have survived.
# 3. The upper-class passengers (Pclass=1) were more likely to have survived.
# 
# 
# 
# To achieve that:
# 1. Drop Attributes /columns - Cabin, Name, Ticker, as they either contain more missing values and also may not contribute directly to survival
# 2. Fill missing values of - Age, Embraked
# 3. Convert to Numeric 
# 4. Categorize
# 5. Add new calculation variables.

# #### 4.1: Drop Attributes

# ##### Cabin, Name, Ticket

# In[ ]:


#Dropping cabin as 77% missing data. Hence, not helpful for predictive model.

titanic_df.drop('Cabin', axis=1, inplace=True) # Reason is: more missing values
titanic_df.drop('Name', axis=1, inplace=True)  # Reason is: its a direct correlation factor for survival
titanic_df.drop('Ticket', axis=1, inplace=True) # Reason is: its a direct correlation factor for survival
titanic_df.head()


# #### 4.2: Fill Missing Values and Convert to Numeric

# ##### Age :

# In[ ]:


# Two methods to fill missing values; Used Method 2


# Method 1 : Hardcode (for reference)
# Fill missing values in Age with a specific value
#value = 1000
#titanic_df['Age'].fillna(value, inplace=True)
#titanic_df['Age'].max()


#Method 2 : use random value in the normal distribution or Bell curve

data = [titanic_df]

for dataset in data:
    mean = titanic_df["Age"].mean()
    std = titanic_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null) # generate random number within the quartiles
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age #assign random number
    dataset["Age"] = age_slice
    dataset["Age"] = titanic_df["Age"].astype(int)   #convert from float to integer


# ##### Embraked :

# In[ ]:


# info on Embarked
titanic_df['Embarked'].describe()


# In[ ]:


# Two methods to fill missing values; Used Method 1


# Method 1 : Hardcode
# Fill missing values in Age with a specific value
titanic_df['Embarked'].fillna('S', inplace=True)  # assign "S" for two missing values


#Method 2 : mode value (for reference)
#embrk_common_value = titanic_df["Embarked"].mode()  # get the mode value of Embarked
#data = [titanic_df]
#for dataset in data:
    #dataset['Embarked'] = dataset['Embarked'].fillna(value=embrk_common_value) # assign the mode value    


# In[ ]:


# categorize 'Age' into 1 to 6 bucket.
ports = {"S": 0, "C": 1, "Q": 2}
data = [titanic_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)


# In[ ]:


# distinct values
titanic_df['Embarked'].value_counts()


# ##### Fare :

# In[ ]:


# Fill missing value with zero

data = [titanic_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)


# ##### Stats on Null values :

# In[ ]:


# Qucik check on the Null value records per attribute
total_missing = titanic_df.isnull().sum()
total_missing_sort =total_missing.sort_values(ascending=False)
total_missing_sort


# In[ ]:


# metadata after the conversion
titanic_df.info()


# ##### Sex :

# In[ ]:


# Convert into numeric.

genders = {"male": 0, "female": 1}
data = [titanic_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)    


# In[ ]:


# distinct values
titanic_df['Sex'].value_counts()


# ##### Embarked :

# In[ ]:


# distinct values
titanic_df['Embarked'].value_counts()


# ##### Age :

# In[ ]:


# categorize 'Age' into 1 to 6 bucket.

data = [titanic_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6


# In[ ]:


# distinct values
titanic_df['Age'].value_counts()


# ##### Fare :

# In[ ]:


data = [titanic_df]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[ ]:


# distinct values
titanic_df['Fare'].value_counts()


# In[ ]:


# dataset  after cleansing and transformation
titanic_df.head(10)


# #### 4.4: Add Calculated Measures

# #####  Age-vs-Class  (age_class):

# In[ ]:


# Passengers with less age and those who occupied Upper Class has higher chance of survival 
# Refer analysis section for more details.

data = [titanic_df]

for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']


# In[ ]:


# distinct values
titanic_df['Age'].value_counts()


# #####  Relation between sibsp and Parch  (relatives, not_alone):

# In[ ]:


# Relative is combination of Sibilings/spouse and Parent.

data = [titanic_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)


# In[ ]:


# distinct values
titanic_df['not_alone'].value_counts()


# In[ ]:


# distinct values
titanic_df['relatives'].value_counts()


# In[ ]:


# dataset  after new measures. Data model is ready for predictive model and further analysis.
titanic_df.head(10)


# ### Step 5: Visulaization and analysis

# In[ ]:


# Passengers occupied in the 1st Class ( ie Pclass = 1) have highest survival rate.

titanic_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# explained the above using a bar graph.
sns.barplot(titanic_df['Pclass'], titanic_df['Survived']);


# In[ ]:


#Female Passengers has high survival rate.
# As mapped in the above , genders = {"male": 0, "female": 1}

titanic_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# 1 ie Female, has high chance of survival. Maybe, many women are travelling in Class=2
# Interestingly Men (0) in class=3 has high chance of survival.

sns.barplot(titanic_df['Pclass'],titanic_df['Sex'],titanic_df['Survived']);


# In[ ]:


# 0 means with relative, 1 means no relatives. refer above section data transformation values
# Passengers who are travelling with relatives ( sibilings, father/mother, sibilings etc) have high survival rate.

titanic_df[['not_alone', "Survived"]].groupby(['not_alone'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# Passengers with relative has high chance of survival.

sns.barplot(titanic_df['not_alone'],titanic_df['Survived']);


# In[ ]:


# As mentioned in Calculation measure, Age_class is combination of age and class.
# Younger (ctegory 1: Age'(> 11 and <= 18) ) age who occupied Upper Class, have high chances of survival. 

titanic_df[['Age_Class', "Survived"]].groupby(['Age_Class'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


#Passengers with age 1 category (ie'Age'(> 11 and <= 18))  in upper class, has hightest chance of survival. 
sns.barplot(titanic_df['Age_Class'],titanic_df['Survived']);


# ### Step 6: Predictive Model.

# In[ ]:


# Y axis for the "Survived" where as X axis for other attributes in the data model

X_train = titanic_df.drop("Survived", axis=1)
Y_train = titanic_df["Survived"]
X_test  = titanic_df.drop("PassengerId", axis=1).copy()


# In[ ]:


# KNN (K-nearest neighbors algorithim) (https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(round(acc_knn,2,), "%")


# ### Step 8: Summary

# To define a data model for machine learning model or identify( or engineer) attributes in analyzing the survival rate, 
# we should get the data, followed by data cleansing, fill null values, other transformations etc.,
# 
# Once data model is ready, then one of the machinlearning models (like KNN) is used to find the survival rate. 

# In[ ]:


Image(url='https://i.gifer.com/5SlH.gif')

