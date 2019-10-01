#!/usr/bin/env python
# coding: utf-8

# #  Exploratory Data Analysis on the Titanic Dataset
# 
# 
# This is my first kernal and the objective of this kernal is to conduct exploratory data analysis (EDA) and statistical modeling on the **Titanic Dataset** in order to gather insights and evenutally predicting survior(0 = Not Survived, 1 = Survived). Out of the 891 passengers that went on board the titanic, approximately 38% of them got surived where as majority 62% did not survive the disaster.
# 
# I have outlined below the process i followed in conducting the aforementioned procedure
# 
# > 1. Import the relevant python libraies for the analysis
# > 2. Load the train and test dataset and set the index if applicable
# > 3. Visually inspect the head of the dataset,Examine the train dataset to understand in particular if the data is tidy, shape of the dataset,examine datatypes, examine missing values, unique counts and build a data dictictionary dataframe
# > 4.  Run discriptive statistics of object and numerical datatypes,  and finally transform datatypes accordingly
# > 5. Carry-out univariate,bivariate and multivariate analysis using graphical and non graphical(some numbers represting the data) mediums
# > 6. Feature Engineering : Extract title from name, Extract new features from name, age, fare, sibsp, parch and cabin
# > 7. Preprocessing and Prepare data for statistical modeling
# > 7. Statistical Modelling

# ## 1. Import the relevant python libraries for the analysis

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display
get_ipython().magic(u'matplotlib inline')


# ## 2. Load the train and test dataset and set the index if applicable

# In[ ]:


#load the train dataset
train = pd.read_csv('../input/train.csv')


# In[ ]:


#inspect the first few rows of the train dataset
display(train.head())


# In[ ]:


# set the index to passengerId
train = train.set_index('PassengerId')


# In[ ]:


#load the test dataset
test = pd.read_csv('../input/test.csv')


# In[ ]:


#inspect the first few rows of the test dataset
display(test.head())


# ## 3. Visually inspect the head of the dataset,Examine the train dataset to understand in particular if the data is tidy, shape of the dataset,examine datatypes, examine missing values, unique counts and build a data dictionary dataframe

# Conditions to check if <span style="color:red"> **data is tidy**</span>
# - Is every column a variable? 
# - Is every row an observation?
# - Is every table a single observational unit?

# In[ ]:


#by calling the shape attribute of the train dataset we can observe that there are 891 observations and 11 columns
#in the data set
train.shape


# In[ ]:


# Check out the data summary
# Age, Cabin and Embarked has missing data
train.head()


# In[ ]:


# identify datatypes of the 11 columns, add the stats to the datadict
datadict = pd.DataFrame(train.dtypes)
datadict


# In[ ]:


# identify missing values of the 11 columns,add the stats to the datadict
datadict['MissingVal'] = train.isnull().sum()
datadict


# In[ ]:


# Identify number of unique values, For object nunique will the number of levels
# Add the stats the data dict
datadict['NUnique']=train.nunique()
datadict


# In[ ]:


# Identify the count for each variable, add the stats to datadict
datadict['Count']=train.count()
datadict


# In[ ]:


# rename the 0 column
datadict = datadict.rename(columns={0:'DataType'})
datadict


# ## 3.   Run discriptive statistics of object and numerical datatypes,  and finally transform datatypes accoringly

# In[ ]:


# get discripte statistcs on "object" datatypes
train.describe(include=['object'])


# In[ ]:


# get discriptive statistcs on "number" datatypes
train.describe(include=['number'])


# ## 4. Carryout univariate and multivariate analysis using graphical and non graphical(some numbers represting the data)

# In[ ]:


train.Survived.value_counts(normalize=True)


# only 38% of the passengers were survived, where as a majority 61% the passenger did not survive the disaster

# #### Univariate Analysis

# In[ ]:


fig, axes = plt.subplots(2, 4, figsize=(16, 10))
sns.countplot('Survived',data=train,ax=axes[0,0])
sns.countplot('Pclass',data=train,ax=axes[0,1])
sns.countplot('Sex',data=train,ax=axes[0,2])
sns.countplot('SibSp',data=train,ax=axes[0,3])
sns.countplot('Parch',data=train,ax=axes[1,0])
sns.countplot('Embarked',data=train,ax=axes[1,1])
sns.distplot(train['Fare'], kde=True,ax=axes[1,2])
sns.distplot(train['Age'].dropna(),kde=True,ax=axes[1,3])


# #### Bivariate EDA

#  - We can clearly see that male survial rates is around 20% where as female survial rate is about 75% which suggests that gender has a strong relationship with the survival rates.
#  
#  - There is also a clear relationship between Pclass and the survival by referring to first plot below. Passengers on Pclass1 had a better survial rate of approx 60% whereas passengers on pclass3 had the worst survial rate of approx 22%
#  
#  - There is also a marginal relationship between the fare and survial rate. 
#  
#  I have quantified the above relationships further in the last statsical modelling section
#  

# In[ ]:


figbi, axesbi = plt.subplots(2, 4, figsize=(16, 10))
train.groupby('Pclass')['Survived'].mean().plot(kind='barh',ax=axesbi[0,0],xlim=[0,1])
train.groupby('SibSp')['Survived'].mean().plot(kind='barh',ax=axesbi[0,1],xlim=[0,1])
train.groupby('Parch')['Survived'].mean().plot(kind='barh',ax=axesbi[0,2],xlim=[0,1])
train.groupby('Sex')['Survived'].mean().plot(kind='barh',ax=axesbi[0,3],xlim=[0,1])
train.groupby('Embarked')['Survived'].mean().plot(kind='barh',ax=axesbi[1,0],xlim=[0,1])
sns.boxplot(x="Survived", y="Age", data=train,ax=axesbi[1,1])
sns.boxplot(x="Survived", y="Fare", data=train,ax=axesbi[1,2])


# #### Joint Plots(continous vs continous)

# In[ ]:


sns.jointplot(x="Age", y="Fare", data=train);


# #### Multivariate EDA
# #### Construct a Coorelation matrix of the int64 and float64 feature types

# - There is a positve coorelation between Fare and Survived and a negative coorelation between Pclass and Surived
# 
# - There is a negative coorelation between Fare and Pclass, Age and Plcass

# In[ ]:


import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))
corr = train.corr()
sns.heatmap(corr,
            mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# ## 6. Feature Engineering Data- Extract title from name, Extract new features from the other features

# #### *New Features*

# In[ ]:


train['Name_len']=train.Name.str.len()


# In[ ]:


train['Ticket_First']=train.Ticket.str[0]


# In[ ]:


train['FamilyCount']=train.SibSp+train.Parch


# In[ ]:


train['Cabin_First']=train.Cabin.str[0]


# In[ ]:


# Regular expression to get the title of the Name
train['title'] = train.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False)


# In[ ]:


train.title.value_counts().reset_index()


# ## 7. Preprocessing and Prepare data for statistical modeling

# ** a. Imput Missing or Zero values to the <span style="color:blue"> Fare variable </span> **

# In[ ]:


# we see that there are 15 Zero values and its reasonbale 
# to flag them as missing values since every ticket 
# should have a value greater than 0
print((train.Fare == 0).sum())


# In[ ]:


# mark zero values as missing or NaN
train.Fare = train.Fare.replace(0, np.NaN)


# In[ ]:


# validate to see if there are no more zero values
print((train.Fare == 0).sum())


# In[ ]:


# keep the index
train[train.Fare.isnull()].index


# In[ ]:


train.Fare.mean()


# #### <span style="color:red">Having missing values in a dataset can cause errors with some machine learning algorithms and either the rows that has missing values should be removed or imputed </span>
# 
# Imputing refers to using a model to replace missing values.
# 
# There are many options we could consider when replacing a missing value, for example:
# 
# >- constant value that has meaning within the domain, such as 0, distinct from all other values.
# >- value from another randomly selected record.
# >- mean, median or mode value for the column.
# >- value estimated by another predictive model.

# In[ ]:


# impute the missing Fare values with the mean Fare value
train.Fare.fillna(train.Fare.mean(),inplace=True)


# In[ ]:


# validate if any null values are present after the imputation
train[train.Fare.isnull()]


# 
# 
# ** b. Imput Missing or Zero values to the <span style="color:blue"> Age variable </span> **

# In[ ]:


# we see that there are 0 Zero values
print((train.Age == 0).sum())


# In[ ]:


# impute the missing Age values with the mean Fare value
train.Age.fillna(train.Age.mean(),inplace=True)


# In[ ]:


# validate if any null values are present after the imputation
train[train.Age.isnull()]


# ** c. Imput Missing or Zero values to the <span style="color:blue"> Cabin variable </span> **

# In[ ]:


# We see that a majority 77% of the Cabin variable has missing values.
# Hence will drop the column from training a machine learnign algorithem
train.Cabin.isnull().mean()


# In[ ]:


train.info()


# ## 8. Statistical Modelling

# In[ ]:


train.columns


# In[ ]:


trainML = train[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
       'Fare', 'Embarked', 'Name_len', 'Ticket_First', 'FamilyCount',
       'title']]


# In[ ]:


# drop rows of missing values
trainML = trainML.dropna()


# In[ ]:


# check the datafram has any missing values
trainML.isnull().sum()


# ### A single predictor model with logistic regression
# we use logistic regression as the response variable is a binary classification

# ** <span style="color:blue"> Regression on survival on Age </span> **

# In[ ]:


# Import Estimator AND Instantiate estimator class to create an estimator object
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[ ]:


X_Age = trainML[['Age']].values
y = trainML['Survived'].values
# Use the fit method to train
lr.fit(X_Age,y)
# Make a prediction
y_predict = lr.predict(X_Age)
y_predict[:10]
(y == y_predict).mean()


# The prediction accuracy is marginally better than the base line accuracy of 61.5% which we got earlier

# ** <span style="color:blue"> Regression on survival on Fare </span> **

# In[ ]:


X_Fare = trainML[['Fare']].values
y = trainML['Survived'].values
# Use the fit method to train
lr.fit(X_Fare,y)
# Make a prediction
y_predict = lr.predict(X_Fare)
y_predict[:10]
(y == y_predict).mean()


# The prediction accuracy got a bit better than the Age variable and much better than 61.5% base accuracy

# ** <span style="color:blue"> Regression on survive on Sex(using a Categorical Variable) </span>  **

# In[ ]:


X_sex = pd.get_dummies(trainML['Sex']).values
y = trainML['Survived'].values
# Use the fit method to train
lr.fit(X_sex, y)
# Make a prediction
y_predict = lr.predict(X_sex)
y_predict[:10]
(y == y_predict).mean()


# The gender of passenger is a strong predictor and purely predciting based on gender, the model accuracy increased to 78%

# ** <span style="color:blue"> Regression on survive on PClass(using a Categorical Variable) </span>  **

# In[ ]:


X_pclass = pd.get_dummies(trainML['Pclass']).values
y = trainML['Survived'].values
lr = LogisticRegression()
lr.fit(X_pclass, y)
# Make a prediction
y_predict = lr.predict(X_pclass)
y_predict[:10]
(y == y_predict).mean()


# Gender of the passenger seems a strong predictor compared to the PClass of the passenger on Survival

# ** <span style="color:blue"> Predicting Survival based on Random forest model </span>**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
X=trainML[['Age', 'SibSp', 'Parch',
       'Fare', 'Name_len', 'FamilyCount']].values # Taking all the numerical values
y = trainML['Survived'].values
RF = RandomForestClassifier()
RF.fit(X, y)
# Make a prediction
y_predict = RF.predict(X)
y_predict[:10]
(y == y_predict).mean()


# Random forest did a good job in predicting the survival with a 97% accuracy
