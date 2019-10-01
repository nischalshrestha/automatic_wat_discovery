#!/usr/bin/env python
# coding: utf-8

# **Titanic Survival Predictions**
# 
# I am new to data science and machine learning and this is my first attempt at Kaggle where I will be trying to predict the possibility of a passenger surviving on the Titanic using the *Titanic: Machine Learning from Disaster dataset*
# 
# We will tackle this problem with the following steps
# 1. Importing the packages and libraries.
# 2. Reading and Exploring the Data.
# 3. Data Analysis.
# 4. Visual Data Analysis.
# 5. Cleaning the Data
# 6. Feature Engineering
# 7. Machine learning
# 8. Submitting our predictions
# 
# I would love to see your feedback in the comments section!

# **1)  Importing the packages and libraries**
# 
# let's start off by importing the necessary libraries for data analysis and visualisation

# In[72]:


#lets load the required packages and libraries for data analysis
import numpy as np 
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

#For data visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# **2) Reading and exploring the data**
# 
#  let's read the training and testing datasets from the provided CSV files and use the ***.head()*** and ***.info()*** methods to take a glimpse at our data

# In[73]:


#importing the training and test datasets
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')                                                               
                       


# In[74]:


#lets take a look at our training data
train_df.head()


# In[75]:


# Now the test dataset
test_df.head()


# The Survived column is missing here because that is what we are supposed to predict with our model.

# In[76]:


# lets see what kind of data we have to work with
train_df.info()


# From above, we can see that we  have 891 rows or samples and  12 columns of types  *int64* , *object* and *float64 *to work with

# **3) Data analysis :**
# 
# Now let's see what features we have to train our model on and what useful insights we can obtain from them.  
# 

# In[77]:


#printing out a list of all the columns in our training dataset
train_df.columns


# ** Types of features : **
# 
# * Categorical : Pclass, Sex, Embarked, Survived
# * Continuous : Age, Fare, Sibsp, Parch, PassengerId
# * Alphanumeric: Ticket, Cabin, Name

# 

# Now that we know what kind of features we are going to work with, let's take a look what information they provide us:

# In[78]:


#printing summary statistics
train_df.describe()


# ** Observations from above summary statistics: **
# * There are a total of 891 passengers in our training dataset.
# * Since the Survived column has dicrete data, the mean gives us the number of people survived from 891 i.e. 38%.
# * Most people belonged to Pclass = 3
# * The maximum Fare paid for a ticket was 512 however the fare prices varied a lot as we can see from the standard deviation of 49

# In[79]:


train_df.describe(include='O')


# Taking a look at our categorical features we find that: 
# * The passneger column has two sexes with male being the most common.
# * Cabin feature has many duplicate values.
# * Embarked has three possible values with most passengers embarking from Southhampton.
# * Names of all passengers are unique.
# * Ticket column also has a fair amount of duplicate values.
# 

# In[80]:


#Finding the percantage of missing values in train dataset
train_df.isnull().sum()/ len(train_df) *100


# In[81]:


#Finding the percentage of Null values in test dataset
test_df.isnull().sum()/ len(test_df) *100


# As we can see the Age column and Embarked column are missing values that we will need to fill.
# The Cabin coulmn has 77% and 78% missing values  in train and test datasets respectively hence, it might be worth considering dropping that feature.

# ** 4) Visual Data Analysis**
# 
# It's time to visualize our data and try to draw some inferences from it

# ** Sex feature**
# 
# let's begin by exploring the Sex column in our trainig data set

# In[82]:


sns.countplot('Sex',data=train_df)
train_df['Sex'].value_counts()


# The number of males on board were clearly more than the female. Now let's see how their survival percentages were:

# In[83]:


#Comparing the Sex feature against Survived
sns.barplot(x='Sex',y='Survived',data=train_df)
train_df.groupby('Sex',as_index=False).Survived.mean()


# As one would assume the number of female who survived was much more than the males who survived i.e. 74%  females as against to 18% males

# How did the Class of each passenger affect their survival?

# In[84]:


#Comparing the Pclass feature against Survived
sns.barplot(x='Pclass',y='Survived',data=train_df)
train_df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Clearly Class had an effect on survival of each passenger with the percentages of survival being 62.96%, 47.28%, 24.23% for Pclass 1, 2 and 3 respectively.
# Thus, belonging to Pclass = 1 had a huge advantage. 

# Did the port from which the passengers embarked have an effect on their Survival?

# In[85]:


#Comparing the Embarked feature against Survived
sns.barplot(x='Embarked',y='Survived',data=train_df)
train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# It seems that the passengers that embarked from port Cherbourg had a higher rate of Survival at 55%. This could be either due to their Sex or socio-economic class.
# Let's move forward to see the effect of having parents or children on-board.

# In[86]:


sns.barplot(x='Parch',y='Survived',data=train_df)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Looks like passengers who had either 1, 2 or 3  had a higher possibility of surviving than the ones had none. However having more than 3 made the possibility even lesser.
# Moving on to the effect of having spouse or siblings on Survival:

# In[87]:


sns.barplot(x='SibSp',y='Survived',data=train_df)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# It seems that having a spouse or 1 sibling had a positive effect on Survival as compared to being alone. Though the chances of survival go down with the number of siblings after 1.

# The Age column has some missing values. We will take care of that later when we clean our training data.
# First we shall proceed by:
# 1.  Plotting a histogram of the age values .
# 2. Taking a look at the median value of age as well as the spread.

# In[88]:


train_df.Age.hist(bins=10,color='teal')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
print("The Median age of passengers is :", int(train_df.Age.median()))
print("The Standard Deviation age of passengers is :", int(train_df.Age.std()))


# It is obvious to assume that younger individuals were more likely to survive, however we should test our assumption before we proceed.

# In[89]:


sns.lmplot(x='Age',y='Survived',data=train_df,palette='Set1')


# Our assumption was right, younger individuals were more likely to survive.

# From the sex column we saw that there was a significant difference in the percentage of men and women that survived.
# Does sex also play a role when it comes to surviving the disaster along with the age?

# In[90]:


sns.lmplot(x='Age',y='Survived',data=train_df,hue='Sex',palette='Set1')


# Interestingly, age has an opposite effect on the survival in men and women. The chances of survival increase as the age of women increases.
# 
# Takeaway: Age feature can have a different  effect on the outcome depending on the sex of the passenger. Perhaps we can use this information in feature engineering

# In[91]:


#Checking for outliers in Age data
sns.boxplot(x='Sex',y='Age',data=train_df)

#getting the median age according to Sex
train_df.groupby('Sex',as_index=False)['Age'].median()


# In[92]:


#plotting the Fare column to see the spread of data
sns.boxplot("Fare",data=train_df)

#Checking the mean and median values
print("Mean value of Fare is :",train_df.Fare.mean())
print("Median value of Fare is :",train_df.Fare.median())


# ** 5)Cleaning Data ***
# 
# Now that we have visualized our data , we can proceed to fill in the NaN values in our test and train datasets and drop the columns that we will not require

# In[93]:


#let's start off by dropping the coulmns we will not be needing
drop_list=['Cabin','Ticket','PassengerId']

train_df = train_df.drop(drop_list,axis=1)
test_passenger_df = pd.DataFrame(test_df.PassengerId)
test_df = test_df.drop(drop_list,axis=1)

test_passenger_df.head()


# Now, let's fill in the missing values for Embarked column in the training dataset. Most people embarked on their journey from Southhampton port. Hence, we will be filling the  two missing values with "S"

# In[94]:


#filling the missing Embarked values in train and test datasets
train_df.Embarked.fillna('S',inplace=True)


# We will replace the NaN values in the age column with the median age

# In[95]:


#filling the missing values in the Age column
train_df.Age.fillna(28, inplace=True)
test_df.Age.fillna(28, inplace=True)


# There is a small fraction of fare values missing in the fare column which we will fill using the median value since there a plenty of outliers in the data.

# In[96]:


#Filling the null Fare values in test dataset
test_df.Fare.fillna(test_df.Fare.median(), inplace=True)


# **6) Feature Engineering**

# *Title Feature*
# The name column might not be useful to us directly but a lot of names have titles like Mr, Mrs, Lady, etc which might indicate the individual's status in the society which can affect the chance of survival.
# 
# We shall try to extract a *Title* feature form the name column which might improve the performanc of our model.

# In[97]:


#combining train and test dataframes to work with them simultaneously
Combined_data = [train_df, test_df]


# In[98]:


#extracting the various title in Names column
for dataset in Combined_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#Plotting the various titles extracted from the names    
sns.countplot(y='Title',data=train_df)  


# There are some titles that are very rare like Capt and Lady. It would be better to group such titles under one name know as 'rare'.
# Some titles also seem to be incorrectly spelled. They also need to be rectified.

# In[99]:


#Refining the title feature by merging some titles
for dataset in Combined_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Special')

    dataset['Title'] = dataset['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    
train_df.groupby('Title',as_index=False)['Survived'].mean().sort_values(by='Survived',ascending=False)


# In[100]:


#Now lets see the distribution of the title feature
sns.countplot(y='Title',data=train_df)


# In[101]:


#Mapping the title names to numeric values
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Special": 5}
for dataset in Combined_data:
    dataset['Title'] = dataset.Title.map(title_mapping)
    dataset['Title'] = dataset.Title.fillna(0)


# As we observed from our data visualization being alone on the titanic had a disadvantage when it came to survival:
# Next we will create a feature IsAlone which depends on the number of family members that can be calculated from the Parch and SibSp columns 

# In[102]:


#Creating a new feature IsAlone from the SibSp and Parch columns
for dataset in Combined_data:
    dataset["Family"] = dataset['SibSp'] + dataset['Parch']
    dataset["IsAlone"] = np.where(dataset["Family"] > 0, 0,1)
    dataset.drop('Family',axis=1,inplace=True)
train_df.head()    


# Getting rid of the columns that are not required anymore:

# In[103]:


#dropping the Name,SibSP and Parch columns
for dataset in Combined_data:
    dataset.drop(['SibSp','Parch','Name'],axis=1,inplace=True)  


# Age had big role to play when it came to survival. Clearly younger people were more likely to survive.
# Hence, it should be worth considering a feature IsMinor for the passengers under the age of 15.

# In[104]:


#Creating another feature if the passenger is a child
for dataset in Combined_data:
    dataset["IsMinor"] = np.where(dataset["Age"] < 15, 1, 0)


# Older female passengers also had a higher chance of survival. Let's create a feature name Old_female that would account for women older tha 50 years on board

# In[105]:


train_df['Old_Female'] = (train_df['Age']>50)&(train_df['Sex']=='female')
train_df['Old_Female'] = train_df['Old_Female'].astype(int)

test_df['Old_Female'] = (test_df['Age']>50)&(test_df['Sex']=='female')
test_df['Old_Female'] = test_df['Old_Female'].astype(int)


# Pclass, Sex and Embarked are the categorical features in our data. we can convert these categorucal variables into dummy variables using the *get_dummies* method in python

# In[106]:


#Converting categorical variables into numerical ones
train_df2 = pd.get_dummies(train_df,columns=['Pclass','Sex','Embarked'],drop_first=True)
test_df2 = pd.get_dummies(test_df,columns=['Pclass','Sex','Embarked'],drop_first=True)
train_df2.head()


# Age and Fare columns have continuous data and there might be fluctuations that do not reflect patterns in the data, which might be noise. That's why wel put people that are within a certain range of age or fare in the same bin. This can be achieved using *qcut* method in *pandas*

# In[107]:


#creating Age bands
train_df2['AgeBands'] = pd.qcut(train_df2.Age,4,labels=False) 
test_df2['AgeBands'] = pd.qcut(test_df2.Age,4,labels=False) 


# In[108]:


#creating Fare bands
train_df2['FareBand'] = pd.qcut(train_df2.Fare,7,labels=False)
test_df2['FareBand'] = pd.qcut(test_df2.Fare,7,labels=False)


# In[109]:


#Dropping the Age and Fare columns
train_df2.drop(['Age','Fare'],axis=1,inplace=True)
test_df2.drop(['Age','Fare'],axis=1,inplace=True)


# Let's take a final look at our training and testing data before we proceed to build our model.

# In[110]:


train_df2.head()
#sns.barplot('AgeBands','Survived',data=train_df2)


# In[111]:


test_df2.head()


# **7) Machine Learning**
# 
# We will try out some different ML models to see which gives us the best result.
# the process will be as follows:
# * Importing the required machine learning libraries from scikit learn.
# * Splitting out training data into train and test datasets to check the performance of our model.
# * Try out different classifying model to see which fits the best.

# In[112]:


#importing the required ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score


# In[113]:


#Splitting out training data into X: features and y: target
X = train_df2.drop("Survived",axis=1) 
y = train_df2["Survived"]

#splitting our training data again in train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)


# In[114]:


#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)
acc_logreg


# Our score also depends on how we had split our training data using *train_test_split*. We should also perform k-fold cross validation to get a more accurate score. Here we will be going with 5 folds.

# In[115]:


#let's perform some K-fold cross validation for logistic Regression
cv_scores = cross_val_score(logreg,X,y,cv=5)
 
np.mean(cv_scores)*100


# In[116]:


#Decision Tree Classifier

decisiontree = DecisionTreeClassifier()
dep = np.arange(1,10)
param_grid = {'max_depth' : dep}

clf_cv = GridSearchCV(decisiontree, param_grid=param_grid, cv=5)

clf_cv.fit(X, y)
clf_cv.best_params_,clf_cv.best_score_*100
print('Best value of max_depth:',clf_cv.best_params_)
print('Best score:',clf_cv.best_score_*100)


# In[117]:


#Random Forest CLassifier

random_forest = RandomForestClassifier()
ne = np.arange(1,20)
param_grid = {'n_estimators' : ne}

rf_cv = GridSearchCV(random_forest, param_grid=param_grid, cv=5)

rf_cv.fit(X, y)
print('Best value of n_estimators:',rf_cv.best_params_)
print('Best score:',rf_cv.best_score_*100)


# In[118]:


gbk = GradientBoostingClassifier()
ne = np.arange(1,20)
dep = np.arange(1,10)
param_grid = {'n_estimators' : ne,'max_depth' : dep}

gbk_cv = GridSearchCV(gbk, param_grid=param_grid, cv=5)

gbk_cv.fit(X, y)
print('Best value of parameters:',gbk_cv.best_params_)
print('Best score:',gbk_cv.best_score_*100)


# **7) Submission.**
# Finally, we are ready to submit our solution to see where we rank. To do so we need to make a submission.csv file that contains only the PassengerId and our predictions for those ID's.

# In[119]:


y_final = clf_cv.predict(test_df2)

submission = pd.DataFrame({
        "PassengerId": test_passenger_df["PassengerId"],
        "Survived": y_final
    })
submission.head()
submission.to_csv('titanic.csv', index=False)


# I hope this notebook helped you out and please free to give any feedback or advice in the comments. I am new and this would help me out a lot!
# 
# **Sources:**
# *  [Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
# * [Titanic Survival Predictions (Beginner)](https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner/notebook)
# * [Machine Learning with Kaggle: Feature Engineering](https://www.datacamp.com/community/tutorials/feature-engineering-kaggle)
