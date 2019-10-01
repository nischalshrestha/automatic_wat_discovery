#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# * [Background](#Background)
# * [Key Questions](#Key-Questions)
# * [Findings and Conclusions](#Find​ings-and-Conclusions)
# * [Load and Preview Data](#Load-and-Preview-Data)
# 	* [Variable Descriptions](#Variable-Descriptions)
# * [Data Wrangling and Exploration](#Data-Wrangling-and-Exploration)
# 	* [Get a High-level View](#High-level-View)
# 	* [High-level Summary](#High-level-Summary)
# 	* [Remove Redundant Columns and Replace Missing Values](#Remove-Redundant-Columns-and-Replace-Missing-Values)
# 	* [Explore Correlations](#Explore-Correlations)
# * [Prediction](#Prediction)

# # Background

# This is the final project of [Udacity Intro to Data Analysis][1] course. In the following analysis, I will predict which passenger will survive in the [Titanic tragedy][2]. ​Based on the given data of survivors, I explore similar characteristics of survivors to find out what types of passengers ​are more likely to survive. The final step is to make a prediction on the unknown passenger data to see if they can survive.
# 
# [1]:https://www.udacity.com/course/intro-to-data-analysis--ud170
# [2]:https://en.wikipedia.org/wiki/Sinking_of_the_RMS_Titanic

# # Key Questions

#  1. How do I choose what kind of data to include in the model? 
#  2. How do I choose which model to use? 
#  3. How do I optimize this model for better prediction?​

# # Findings and Conclusions

# After exploring the dataset a bit, I discovered potential correlations between variables such as gender and survival. For the sake of simplicity, I chose a simple Logistic Regression for modeling. It yields ~75% accuracy in the unknown test data. I.e. given any unknown passenger, the model is able to successfully predict it's survival 3 out of 4 passengers.
# 
# Here are a few things to note in the model:
# 
# - As I've discovered some "correlations",  no solid statistics analysis was performed to imply any causality between variables.
# - I found a specific age interval in the training dataset that showed significantly different survival rate. Survival rate drops significantly if someone is older than 10. 
# - Having some (1~3) family members on the ship are likely to increase the survival rate. However having too many or traveling alone is likely to decrease someone's chance of survival.​

# In[ ]:


# Import libraries

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid',{'axes.grid' : False})
sns.set_context(rc = {'patch.linewidth': 0.0})
bar_settings = {'color': sns.xkcd_rgb['grey'], 'ci': None}
color_settings = {'color': sns.xkcd_rgb['grey']}
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression


# # Load and Preview Data

# In[ ]:


# get titanic training dataset & test csv files as a DataFrame
train_df = pd.read_csv('../input/train.csv')
test_df  = pd.read_csv('../input/test.csv')

# preview the data
train_df.head()


# ## Variable Description​

# |Header Name|Descriptions|
# |-----------|------------|
# |Survived   | Survival (0 = No; 1 = Yes)
# |Pclass     |Passenger Class  (1 = 1st; 2 = 2nd; 3 = 3rd)
# |Name       |Name
# |Sex        |Sex
# |Age        |Age
# |SibSp      |Number of Siblings/Spouses Aboard
# |Parch      |Number of Parents/Children​ Aboard
# |Ticket     |Ticket Number
# |Fare       |Passenger Fare
# |Cabin      |Cabin
# |Embarked   |Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# In[ ]:


train_df.info()
print("----------------------------")
test_df.info()


# # Data Wrangling and Exploration

# ## High-level View

# In[ ]:


## Survived - Survival (0 = No; 1 = Yes)

total_passengers = train_df['Survived'].count()
survived_passengers = train_df['Survived'].sum()
survived_ratio = survived_passengers/total_passengers

print('Passengers in training data:',total_passengers)
print('% of survivors:',(survived_ratio*100).round(1),'%')

# A horizontal line of average survival rate

def avg_survived(survived_ratio):
    print(plt.axhline(y=survived_ratio,ls=":", c='.5'))
    print(plt.legend(['Avg. survival rate'],loc='center right'))


# In[ ]:


# Pclass - Survival (0 = No; 1 = Yes)

print('Number of passengers in each class: ',train_df.groupby('Pclass').count()['PassengerId'])
sns.barplot('Pclass','Survived', data=train_df, **bar_settings)
avg_survived(survived_ratio)


# In[ ]:


#Sex

sns.barplot('Sex','Survived', data=train_df,**bar_settings)
avg_survived(survived_ratio)


# In[ ]:


#Age
initial_age_values = train_df['Age'].copy().dropna().astype(int)
plt.hist(initial_age_values,**color_settings)


# In[ ]:


# Family - SibSp & Parch
fig_family, (axis1,axis2) = plt.subplots(1,2)

# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(x, y)
# ax1.set_title('Sharing Y axis')
# ax2.scatter(x, y
avg_sibsp_survived = train_df.groupby('SibSp',as_index=False)['SibSp','Survived'].mean()
avg_parch_survived = train_df.groupby('Parch',as_index=False)['Parch','Survived'].mean()

sns.barplot(x='SibSp',y='Survived', data=avg_sibsp_survived, ax=axis1, **bar_settings)
sns.barplot(x='Parch',y='Survived', data=avg_parch_survived, ax=axis2, **bar_settings)
avg_survived(survived_ratio)


# In[ ]:


# Ticket
# It seems like there are a lot of different tickets and they don't follow clear patterns

# unique_tickets = train_df.groupby(['Ticket'], as_index=False)['PassengerId'].count()
# unique_tickets.rename(columns={'PassengerId':'Counts'})


# In[ ]:


# Fare

fare_boxplt = sns.boxplot(x=train_df['Fare'],fliersize=3,**color_settings)
fare_boxplt.set(xlim=(0, 250))


# In[ ]:


# Cabin

print('# of non-NAN or non-null values: ', train_df['Cabin'].count())
print('Total number of rows: ', len(train_df['Cabin']))


# In[ ]:


# Embarked

train_df.groupby(['Embarked'])['PassengerId'].count()


# ## High-level Summary

# Before diving deeper, here are a few things to note:
# - Some columns are probably redundant for analysis, I can drop them:
# 
# |Column Name|Reason|
# |-----------|------|
# |PassengerId| No practical meaning 
# |Ticket     | No information is provided to explain different ticket numbers
# |Cabin      | Too many NaN values
#  
# - For the sake of simplicity, I also want to drop the following columns because *intuitively* I don't think these variables are useful in prediction. I could be wrong, though.
# 
# |Column Name|Reason|
# |-----------|------|
# |Name       |SibSp and Parch data should be enough in this simple prediction|
# |Embarked   | Logically speaking, the port chosen shouldn't affect chances of survival
# 
# - Missing values in *Age* and *Fare* column in train_df/test_df
# 
# - Without knowing the exact causal relations  between variables, I noticed some potential correlations to explore further:
# 
# Survived v.s. Pclass/Fare
# 
# Survived v.s. Sex
# 
# Survived v.s. SibSp/Parch
# 
# Survived v.s  Age
# 
# 
# 
# 

# ## Remove Redundant Columns and Replace Missing Values

# In[ ]:


# Remove redundant columns

train_df.drop(['PassengerId','Ticket','Cabin','Name','Embarked'],axis=1,inplace=True)
test_df.drop(['Ticket','Cabin','Name','Embarked'],axis=1,inplace=True)

train_df.info()
print("----------------------------")
test_df.info()


# In[ ]:


# Replace missing values in Age column

# Generate random age numbers between (mean - std) & (mean + std) for missing age values
train_age_mean = train_df['Age'].mean()
test_age_mean = test_df['Age'].mean()

train_age_std = train_df['Age'].std()
test_age_std = test_df['Age'].std()

# count # of NaN values
train_count_nan_age = train_df['Age'].isnull().sum()
test_count_nan_age = test_df['Age'].isnull().sum()

rand_train = np.random.randint(train_age_mean - train_age_std, train_age_mean + train_age_std, size=train_count_nan_age) 
rand_test = np.random.randint(test_age_mean - test_age_std, test_age_mean + test_age_std, size=test_count_nan_age) 

# Replace initial NaN values with new set of random numbers
train_df['Age'].loc[train_df['Age'].isnull()] = rand_train
test_df['Age'].loc[test_df['Age'].isnull()] = rand_test

# Convert from float to int
train_df['Age'] = train_df['Age'].astype(int)
test_df['Age'] = test_df['Age'].astype(int)

# Plot original age values and new age values

fig, (axes1, axes2) = plt.subplots(nrows=1,ncols=2, figsize=(15,5))

axes1.set_title('Initial Age Values')
axes2.set_title('New Age Values - NaN Replaced')
axes1.set_ylim(0,250)

axes1.hist(initial_age_values, **color_settings)
axes2.hist(train_df['Age'], color=sns.xkcd_rgb['light grey'])


# In[ ]:


# Replace missing values in Fare column

# Only need to replace one missing value in Fare column at test_df
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# Convert from float to int
train_df['Fare'] = train_df['Fare'].astype(int)
test_df['Fare'] = train_df['Fare'].astype(int)


# In[ ]:


# Take a quick look at current dataset

# train_df.info()
# print("----------------------------")
# test_df.info()


# ## Explore Correlations

# In[ ]:


# Survived v.s. Pclass/Fare
# Create dummy variables for Pclass column, & drop 3rd class as the base column

train_pclass_dummies = pd.get_dummies(train_df['Pclass'])
test_pclass_dummies = pd.get_dummies(test_df['Pclass'])

train_pclass_dummies.columns = ['Class1','Class2','Class3']
test_pclass_dummies.columns = ['Class1','Class2','Class3']

train_pclass_dummies.drop('Class3',axis=1,inplace=True)
test_pclass_dummies.drop('Class3',axis=1,inplace=True)

# Merge new dummy variables into origional dataset
train_df = train_df.join(train_pclass_dummies)
test_df = test_df.join(test_pclass_dummies)

# We don't need Pclass column now
train_df.drop('Pclass', axis=1, inplace=True)
test_df.drop('Pclass', axis=1, inplace=True)


# In[ ]:


# Correlation between fare and survival
# Bubble size = # of people in that group

fare_range = np.arange(0,300,20)
fare_groups = pd.cut(train_df['Fare'], fare_range)
grouped_fare = train_df.groupby(fare_groups)['Survived'].mean()

# count()*3 -> just to make the bubble size bigger
num_people_fare_groups = train_df.groupby(fare_groups)['Survived'].count()*3
plt.scatter(fare_range[:len(fare_range)-1], grouped_fare, s=num_people_fare_groups.tolist(), **color_settings)
plt.xlim(-20,300)
avg_survived(survived_ratio)


# In[ ]:


# Survived v.s. Sex
# Survived v.s. Age

# Chance of survival is higher for female
# Now we want to explore the chance of  survival across different age groups
# Bubble size = # of people in that group

age_range = np.arange(0,100,10)

train_df_male_sex = train_df[train_df.Sex == 'male']
train_df_female_sex = train_df[train_df.Sex == 'female']

age_groups_male = pd.cut(train_df_male_sex['Age'], age_range, include_lowest=True)
age_groups_female = pd.cut(train_df_female_sex['Age'], age_range, include_lowest=True)

grouped_age_male = train_df_male_sex.groupby(age_groups_male)['Survived'].mean()
grouped_age_female = train_df_female_sex.groupby(age_groups_female)['Survived'].mean()

# *3 to make the bubble larger
num_people_age_groups_male = train_df_male_sex.groupby(age_groups_male)['Survived'].count()*3
num_people_age_groups_female = train_df_female_sex.groupby(age_groups_female)['Survived'].count()*3

plt.scatter(age_range[:len(age_range)-1], grouped_age_male, s=num_people_age_groups_male.tolist(), color=sns.xkcd_rgb['light grey'])
plt.scatter(age_range[:len(age_range)-1], grouped_age_female, s=num_people_age_groups_female.tolist(), **color_settings)

avg_survived(survived_ratio)
plt.xlim(-2,80)


# As we can see from the scatter chart above, passengers below 10 can be classified as the same group. Passengers above 10 can be classified as another two gro​ups.

# In[ ]:


def passengers(passenger):
    age, sex = passenger
    if age <= 10:
        return 'child'
    else:
        return sex

train_df['Person'] = train_df[['Age','Sex']].apply(passengers, axis=1)
test_df['Person'] = test_df[['Age','Sex']].apply(passengers, axis=1)

# Create dummy variables for Person column

person_dummies_train = pd.get_dummies(train_df['Person'])
person_dummies_test = pd.get_dummies(test_df['Person'])

person_dummies_train.columns = ['Child','Female','Male']
person_dummies_test.columns = ['Child','Female','Male']

# Merge dummy variables into initial dataset

train_df = train_df.join(person_dummies_train)
test_df = test_df.join(person_dummies_test)

# Plot survival rate of Child, Female and Male
train_mean_person_survived = train_df.groupby('Person', as_index=False)['Person','Survived'].mean()
sns.barplot(x=['Child','Female','Male'],y=train_mean_person_survived['Survived'], **bar_settings)
avg_survived(survived_ratio)

# drop Person and Sex column as we don't need it anymore

train_df.drop('Person', axis=1, inplace=True)
test_df.drop('Person', axis=1, inplace=True)

train_df.drop('Sex', axis=1, inplace=True)
test_df.drop('Sex', axis=1, inplace=True)

# Drop Male as the base column

train_df.drop('Male', axis=1, inplace=True)
test_df.drop('Male', axis=1, inplace=True)


# In[ ]:


# Survived v.s. SibSp/Parch

train_df['Family'] = train_df['SibSp'] + train_df['Parch']
test_df['Family'] = test_df['SibSp'] + test_df['Parch']

train_mean_survived_with_family = train_df.groupby('Family', as_index=False)['Family','Survived'].mean()

plt.scatter(x='Family', y='Survived', data=train_mean_survived_with_family, **color_settings)
plt.xticks(np.arange(0,11,1))
avg_survived(survived_ratio)

# Drop SibSp and Parch column as we don't need them

train_df.drop(['SibSp','Parch'], axis=1, inplace=True)
test_df.drop(['SibSp','Parch'], axis=1, inplace=True)


# From the chart above we can somehow identify three groups: 
# - Alone (single passenger)
# - With_family (family members < 4)
# - With_big_family (family members >=4)

# In[ ]:


# Transform family member counts into categorical data

train_df['Family'].loc[train_df['Family'] == 0] = 'Alone'
train_df['Family'].loc[train_df['Family'].isin([1,2,3])] = 'With_family'
train_df['Family'].loc[train_df['Family'].isin(np.arange(4,11,1))] = 'With_big_family'

test_df['Family'].loc[test_df['Family'] == 0] = 'Alone'
test_df['Family'].loc[test_df['Family'].isin([1,2,3])] = 'With_family'
test_df['Family'].loc[test_df['Family'].isin(np.arange(4,11,1))] = 'With_big_family'

# Get dummy variables and merge with initial dataset
family_dummies_train = pd.get_dummies(train_df['Family'])
train_df = train_df.join(family_dummies_train)

family_dummies_test = pd.get_dummies(test_df['Family'])
test_df = test_df.join(family_dummies_train)

# Drop Family column as we already have dummy variables
# Drop Alone as the base column

train_df.drop('Family', axis=1, inplace=True)
test_df.drop('Family', axis=1, inplace=True)

train_df.drop('Alone', axis=1, inplace=True)
test_df.drop('Alone', axis=1, inplace=True)


# In[ ]:


# train_df.info()
# train_df.head()


# # Prediction

# In[ ]:


# Define training and testing sets

X_train = train_df.drop("Survived",axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)


# In[ ]:


# Get Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df['Coefficient Estimate'] = pd.Series(logreg.coef_[0])

# preview
coeff_df


# In[ ]:



submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic_Luke.csv', index=False)

