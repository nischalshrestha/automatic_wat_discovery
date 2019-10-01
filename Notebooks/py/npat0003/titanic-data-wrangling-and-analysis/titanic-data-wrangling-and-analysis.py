#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset Analysis
# 
# ## By: Nick Patil
#           14/12/2016

# **1. Data Parsing**
# ---------------
# 
# Import the libraries to read, plot and analyse the data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


# #**Use pandas to read the csv into the dataframe**

# In[ ]:


titanic1 = pd.read_csv('../input/train.csv')
titanic2 = pd.read_csv('../input/test.csv')


# ## Merge the two dataframes

# In[ ]:


titanic = pd.merge(titanic1, titanic2, how='outer')


# ##Take an overview of the data

# In[ ]:



titanic.head()


# In[ ]:


titanic.info()


# ### There are few missing values for Age, Cabin columns
# ### Also Survived column has missing values for which we need to create a model for predictions

# In[ ]:


# The name column can be splitted into more meaningful columns for better analysis 
titanic.Name.unique()


# In[ ]:


# Lets seperate the titles from the name 
coltitle = titanic['Name'].apply(lambda s: pd.Series({'Title': s.split(',')[1].split('.')[0].strip(),
                                                   'LastName':s.split(',')[0].strip(), 'FirstName':s.split(',')[1].split('.')[1].strip()}))
print (coltitle)


# In[ ]:


# Add the columns to the titanic dataframe
titanic = pd.concat([titanic, coltitle], axis=1) 
# Drop the Name column
titanic.drop('Name', axis=1, inplace=True)
print (titanic.head())


# In[ ]:


# Lets check the number of male and female
titanic.Sex.value_counts()


# In[ ]:


# Lets set a style for all the plots
print (style.available)
style.use('classic')


# In[ ]:


# Lets plot the number of male and females on the ship
titanic.Sex.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


# Lets check the number of casualties on the ship
titanic.Survived.value_counts()


# In[ ]:


# Lets plot the casualties
titanic.Survived.value_counts().plot(kind='bar', title='Number of people survived [0 - Not Surv, 1 - Surv]\n')
plt.show()


# ## Lets now find number of passengers based on their Titles

# In[ ]:


# We can use the title column to get an inside
titanic.Title.unique()


# In[ ]:


# Also reassign mlle, ms, and mme accordingly
titanic.loc[titanic['Title']=='Mlle', 'Title']='Miss'.strip()
titanic.loc[titanic['Title']=='Ms', 'Title']='Miss'.strip()
titanic.loc[titanic['Title']=='Mme', 'Title']='Mrs'.strip()


# In[ ]:


# Get the count of female and male passengers based on titles
tab = titanic.groupby(['Sex', 'Title']).size()
print (tab)


# In[ ]:


# Now lets get the count of unique surnames 
print (titanic.LastName.unique().shape[0])


# ##Total number of families on the ship

# In[ ]:


titanic['total_members'] = titanic.SibSp + titanic.Parch + 1


# ## Do families sink or swim together based on number of family members
# 
# 
# ----------
# 

# In[ ]:


survivor = titanic[['Survived', 'total_members']].groupby('total_members').mean()
survivor.plot(kind='bar')
plt.show()


# ##We can see that there’s a survival penalty to singletons and those with family sizes above 4¶
# 

# In[ ]:


titanic.isnull().sum()


# ##Drop unnecessary columns, these columns won't be useful in analysis and prediction

# In[ ]:


# Drop the Ticket and Cabin column 
titanic.drop('Cabin', axis=1, inplace=True)
titanic.drop('Ticket', axis=1, inplace=True)


# In[ ]:


# There is one missing value in Fare
titanic[titanic.Fare.isnull()==True]


# In[ ]:


titanic[['Pclass', 'Fare']].groupby('Pclass').mean()


# In[ ]:


titanic.loc[titanic.PassengerId==1044.0, 'Fare']=13.30


# In[ ]:


# Check the null values in Embarked column
titanic.Embarked.isnull().sum()


# In[ ]:


titanic[titanic['Embarked'].isnull() == True]


# ## Impute missing value based on Survived column
# We see that they paid 80 dollars respectively and their classes are 1 and 1, also they survived. So lets try to find where they embarked from.

# In[ ]:


# Lets try to find the embark based on survived
titanic[['Embarked', 'Survived']].groupby(['Embarked'],as_index=False).mean()


# We will go with C as the passengers survived and there is 55% chance for surviving with Embark C

# In[ ]:


# Also lets try to find the fare based on Embarked 
titanic[['Embarked', 'Fare']].groupby('Embarked').mean()


# ##The fare they paid is 80 dollars which is close to C, hence we can impute C as the missing value. 

# In[ ]:


# Imputting the missing value
titanic.loc[titanic['Embarked'].isnull() == True, 'Embarked']='C'.strip()


# ## Check the missing values for Age

# In[ ]:


titanic.Age.isnull().sum()


# In[ ]:


titanic.Age.plot(kind='hist')
plt.show()


# ##  The Age can be predicted based on Sex, Title and Pclass of existing customer and imputting the median age value.

# In[ ]:


pd.pivot_table(titanic, index=['Sex', 'Title', 'Pclass'], values=['Age'], aggfunc='median')


# In[ ]:


# a function that fills the missing values of the Age variable
    
def fillAges(row):
    
    if row['Sex']=='female' and row['Pclass'] == 1:
        if row['Title'] == 'Miss':
            return 29.5
        elif row['Title'] == 'Mrs':
            return 38.0
        elif row['Title'] == 'Dr':
            return 49.0
        elif row['Title'] == 'Lady':
            return 48.0
        elif row['Title'] == 'the Countess':
            return 33.0

    elif row['Sex']=='female' and row['Pclass'] == 2:
        if row['Title'] == 'Miss':
            return 24.0
        elif row['Title'] == 'Mrs':
            return 32.0

    elif row['Sex']=='female' and row['Pclass'] == 3:
        
        if row['Title'] == 'Miss':
            return 9.0
        elif row['Title'] == 'Mrs':
            return 29.0

    elif row['Sex']=='male' and row['Pclass'] == 1:
        if row['Title'] == 'Master':
            return 4.0
        elif row['Title'] == 'Mr':
            return 36.0
        elif row['Title'] == 'Sir':
            return 49.0
        elif row['Title'] == 'Capt':
            return 70.0
        elif row['Title'] == 'Col':
            return 58.0
        elif row['Title'] == 'Don':
            return 40.0
        elif row['Title'] == 'Dr':
            return 38.0
        elif row['Title'] == 'Major':
            return 48.5

    elif row['Sex']=='male' and row['Pclass'] == 2:
        if row['Title'] == 'Master':
            return 1.0
        elif row['Title'] == 'Mr':
            return 30.0
        elif row['Title'] == 'Dr':
            return 38.5

    elif row['Sex']=='male' and row['Pclass'] == 3:
        if row['Title'] == 'Master':
            return 4.0
        elif row['Title'] == 'Mr':
            return 22.0


titanic['Age'] = titanic.apply(lambda s: fillAges(s) if np.isnan(s['Age']) else s['Age'], axis=1)


# ##Plot after imputting the missing values

# In[ ]:


titanic.Age.plot(kind='hist')
plt.show()


# # Prediction for Survived
# 
# ###Sex, Pclass, Age, Embarked, Kids, Mother, total_members

# In[ ]:


titanic.info()


# ##Convert objects to numeric for predictions

# In[ ]:


# Convert sex to 0 and 1 (Female and Male)
def trans_sex(x):
    if x == 'female':
        return 0
    else:
        return 1
titanic['Sex'] = titanic['Sex'].apply(trans_sex)

# Convert Embarked to 1, 2, 3 (S, C, Q)
def trans_embark(x):
    if x == 'S':
        return 3
    if x == 'C':
        return 2
    if x == 'Q':
        return 1
titanic['Embarked'] = titanic['Embarked'].apply(trans_embark)    
    


# In[ ]:


# Add a child and mother column for predicting survivals
titanic['Child'] = 0
titanic.loc[titanic['Age']<18.0, 'Child'] = 1
titanic['Mother'] = 0
titanic.loc[(titanic['Age']>18.0) & (titanic['Parch'] > 0.0) & (titanic['Sex']==0) & (titanic['Title']!='Miss'), 'Mother'] =1


# ##predict who survives among passengers of the Titanic based on variables that we carefully curated and treated for missing values

# In[ ]:


titanic.isnull().sum()


# ##We divide the datasource into training and test data based on Null values in Survived column

# In[ ]:


# Feature selection for doing the predictions
features_label = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'total_members', 'Child', 'Mother']
target_label= ['Survived']
train = titanic[titanic['Survived'].isnull()!= True]
test = titanic[titanic['Survived'].isnull()== True]


# In[ ]:


print (train.shape)
print (test.shape)


# ##Random Forest Regression 

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X=train[features_label], y=train[target_label])

Y_pred = random_forest.predict(X=test[features_label])

random_forest.score(X=train[features_label], y=train[target_label])


# ##Using Logistic Regression to predict and imputing the predicted values into the Survived column with null values

# In[ ]:


# Logistic Regression
regr = LogisticRegression()
regr.fit(X=train[features_label], y=train[target_label])
regr.score(X=train[features_label], y=train[target_label])


# In[ ]:


# Predicted Values for Survived
predict_t = regr.predict(X=test[features_label])
print (predict_t)


# In[ ]:


# Insert the predicted values for the missing rows for Survived column
titanic.loc[titanic['Survived'].isnull()== True, 'Survived']= predict_t


# ##Extra Trees model for selecting Features based on importance

# In[ ]:


# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X=train[features_label], y=train[target_label])
# display the relative importance of each attribute
importance = model.feature_importances_
print (importance)


# In[ ]:


# model is of type array, convert to type dataframe

imp = pd.DataFrame({'feature':features_label,'importance':np.round(model.feature_importances_,3)})
imp = imp.sort_values('importance',ascending=False).set_index('feature')
print (imp)
imp.plot.bar()
plt.show()


# ##passengers who survived based on sex, class, child and mothers

# In[ ]:


print ("\nThe number of passengers based on Sex\n")
print (titanic['Sex'].value_counts()) 

print ("\nThe number of survivors based on Sex\n")
print(titanic[['Survived', 'Sex']].groupby('Sex').sum()) 

print ("\nThe number of passengers based on Pclass\n")
print (titanic['Pclass'].value_counts())
       
print("\nThe number of survivors based on Pclass\n")
print(titanic[['Survived', 'Pclass']].groupby('Pclass').sum()) 

print ("\nThe number of passengers who are Mother\n")
print (titanic['Mother'].value_counts())
       
print ("\nThe number of survivors based on Mother\n")
print (titanic[['Survived', 'Mother']].groupby('Mother').sum())


# # Inferences
# ### From the above more females survived then men, More of VIP Pclass(1) passengers survived then common passengers, Mother survivor is also high

# ## Convert the columns to their string values

# In[ ]:


# Convert sex to 0 and 1 (Female and Male)
def trans_sex(x):
    if x == 0:
        return 'female'
    else:
        return 'male'
titanic['Sex'] = titanic['Sex'].apply(trans_sex)

# Convert Embarked to 1, 2, 3 (S, C, Q)
def trans_embark(x):
    if x == 3:
        return 'S'
    if x == 2:
        return 'C'
    if x == 1:
        return 'Q'
titanic['Embarked'] = titanic['Embarked'].apply(trans_embark) 


# ## Export the data into a CSV file

# In[ ]:


titanic.to_csv('titanic.csv')


# ##Summary:
# Thanks Kaggle for the opportunity to do this project. As I am new to Data Science enjoyed doing it. 
# Suggestions are welcome.
# 

# In[ ]:




