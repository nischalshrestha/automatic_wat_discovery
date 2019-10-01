#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# # Load CSV
# First, we load the CSV FIles

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# Combine both datasets to apply common operations
combine = [df_train, df_test]


# # Inspect the datasets
# 
# We inspect the data to see how it's formed.
# We note the following for each dataset (train and test):
# [](http://)
# ## Training Data
# - There are 891 rows of data, 1 for each passenger
# - The *Age* column has 177 **(19.86%)** missing values.
# -  The *Cabin* column has 694 **(77.89%)** missing values
# - The *Embarked* column has 2 **(0.002%)** missing values
# ## Test Data
# - There are 418 rows of data, 1 for each passenger.
# - The *Age* column has 86 **(20.57%)** missing values (similar to the training data).
# - The *Cabin* column has 327 **(78.22%)** missing values (similar to the training data).
# - The *Fare* column has 1 **(0.002%)** missing value.
# 

# In[ ]:


# Print the training data information
print('TRAIN INFO \n',df_train.info())
# Print a separator line
print('-'*50)
# Print the test data information
print('TEST INFO \n',df_test.info())


# ## Categorical values
# The following columns are categorical:
# - **Survived**: Boolean indicating 1 if the passenger survived or 0 if the passenger did not survive.
# - **Pclass**: An int describing the ticket class, which also refers to the passenger's socio-economic class. Values are: 
#     - 1: Upper
#     - 2: Middle
#     - 3: Lower
# 
# - **Sex**: A string describing the passenger's gender.
# - **Embarked**: A string representing the port of embarcation. Values are:
#     - C = Cherbourg
#     - Q = Queenstown
#     - S = Southampton

# In[ ]:


df_train.head()


# ## Distribution of numerical data
# From the distribution of the training data, we can make the following assumptions:
# - 38.38% of the passengers survived
# * - Most passengers traveled without parents or children

# In[ ]:


df_train.describe()


# ## Distribution of dategorical data
# - Cabin only has values on 204 rows, and seems that there are many duplicates (unique: 147)
# - Ticket has values on all rows, but contains many duplicates (unique 681)

# In[ ]:


df_train.describe(include=['O'])


# # Exploratory Analysis
# ### Survival rate by Pclass <br>
# We check the survival rate by Pclass and see that there seem's to be a relationship bewteen Pclass and Survival rate

# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=df_train.groupby('Pclass', as_index=False).agg({'Survived':'mean'}))
plt.xlabel(s='Pclass',fontsize=16)
plt.ylabel(s='Survival Rate',fontsize=16)
plt.title('Survival Rate by Pclass', fontsize=18)


# ### Survival rate by Gender
# Another variable to check is the gender. In this case we see that Gender was also a KEY factor in survival rates

# In[ ]:


sns.barplot(x='Sex', y='Survived', data=df_train.groupby('Sex', as_index=False).agg({'Survived':'mean'}))
plt.xlabel(s='Gender',fontsize=16)
plt.ylabel(s='Survival Rate',fontsize=16)
plt.title('Survival Rate by Gender', fontsize=18)


# ### Survival Distribution by Age
# - We can see that passengers with age 0 to 4 had a high survival rate.
# - Passengers with ages 15 to 35 had a lower survival rate.

# In[ ]:


g = sns.FacetGrid(df_train, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# ### Survival by Port of Embarcation and Gender
# Plot the survival rate by segmented by port of embarkation and gender.<br>
# Again we see a strong survival rate for females in all ports, but the ratio bewteen ports and gender varies:
# - In order of survival:
#     - Females survived most when embarked on Port C, then Port Q and least on Port S
#     - Males survived most when embarked on Port C, then Port S and least on Port Q

# In[ ]:


df_embarked_gender = df_train.groupby(['Embarked','Sex'], as_index=False).agg({'Survived':'mean'})
sns.barplot(x='Embarked', y='Survived', hue='Sex', data=df_embarked_gender)
plt.xlabel(s='Embarked',fontsize=16)
plt.ylabel(s='Survival Rate',fontsize=16)
plt.title('Survival Rate by Embarked port and Gender', fontsize=18)


# # Data Munging
# 

# ## Cleaning and imputing values
# - Since the Cabin column has so many duplicates, and missing values, we are going to drop this column from both datasets
# - Ticket has values on all rows, but contains many duplicates

# In[ ]:


# Drop columns Cabin and Ticket on both datasets
df_train = df_train.drop(['Cabin','Ticket'], axis=1)
df_test = df_test.drop(['Cabin','Ticket'], axis=1)

combine = [df_train, df_test]


# ### Impute Age Values
# Using the Sex, Embarked and Pclass , we wil ltry to impute the age using the mean of each category

# In[ ]:


for dataset in combine: # For each Dataset
    for gender in dataset['Sex'].unique(): # For each gender
        for embarked in dataset.loc[~dataset['Embarked'].isnull(),'Embarked'].unique(): # For each port
            for pclass in dataset.loc[~dataset['Pclass'].isnull(),'Pclass'].unique(): # For each class
                
                # Get a Dataframe with not null values to guess the age (using dropna)
                guess_df = dataset.loc[(dataset['Sex'] == gender) & (dataset['Embarked'] == embarked) & (dataset['Pclass'] == pclass),'Age'].dropna()
                
                # Get the mean
                guessed_age = guess_df.mean()
                
                # Set it to the dataset
                dataset.loc[(dataset['Age'].isnull()) & (dataset['Sex'] == gender) & (dataset['Embarked'] == embarked) & (dataset['Pclass'] == pclass),'Age'] = guessed_age


# ### Impute Fare null values
# - Fare null values will be imputed using the Median of Port, PClass and Age band<br>
# 

# In[ ]:


for dataset in combine: # For each Dataset
    for gender in dataset['Sex'].unique(): # For each gender
        for embarked in dataset.loc[~dataset['Embarked'].isnull(),'Embarked'].unique(): # For each port
            for pclass in dataset.loc[~dataset['Pclass'].isnull(),'Pclass'].unique(): # For each class
                for age in dataset.loc[~dataset['Age'].isnull(),'Age'].unique(): # For each age
                    
                    # Get a Dataframe with not null values to guess the fare (using dropna)
                    guess_df = dataset.loc[(dataset['Sex'] == gender) & (dataset['Embarked'] == embarked) & (dataset['Pclass'] == pclass) & (dataset['Age'] == age),'Fare'].dropna()
                
                    # Get the median
                    guessed_fare = guess_df.median()
                    
                    # Set it to the dataset
                    dataset.loc[(dataset['Fare'].isnull()) & (dataset['Sex'] == gender) & (dataset['Embarked'] == embarked) & (dataset['Pclass'] == pclass) & (dataset['Age'] == age),'Fare'] = guessed_fare



# Assign the median of the entire dataset for the rows we couldn't guess the fare
fare_median = df_train.loc[~df_train['Fare'].isnull(), 'Fare'].median()

df_train.loc[df_train['Fare'].isnull(), 'Fare'] = fare_median
df_test.loc[df_test['Fare'].isnull(), 'Fare'] = fare_median

combine = [df_train, df_test]


# ### Impute Embarked null values
# - Embarked null values will be imputed using the Median of Sex, Fare Category, PClass and Age band

# In[ ]:


def impute_embarked(use_fare=True):
    for dataset in combine: # For each Dataset
        # Should use fare to impute Embark port?
        if(use_fare):
            for gender in dataset['Sex'].unique(): # For each gender
                for fare in dataset.loc[~dataset['Fare'].isnull(),'Fare'].unique(): # For each fare
                    for pclass in dataset.loc[~dataset['Pclass'].isnull(),'Pclass'].unique(): # For each class
                        for age in dataset.loc[~dataset['Age'].isnull(),'Age'].unique(): # For each age
                        
                            # Get a Dataframe with not null values to guess the fare (using dropna)
                            guess_df = dataset.loc[(dataset['Sex'] == gender) & (dataset['Fare'] == fare) & (dataset['Pclass'] == pclass) & (dataset['Age'] == age),'Embarked'].dropna()
                        
                            # if the dataframe has values
                            if (len(guess_df) > 0):
                                # Get the mode
                                guessed_port = guess_df.mode()[0]
                            
                                # Set it to the dataset
                                dataset.loc[(dataset['Embarked'].isnull()) & (dataset['Sex'] == gender) & (dataset['Fare'] == fare) & (dataset['Pclass'] == pclass) & (dataset['Age'] == age),'Embarked'] = guessed_port
        # Dont use fare to impute Embarked port
        else:
            for gender in dataset['Sex'].unique(): # For each gender
                for pclass in dataset.loc[~dataset['Pclass'].isnull(),'Pclass'].unique(): # For each class
                    for age in dataset.loc[~dataset['Age'].isnull(),'Age'].unique(): # For each age
                        
                        # Get a Dataframe with not null values to guess the fare (using dropna)
                        guess_df = dataset.loc[(dataset['Sex'] == gender) & (dataset['Pclass'] == pclass) & (dataset['Age'] == age),'Embarked'].dropna()
                    
                        # if the dataframe has values
                        if (len(guess_df) > 0):
                            # Get the mode
                            guessed_port = guess_df.mode()[0]
                        
                            # Set it to the dataset
                            dataset.loc[(dataset['Embarked'].isnull()) & (dataset['Sex'] == gender) & (dataset['Pclass'] == pclass) & (dataset['Age'] == age),'Embarked'] = guessed_port
                            
impute_embarked(use_fare=False)

# Get the most frequent Port from all data for the ones that we couldn't find.
freq_port = df_train.loc[~df_train['Embarked'].isnull(),'Embarked'].mode()[0]
# Set the most frequent port to the null values
df_train.loc[df_train['Embarked'].isnull(),'Embarked'] = freq_port
df_test.loc[df_test['Embarked'].isnull(),'Embarked'] = freq_port

combine = [df_train, df_test]


# ## Convert Categorical Values
# - We are going to convert the categorical values into dummies
# 

# ### Gender to Dummies

# In[ ]:


# Get Dummies for Sex Column
df_sex_dummies = pd.get_dummies(df_train['Sex'], prefix='sex_', drop_first=True)
df_train = pd.concat([df_train, df_sex_dummies], axis=1)   
df_train = df_train.drop('Sex', axis=1)

df_sex_dummies = pd.get_dummies(df_test['Sex'], prefix='sex_', drop_first=True)
df_test = pd.concat([df_test, df_sex_dummies], axis=1)   
df_test = df_test.drop('Sex', axis=1)


# ## Dummies for Port of Embark

# In[ ]:


# Get Dummies for Sex Column
df_sex_dummies = pd.get_dummies(df_train['Embarked'], prefix='embarked_', drop_first=True)
df_train = pd.concat([df_train, df_sex_dummies], axis=1)   
df_train = df_train.drop('Embarked', axis=1)

df_sex_dummies = pd.get_dummies(df_test['Embarked'], prefix='embarked_', drop_first=True)
df_test = pd.concat([df_test, df_sex_dummies], axis=1)   
df_test = df_test.drop('Embarked', axis=1)


# ## Dummie for Pclass

# In[ ]:


# Get Dummies for Sex Column
df_sex_dummies = pd.get_dummies(df_train['Pclass'], prefix='pclass_', drop_first=True)
df_train = pd.concat([df_train, df_sex_dummies], axis=1)   
df_train = df_train.drop('Pclass', axis=1)

df_sex_dummies = pd.get_dummies(df_test['Pclass'], prefix='pclass_', drop_first=True)
df_test = pd.concat([df_test, df_sex_dummies], axis=1)   
df_test = df_test.drop('Pclass', axis=1)


# ## Creating Title Feature (not!)
# On the [Titanic Data Science Solutions Notebook](https://www.kaggle.com/startupsci/titanic-data-science-solutions) by [Manav Sehgal](https://www.kaggle.com/startupsci) he wrote a routine to get the Title from the passengers name, and created a new feature. <br>
# My main analysis for NOT doing this, is that makes the data more complex and it doesn't really add much value, since Title is heavily tied to Sex, so we skip this proposed step.
# ![](http://)** Taking this into account we can safely drop the Name Column**

# In[ ]:


# Drop the Name column on both datasets
df_train = df_train.drop(['Name'], axis=1)
df_test = df_test.drop(['Name'], axis=1)

combine = [df_train, df_test]


# ## Creating family related features
# Based on the Manav Seghal notebook, we also create a feature called Familysize, which combines Parch and SibSp features.
# [](http://) <br>
# Also, we create the IsAlone Feature semented by Gender wich indicates if the passenger had family with a boolean instead of the number of family members and the gender, since the chances of survivng being Alone are different for males and females. <br>
# A correlation heatmap can show a small correlation between these values (0.43 and 0.34).

# In[ ]:


for dataset in combine:
    # Family Size Feature
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Is alone Feature
    dataset['IsAlone'] = 0
    dataset.loc[(dataset['FamilySize'] == 1), 'IsAlone'] = 1


# Display a heatmap to see the correlation between the values.
sns.heatmap(df_train[['FamilySize','Survived','IsAlone']].corr(), annot=True)


# ### Drop Family Size Feature
# The Family Size feature helped us create the IsAlone feature, but has no actual correlation to the survival chance, so we can drop it.

# In[ ]:


df_train = df_train.drop(['FamilySize','SibSp','Parch'], axis=1)
df_test = df_test.drop(['FamilySize','SibSp','Parch'], axis=1)

combine = [df_train, df_test]


# # Model Prediction

# In[ ]:


# Get the Features without the Label
X_train = df_train.drop(['Survived','PassengerId'], axis=1)
Y_train = df_train['Survived']

# Get the Test values
X_test = df_test.drop(['PassengerId'], axis=1).copy()


# ## Random Forests
# Based on Manav's notebook, the Random Forests is the best option for prediction on this Dataset.
# Let's see how well we did.<br>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=50)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
score = random_forest.score(X_train, Y_train)

print('Accuracy',score)


# In[ ]:


submission = pd.DataFrame(
    {
        'PassengerId': df_test['PassengerId'],
        'Survived': Y_pred
    })

#submission.to_csv('../input/my_submission.csv', index=False)
submission


# In[ ]:




