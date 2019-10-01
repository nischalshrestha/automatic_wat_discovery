#!/usr/bin/env python
# coding: utf-8

# <a id="top-of-notebook"></a>
# # <center>Titantic: Learning to Machine Learn</center>
# <p>
# This notebook may be too simple for an intermediate Kaggler, but assumes you know the Python coding basics. There are plenty of Kernels out there that go at a faster pace. The objective of this notebook is to help beginners create their first model in a structured way. The score may not be high, but could be a great learning experience.
# <p>
# Also note, I'm still a beginner as well. So take this notebook as just a small supplement of ideas that might help you through your learning journey in data science!
# 
# <p>
# ## Before Starting
# What works for me when reading through a tutorial is to open a tab of *my* Kernel and a tab of the *tutorial* Kernel. <br>
# This way, I can work along with the tutorial and lookup anything that might be going wrong.
# 
# ## Kaggle Competition Process
# Start every competition with an overview of what is your gameplan.<br>This roadmap will help you if you get stuck and don't know what to do next.
# 1. [Problem Definition](#problem)
# 1. [Gather Data](#gather)
# 1. [Prep/Clean Data](#prep)
# 1. [Explore Data](#explore)
# 1. [Model Solution](#model)
# 1. [Clean Up Solution](#clean-soln)
# 1. [Submit Results](#results)

# <a id="problem"></a>
# ## 1. Problem Definition
# The first step is the easiest, yet very important. What is the point of your analysis?<br>
# Sometimes I find myself so deep into code that I forget the main objective.<br>
# Come to this section when you need a break so that you remember why you are doing all this work.
# <p>
# From the **Competition Description**, you can find the main objective of the competition:
# <p>
# > Complete the analysis of what sorts of people were likely to survive [the Titanic shipwreck]
# 
# <font size="1"><a href="#top-of-notebook">Go Back to Top</a></font>

# <a id="gather"></a>
# ## 2. Gather Data
# Load all your data from the competition in an easy-to-remember format for you to use.

# ### <font color="#696969">A. Import Python Modules</font>
# Some people call these things libraries instead of modules. Basically, you are importing a block of code that has been created to be a tool for your analysis.

# In[ ]:


import numpy as np # Used for linear algebra
import pandas as pd # Used for data processing
import os

''' Prints a list of all files in directory '''
print(os.listdir("../input"))
# This code is the default coding that comes when you first start a notebook
# The 'os' module and 'listdir' function is not really necessary to keep with your code. Its just a FYI.

import warnings
warnings.filterwarnings('ignore')
# This code prevents any annoying warnings that might show up when running your code.


# ### <font color="#696969">B. Assign Variables to Data</font>
# Assigning a variable is when you use a string of text to reference code or data in a short and memorable way. 
# <p>
# Most Kernels I've seen use the convention of "test_df" or "train_df" as their DataFrame variables, <br>
# but I prefer the opposite order: df_train and df_test. <br>
# It makes sense in my head. Use what makes the most sense to you.

# In[ ]:


''' Load Data '''
''' 
Extra Lesson:
'try' and 'except' methods are like the training wheels to riding a bicycle. 
They are not necessary but are useful as a beginner starts coding. 
The code used in the 'try' method will first run and if there is an error to 
the code the 'except' method will output instead.
'''
try:
    df_train = pd.read_csv("../input/train.csv")
    df_test = pd.read_csv("../input/test.csv")
    print('Files are loaded')
except:
    print('Something went wrong.')


# ### <font color="#696969">C. Combining All Data Together in One Dataset</font>
# In my previous versions of this notebook, I worked on the train and test datasets independent of each other. <br>
# I learned that this is not always the best practice, as you should combine the test and train dataset together to <br>
# base your model on the complete dataset. Here I will create a combined dataset called "ship"

# In[ ]:


ship = df_train.append(df_test, ignore_index=True)
# The append function add the dataframe you put inside the parenthesis to the bottom of your selected dataframe.
# In this case, we are adding df_test to the bottom of df_train. We use ignore_index=True to prevent
# any confusion between the two dataframes. We are saying, ignore the index on df_test and just use
# the df_train index.


# __*Before Moving Forward*__<br>
# Use this time to review the data in Excel, Google Sheets, or Python. <br>
# By reviewing the data, you can go into the cleaning phase with an idea of what data could be removed or added.
# <p>
# If you do not know how to download the data, click on the "Data" tab above or on the <br> 
# Competition page and next to the CSV files should be a download icon to download the files.
# 
# <font size="1"><a href="#top-of-notebook">Go Back to Top</a></font>

# <a id="prep"></a>
# ## 3. Prep/Clean Data
# This step is arguably the most important part of the Data Science process. 
# 
# > "Garbage in, garbage out" 
# 
# The above phrase is constantly stated in data research tutorials. Basically, clean data creates a strong foundation for the analysis to be built upon.<br>
# Cleaning and prepping data is also called Data Wrangling which is said because a person has to pull "wild" data into a manageable state
# <p>
# Skip to:
# [Clean-up Section](#3-clean)

# ### <font color="#696969">A. Use Columns Function to See Column Headers</font>
# (Code Directory: pandas.DataFrame.columns)

# In[ ]:


''' This will show the columns in the data '''
print('"ship" Dataset Columns: ', *ship.columns + ',')
# The star infront of the ship.columns code prints the list in a cleaner way. 
# Try running the code without the star and you will see what I mean.


# ### <font color="#696969">B. Check Differences  Between Train and Test Data</font>

# In[ ]:


''' This will check which column was removed in the test data '''
column_diff = list(column for column in df_train.columns if column not in df_test.columns) 
##  The style of for loop above is called List Comprehension. It runs much faster than nested for loops and is a must learn for beginners!
print('The only differences between test and train data: ', column_diff)


# ### <font color="#696969">C. First Glance at Train Data</font>
# We can run a quick check on the percentage of people that survived in our Training data to get an idea of around how many should survive in our Test data

# In[ ]:


df_train['Survived'].value_counts(normalize=True) ## Normalize just makes the values into percentages for us


# From the code above, about 38% of passsengers survived and 62% did not.

# #### 1 ) Use "info" Function
# (Code Directory: pandas.DataFrame.info)

# In[ ]:


''' Checking the data types of each column '''
print('------- "ship" Data -------')
ship.info()


# A couple things to notice here:
# * Age, Cabin, Embarked, and Fare
#     * The number of non-null age values is lower than most of the other columns. <br>
#     This means that we have holes in our data that we have to fill
# * object vs float64 vs int64
#     * In order for a ML algorithm to run, all your data must be in the form of numbers.<br> 
#     So our goal is to convert the object columns to either a float or integer datatype.
# * Memory Usage
#     * As you work with larger datasets, the memory usage info tells you if you should<br>
#     maybe trim some of your data. Using memory into the GBs can slow your system down<br>
#     (Note: One GB is 1000 MB and one MB is 1000 KB)

# #### 2 ) Check Columns with Null Values
# This is another method for checking which columns have some holes.

# In[ ]:


print('"ship" columns with null values:\n', ship.isnull().sum())


# ### <font color="#696969">D. Make Original Copy of Data</font>
# Before making changes to your data, make a copy using the code below.<br>
# By having an original copy of your data, changes can be made without any fear of loss of information. <br>
# If you do not understand why, I think you will understand if you make a mistake in the later steps.

# In[ ]:


try:
    ship_orig = ship.copy()
    print('Copies have been made')
except:
    print('Something went wrong.')


# After making a copy, use the code below anytime you need to reset your data back to the original.

# In[ ]:


''' Run this cell if you made a mistake to your data and wish to go back to your original '''
ship = ship_orig.copy()
print('Back up data has been used to restore your data.')


# ### <font color="#696969">E. Drop Columns</font>
# Dropping columns that do not help the analysis can help declutter your data.<br>
# * **Are there any columns in this dataset that can be excluded?**
#     * **PassengerId** - Id numbers may be useful when needing to use SQL to join different datasets together,<br> 
#     but in this case there is only one main dataset to handle. Therefore, PassengerId could be removed to avoid any confusion to the ML algorithm.
#     * **Ticket** - Take a look at the ticket column data. It seems like valuable information could be extracted from here, <br>
#     but it would require more advanced techniques beyond whats covered in this notebook. We will remove this column in this case.

# In[ ]:


# drop unnecessary columns, these columns will not be used in this analysis and prediction
columns_to_drop = ['PassengerId', 'Ticket']
ship = ship.drop(columns_to_drop, axis=1)
print('The following columns have been dropped: ', columns_to_drop)
del columns_to_drop


# **Note:** Just because I dropped these columns does not mean you cannot extract value out of these columns. <br>
# For example, the Ticket column could be analyzed further to see if having a certain ticket type could correlate to who survived.<br>
# It is important to keep an eye for little ideas that could spark value into your analysis.

# ### <font color="#696969">F. "head" and "tail" Function </font>
# The "head" function will print the first 5 rows of the data, unless you specify a number in the parenthesis <br>
# to indicate what row number to stop at from the beginning of the dataset. The "tail" function is basically <br>
# the same as "head" except it starts from the bottom of the data set or the "tail" of the data.

# In[ ]:


ship.head()


# In[ ]:


ship.tail()


# #### What can be seen from this function
# If you are new to Python, you might notice the "NaN" value under the "Cabin" column. "NaN" stands for "not a number".<br>
# Notice how the tail is missing Survived values. This is because we added the test dataset to the end of the train dataset earlier!

# ### <font color="#696969">G. Describe Function</font>
# This function outputs the basic statistics of your numerical data. <br>This function can be re-run as you make edits to your data and can be used to make simple observations.

# In[ ]:


df_train.describe()


# #### What can be seen from this function
# Ok. This looks like a lot, but lets break it down in to small observations:.<br>
# * Survived
#     * Out of 891 people, about 38.38% survived (according to the mean)
# * Pclass
#     * At least 50% of the people were in Pclass 3 (according to 50%)
# * Age
#     * The average age is 29-30 years old (according to mean)
#     * Oldest person was 80 (according to max)
#     * Youngest was less than a year old (according to min)
# * SibSp
#     * A majority of passengers did not have a sibling or spouse (according to 50%)
# * Parch
# * Fare
#     * The lowest ticket cost was 0 (according to min) [Does this value raise an eyebrow?]
#     * The highest ticket cost was 512.33 (according to max)
# 
# 
# <p>
# Are all these observations useful? Probably not, but having these statements in mind can help with the analysis

# ### <font color="#696969">H. Handling Missing Values</font>
# Recall earlier we used the "info" function to display the count of non-null values in each column<br>
# Now it is time to fill in those gaps with different techniques.
# <p>
# In case you are wondering, why would we fill in these holes. When running the code for visualizations, <br>
# some of the code requires all data to be filled in order to run (meaning blanks will cause the code to error out).<br>
# There are other reasons as well which we could get into later.

# #### Code to Check If Column Has Missing Values
# As you are working along in your notebook, come back to the cell below to check your clean up progress.<br>
# The meat of the code below is the text that says: <br>
# `ship[col].isnull().any() == True` 
# 
# What this code is saying is that if **any** value in the column (or **col**) you selected has a **null** value. Print out the column.

# In[ ]:


print('Missing Values in "ship" data: ', list(col for col in ship.columns if ship[col].isnull().any() == True))


# #### Embarked Column
# The technique used to fill in the blanks here is to use the highest occuring value to fill in our blank.

# In[ ]:


ship['Embarked'].value_counts() ## Memorize the "value_counts" function as it is used quite often


# Although this method of filling in the blank with the most frequent value may make our data less accurate,<br>
# the damage should not be significant enough to keep us from going forward.

# In[ ]:


ship['Embarked'].mode()


# Recall from statistics 101, that the mode is the value that occurs the most amount of times in a series.<br>
# So in this case, the mode for the Embarked column is "S".

# In[ ]:


# Fill the two missing values with the most occurred value, which is "S".
ship["Embarked"] = ship["Embarked"].fillna("S")
print('Embarked column blanks have been filled with the value "S"')


# #### Fare Column (Test Data Only)
# For this data, notice the Fare values are scattered all over the place. In fact, lets try making a scatter plot of all the different fares.

# In[ ]:


# matplotlib is a module used for plotting data
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
x_data = ship['Fare']
plt.scatter(x_data.index, x_data)
plt.xlabel('Index Number')
plt.ylabel('Fare Price')
plt.show()


# So notice from this scatter plot that the technique we used for the "Embarked" column may not be as effective than if we took the median value. Are you starting to see how we can use statistics to help us fill in gaps in our data?

# In[ ]:


# We will fill the blank using some estimated guess
median_value = ship["Fare"].median()
print('The median value for the "Fare" column is: ', median_value)
ship["Fare"].fillna(median_value, inplace=True)
print('Fare column blanks have been filled with the median value of the column')
del median_value


# #### Age Column (Both datasets)
# Very similar to what we did with the "Fare" column, we can use the Imputer tool from the Sci-Kit Learn to handle filling in the blanks.<br>
# If you started the Kaggle tutorial for Machine Learning, there is a great lesson on [Imputation](https://www.kaggle.com/dansbecker/handling-missing-values). I will apply the Simple Imputer tool here.<br> 
# All it is doing is filling in the blanks with the mean value of the column.

# In[ ]:


print('Number of null values in Age column:', ship['Age'].isnull().sum())
print('The mean of the Age column: ', ship['Age'].mean())


# In[ ]:


## This will import the Imputer module from Sci-Kit Learn into our Kernel
from sklearn.impute import SimpleImputer
imputer_tool = SimpleImputer()

# Lets make a temporary variable with what will be imputed
ship_numeric = ship.select_dtypes(exclude=['object'])
# Note that Imputation works only with numerical type of data (i.e. Age, Weight, Temperature, etc.)
cols_with_missing = ['Age']

for col in cols_with_missing:
    ship_numeric.loc[:, col + '_was_missing'] = ship_numeric[col].isnull()

# Imputation
ship_numeric_imp = imputer_tool.fit_transform(ship_numeric.values)
ship_numeric = pd.DataFrame(ship_numeric_imp, index=ship_numeric.index, columns=ship_numeric.columns)
ship_numeric.head()


# You may notice some warnings above the table, if you did not run the "warnings off" code near the top.<br>
# No need to worry for now. From to time to time, new updates to modules will require you to adjust your code accordingly.<br>
# Moving on, below will give us a peak into what our Imputer tool filled the blanks with.

# In[ ]:


ship_numeric.loc[lambda df: df.Age_was_missing == 1, :][:5] ## This pulls up a sample of the records that had the age value missing.


# Nice! The blank age values were filled with the mean calculated earlier.<br>
# Now lets bring the data back to the main dataframe.

# In[ ]:


ship['Age'] = ship_numeric['Age'].copy()
print('Are there any null values?', ship['Age'].isnull().any())
ship['Age'] = ship_numeric['Age'].copy()
ship['Age_was_missing'] = ship_numeric['Age_was_missing'].copy() # We will use this later
del ship_numeric, ship_numeric_imp # We don't need these variables anymore.
print('Age data has been copied back to main dataset')
print('Age_was_missing column was added to main dataset')


# ### Cabin Column
# Lets take a look at this data

# In[ ]:


print('NaN "Cabin" values in "ship" dataset: %s out of %s' % (ship['Cabin'].isnull().sum(), len(ship)))


# Now, there's a decision that should be made here. <br>
# With a majority of the data missing from this column, <br>
# should we drop this data from our dataset or try to find some value from this data?<br>Lets try to run one more test.

# In[ ]:


cabin_data = {}
cabin_data['With Cabin Name - Survived'] = ship['Cabin'].loc[ship['Cabin'].notnull() & (ship['Survived'] == 1)].count()
cabin_data['With Cabin Name - Deceased'] = ship['Cabin'].loc[ship['Cabin'].notnull() & (ship['Survived'] == 0)].count()
cabin_data['No Cabin Name - Survived'] = len(ship['Cabin'].loc[ship['Cabin'].isnull() & (ship['Survived'] == 1)])
cabin_data['No Cabin Name - Deceased'] = len(ship['Cabin'].loc[ship['Cabin'].isnull() & (ship['Survived'] == 0)])

plt.bar(list(cabin_data.keys()) ,(list(cabin_data.values())))
plt.xticks(rotation='vertical')
plt.show()
cabin_data


# In[ ]:


cabin_notnull = df_train['Cabin'].loc[df_train['Cabin'].notnull()].astype(str).str[0]
cabin_notnull = pd.DataFrame([cabin_notnull, df_train['Survived'].loc[df_train['Cabin'].notnull()]]).T


# In[ ]:


cabin_notnull['Cabin'].value_counts()


# In[ ]:


cabin_values = list(cabin_notnull['Cabin'].unique())
cabin_values
cabin_sums = [((cabin_notnull['Survived'].loc[cabin_notnull['Cabin'] == x].sum()) / (len(cabin_notnull.loc[cabin_notnull['Cabin'] == x]))) for x in cabin_values]
cabin_sums

fig, ax=plt.subplots()
ax.set(xlabel="Cabins", ylabel="Survival Rate")
ax.bar(cabin_values, cabin_sums)
plt.show()
print(*[('Survival rate for cabin %s:  %s \n' % (x, round(y, 2))) for x, y in zip(cabin_values, cabin_sums)])


# Hmm. Notice from above that the people which had Cabin values were more likely to survive than those who had null values for the Cabin column.
# <p>
# In my previous versions of this notebook, I dropped the Cabin column from the analysis<br>
# Then, after running some tests with the Cabin column included. It was not able to raise my score.<br>
# After reviewing other notebooks as well, they say that the Cabin column just does not have enough <br>
# data to improve our predictions.
# <p>
# Now, I'm back to dropping the Cabin column.

# In[ ]:


#ship['Cabin_bool'] = ship["Cabin"].notnull().astype('int')


# In[ ]:


ship.drop("Cabin", axis=1, inplace=True)
print('Cabin column has been dropped.')


# ### Before Moving Forward
# Take some time to revisit some of the introductory code and maybe re-run a few of them to see how everything changed.<br>
# You can go back to the "info" or "describe" function section and check if all the holes have been filled.

# <a id="3-clean"></a>
# ## Now Time to Clean Up Data
# Wow that was a lot of prep work! Definitely a necessary step. <br>
# After filling in the gaps, there is still much cleaning to do. The cleaning is required because the machine learning tools cannot run with "categorical" information in the data. <br>
# The machine will basically say, what do I do with this unclassified data? This is why the clean up section is the tedioius but very important.

# ### Clean-up: Embarked Column
# Dummy method. The dummy method simply creates a column for each option in a categorical column.<br>
# So if you recall, we had 3 different options for the Embarked column. What the dummies method will do<br>
# is create three separate columns for each option and simply put a 1 under which option that particular <br>
# record belongs to and put a 0 for the other two columns. This is so we can convert our object data column<br>
# to integer style columns that we can feed into the ML algorithm!

# In[ ]:


embark_dummies_titanic  = pd.get_dummies(ship['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

ship = ship.join(embark_dummies_titanic)

ship.drop(['Embarked'], axis=1,inplace=True)
print ('Embarked column has been dropped. C and Q columns have been added.')


# ### Clean-up: Cabin Column
# Below is the methodology I used to clean up the Cabin column.  I took this code out of the analysis since the data did not help despite showing a high percentage of survival for those with cabin values.
# ```
# ship['Cabin'].head(10)
# def get_cabin(cab_value):
#     if pd.isnull(cab_value):
#         return 0
#     elif cab_value[0].find('B') == 0:
#         return 'B'
#     elif cab_value[0].find('D') == 0:
#         return 'D'
#     elif cab_value[0].find('E') == 0:
#         return 'E'
#     else:
#         return 0
# ship['Has_cabin'] = ship['Cabin'].apply(get_cabin)
# ship['Has_cabin'].head(10)
# ship.drop(['Cabin'],axis=1,inplace=True)
# cabin_dummies = pd.get_dummies(ship['Has_cabin'])
# cabin_dummies.head()
# cabin_dummies.drop(0, axis=1, inplace=True)
# cabin_dummies.head()
# ship = ship.join(cabin_dummies)
# ship.drop(['Has_cabin'], axis=1, inplace=True)
# print('Cabin columns have been added.')
# ```

# ### Clean-up: Pclass Column
# Dummy method. 

# In[ ]:


# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic  = pd.get_dummies(ship['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

ship.drop(['Pclass'],axis=1,inplace=True)

ship = ship.join(pclass_dummies_titanic)
print ('Pclass column has been changed.')


# ### Feature Engineering: Creating a Family Column
# Instead of having two columns Parch & SibSp, we can combine them into one column <br>
# representing if the passenger had any family member aboard or not.<br>
# This will make the ML algo decide if having any family member will increase chances of survival.

# In[ ]:


ship['Family'] =  ship["Parch"] + ship["SibSp"] # Adding together Parch and SibSp columns
ship['Family'].loc[ship['Family'] > 0] = 1
ship['Family'].loc[ship['Family'] == 0] = 0

'''Notice here: We have created a new column called "Family" and given it the values of only 1 or 0'''
'''When you create new columns of data, this is called Feature Engineering'''

# drop Parch & SibSp
ship = ship.drop(['SibSp','Parch'], axis=1)

print('SibSp and Parch columns have been dropped. Family column has been added.')


# ### Feature Engineering: Sex Column
# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.<br>
# So what the code below is doing is taking the Age and Sex data and making a new <br>
# column called "Person" which will contain the options: child, male, or female.

# In[ ]:


def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 18 else sex
    
ship['Person'] = ship[['Age','Sex']].apply(get_person, axis=1) # The apply function will run the get_person function to the 'Age' and 'Sex' columns

# No need to use Sex column since we created Person column that contains the Sex of the passenger
ship.drop(['Sex'],axis=1,inplace=True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(ship['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

ship = ship.join(person_dummies_titanic)

# Now we can drop the Person column we created, since we converted it to a numerical datatype using the dummy method
ship.drop(['Person'],axis=1,inplace=True)
print('Age and Sex columns have been dropped. Child and Female columns have been added.')


# ### Feature Engineering: Name Column

# In[ ]:


ship['Title'] = ship['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
print('Title column has been created with values')
ship.head()


# In[ ]:


ship['Title'].value_counts()


# In[ ]:


titles = ship['Title'].unique()
titles


# In[ ]:


#titleGenderAge = pd.DataFrame(index = titles, columns = ['Gender', 'Min Age', 'Median Age', 'Max Age', 'Count'])
ship.groupby('Title', sort=False)['Age'].agg(['mean', 'min', 'median', 'max', 'count'])


# Ok. Something to notice here is that we now have a general idea of the age of what a person would be by simply knowing the title.<br>
# This means we can revisit our Age column values that were null and where we put in the mean Age of the whole dataset and <br>
# try to make a more accurate guess as to what the age of the passenger was.
# 

# ### Cleanup: Lumping the Rare Title of Ladies together with Mrs and Miss

# In[ ]:


ship['Title'].loc[ship['Title'] == 'the Countess'] = 'Mrs'
ship['Title'].loc[ship['Title'] == 'Ms'] = 'Mrs'
ship['Title'].loc[ship['Title'] == 'Lady'] = 'Mrs'
ship['Title'].loc[ship['Title'] == 'Dona'] = 'Mrs'
ship['Title'].loc[ship['Title'] == 'Mlle'] = 'Miss'
ship['Title'].loc[ship['Title'] == 'Mme'] = 'Miss'
print('Rare lady titles have been added to larger group titles')


# 
# ### Cleanup: Combining Rare Title Names as One Option

# In[ ]:


print(ship['Title'].value_counts())
stat_min = 10 # while small is arbitrary, we'll use the common minimum in statistics: 
              # http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
title_names = (ship['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

# apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: 
# https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
ship['Title'] = ship['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(ship['Title'].value_counts())
print("-"*10)
print('Rare title names have been combined into a "Misc" option')


# ### Label Encoder
# Similar to the get dummies method, the label encoder will instead make one column with a number value for each option.<br>
# So a title of "Mr" will be for example number 1 and a "Mrs" title will be value of 2 and so on.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics


# In[ ]:


label = LabelEncoder()
ship['Title_Code'] = label.fit_transform(ship['Title'])
print('Fit Transform function has been run.')
ship['Title_Code'].head()


# In[ ]:


ship.drop(['Name'],axis=1,inplace=True)
print('Name column has been dropped.')


# In[ ]:


title_data = {}
title_data['Mr - Survived'] = ship['Title_Code'].loc[(ship['Title_Code'] == 3) & (ship['Survived'] == 1)].count()
title_data['Mr - Deceased'] = ship['Title_Code'].loc[(ship['Title_Code'] == 3) & (ship['Survived'] == 0)].count()
title_data['Miss - Survived'] = ship['Title_Code'].loc[(ship['Title_Code'] == 2) & (ship['Survived'] == 1)].count()
title_data['Miss - Deceased'] = ship['Title_Code'].loc[(ship['Title_Code'] == 2) & (ship['Survived'] == 0)].count()
title_data['Mrs - Survived'] = ship['Title_Code'].loc[(ship['Title_Code'] == 4) & (ship['Survived'] == 1)].count()
title_data['Mrs - Deceased'] = ship['Title_Code'].loc[(ship['Title_Code'] == 4) & (ship['Survived'] == 0)].count()
title_data['Master - Survived'] = ship['Title_Code'].loc[(ship['Title_Code'] == 0) & (ship['Survived'] == 1)].count()
title_data['Master - Deceased'] = ship['Title_Code'].loc[(ship['Title_Code'] == 0) & (ship['Survived'] == 0)].count()
title_data['Misc - Survived'] = ship['Title_Code'].loc[(ship['Title_Code'] == 1) & (ship['Survived'] == 1)].count()
title_data['Misc - Deceased'] = ship['Title_Code'].loc[(ship['Title_Code'] == 1) & (ship['Survived'] == 0)].count()

plt.figure(figsize=(16,4))
plt.bar(list(title_data.keys()) ,(list(title_data.values())))
plt.show()
title_data


# Notice that the categories that had a majority of survivors were the Master, Miss, and Mrs titles.

# In[ ]:



title_dummies_titanic  = pd.get_dummies(ship['Title'])
title_dummies_titanic.drop(['Misc', 'Mr'], axis=1, inplace=True)

ship.drop(['Title', 'Title_Code'],axis=1,inplace=True)

ship = ship.join(title_dummies_titanic)
print ('Title column has been dropped. "Master", "Mrs", and "Miss" has been added.')


# In[ ]:


ship.sample(5)


# ## 4. Explore Data
# Exploring the data will help you get a deeper understanding of the data you are working with. 
# When I first got started, I got intimidated with all these fancy looking graphs and charts.
# Just know that these visuals are just supplements to help you refine your model. 
# So while you may have a simple model right now. Just keep going in your studies to get better,
# then the fancy graphs will start to make sense.
# <p>
# Most likely after exploring the data further, you will have to go back into the previous step to clean up and prep the data to get a more accurate model. This step should not prevent you from moving forward into creating a solution and submitting a model for a score. Even though you might get a low score at first, keep trying to make adjustments to see if you can improve your score.

# In[ ]:


ship.sample(5)


# ### Exploring the Fare Data

# Lets take a look to see if having a higher fare value will increase the odds of living

# In[ ]:


print('Number of  people who survived with a Fare > 50: ',
      len(ship['Survived'].loc[(ship['Fare'] > 50) & (ship['Survived'] == 1)]))
print('Number of  people who died with a Fare > 50: ',
      len(ship['Survived'].loc[(ship['Fare'] > 50) & (ship['Survived'] == 0)]))


# Maybe we can add a new feature here with the Fare column

# In[ ]:


ship['High_Fare'] = [1 if x > 50 else 0 for x in ship['Fare']]
print('High Fare column created.')


# ### Exploring the Age Data

# In[ ]:


ship.loc[ship['Age_was_missing'] == 1].sample(5)


# The goal will be to re-impute the age values with the appropriate mean based on the title of the person. (Still a work in progress!)

# In[ ]:


ship.drop(['Age_was_missing'],axis=1,inplace=True)
print('Dropped "Age_was_missing" column from dataset.')


# ### Split Training and Test Data

# ### Correlation Heatmap of Dataset
# This is a very useful table that shows how all the variables are correlated to each other, espcially which variables are correlated to "survival".<br>
# The higher the number, the more correlated the variables are to each other. We are basically looking to use variables that have a strong <br>
# correlation to predicting if survival will be a 1 value.

# In[ ]:


import seaborn as sns
#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(ship)


# Interestingly, the "age" variable has a negative correlation to survival. Try running the Kernel with the Age column, then without (by removing the comment hashtags below).

# In[ ]:


ship.drop(['Age'],axis=1,inplace=True)


# <a id="model"></a>
# ## 5. Model Solution

# ### 5.A. Import Tools

# In[ ]:


# Machine Learning Tool called Sci-Kit Learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ### Define Train and Test Datasets

# In[ ]:


df_tr = ship[ship.Survived.notnull()]
df_tr.head()


# In[ ]:


df_te = ship[ship.Survived.isnull()]
df_te.drop(['Survived'], axis=1, inplace=True)
df_te = df_te.reset_index(drop=True)
df_te.head()


# In[ ]:


x_train = df_tr.drop("Survived",axis=1)
y_train = df_tr["Survived"]
x_test  = df_te.copy()
print('Test and Train ML variables are ready.')


# ### Logistic Regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
logreg.score(x_train, y_train)


# ### Random Forests

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=300)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test).astype(int)
random_forest.score(x_train, y_train)


# What this shows, is the performance of Random Forests after being trained with the x_train data.

# ## 6. Clean Up Solution

# ### Get Correlation Coefficient

# In[ ]:


df_coeff = pd.DataFrame(ship.columns.delete(0))
df_coeff.columns = ['Features']
df_coeff["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
df_coeff


# In[ ]:


df_test['Survived'] = y_pred.astype(int)
df_test.to_csv('SurvivedList.csv')
df_test


# ## 7. Submit Results
# Time to decide which predictions you wish to use for your submission for a score.

# In[ ]:


submission = df_test[["PassengerId", "Survived"]]


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('titanic.csv', index=False, header=['PassengerID', 'Survived'])


# ## What's Next?
# Well you did it! You made it through **one** cycle of the process of creating a ML model.<br>
# Too bad the first time around usually doesn't score very high. Time to go back to the beginning and tweak the model to see if you can make it better!<br>
# This is the fun part about data science. Its like a puzzle that you can keep rotating puzzle pieces to see if it works better than before.<br>
# Have fun and hope you learned something!

# # <center> Under Construction. <br>Still more to come! </center>

# ## Change Log:
# * 7/3/2018 : Notebook created
# * 7/4/2018 : Finished Imputer section for Age column
# * 7/11/2018 : Added the Title cleanup section. (Sourced from ldfreeman3's notebook below). Need to add more to Exploring Data section.
# * 7/19/2018: Added new technique of working with a combined dataset instead of two separated test and train datasets.
# * 7/26/2018: Added Cabin column into analysis! Still have to work on the Age column for better results.
# * 7/30/2018: Took out Cabin column from analysis after trying several methods to raise score. There just isn't enough Cabin data!

# ## Sources of Information:
# I would not be able to post this notebook without the help of the following sources.<br>
# I did copy some of their code over to this notebook and made some changes to suit my style.<br>
# Please take a look at these other great notebooks for more insight and information!
# * Other Titanic Kernels:
#     * [User: omarelgabry / Notebook: A Journey Through Titanic](https://www.kaggle.com/omarelgabry/a-journey-through-titanic)
#     * [User: zlatankr / Notebook: Titanic Random Forest](https://www.kaggle.com/zlatankr/titanic-random-forest-82-78)
#     * [User: ldfreeman3 / Notebook: A Data Science Framework: To Achieve 99% Accuracy](https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy) <-- High Scoring Notebook

# In[ ]:




