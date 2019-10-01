#!/usr/bin/env python
# coding: utf-8

# Credit goes to https://github.com/mikemac8888/pythonbrno

# # Python Brno - Part 1A - Python
#  
# 
# ## Workshop Outline
# 
# 
# ### Part 1A - Python
# 1. Data Structure Review
# 
# ### Part 1B - Pandas
# 1. Exploring the titanic data
# 2. Selection
# 3. Filtering
# 4. More functions
# 5. Intro to groupby

# ### Questions
# - Can you hear me in the back?
# - Am I speaking too fast for anybody?
# 
# 
# - Who's studying Computer Science?
# - Who works as a programmer?
# - Who works as a python programmer?
# - Who's used pandas?
# 
# 
# - Who has been programming in python for less than 2 years?

# ### Warm up
# 
# - Shake hands & Hello

# ### Jupyter Introduction
# 
# - Blue border: Command mode
# - Green border: Edit mode
# - Esc: Switch from edit mode to command mode
# 
# #### Command mode:
# - Shift + Enter: Run cell and move cursor to next cell
# - Ctrl + Enger: Run cell but keep cursor on current cell
# 
# 
# - a: add cell above this cell
# - b: add cell below this cell
# 
# 
# - x: delete this cell
# - z: undo delete
# 
# - o: hide cell output

# ### Data Structure Review
# - Reference: https://docs.python.org/3/tutorial/datastructures.html

# #### Lists - Mutable sequences of items

# In[ ]:


fruits = ['orange', 'apple', 'pear', 'banana', 'apple', 'banana']


# In[ ]:


# Calculate the length of fruits


# In[ ]:


# Insert 'kiwi' into fruits and show the result without using print()


# In[ ]:


# Create a new list with the first 3 elements of fruits


# In[ ]:


# Create a new list with the last 2 elements of fruits


# In[ ]:


# %load solutions/1-1.py


# Dictionary Example

# In[ ]:


czech_ryanair = {
    'BRQ' : ['LTN'],
    'PRG' : ['CRL', 'CIA', 'BGY', 'TPS','DUB','STN','LPL'],
    'OSR' : ['BGY', 'STN'],
}


# In[ ]:


# Iterate through czech_ryanair printing the keys and values
for key, value in czech_ryanair.items():
    print('Key: {}, Value: {}'.format(key, value))


# Dictionary Exercises

# In[ ]:


# Which destinations are accessible from Prague (PRG)?


# In[ ]:


# Construct a list of all the destinations ryanair flies to from czech republic


# In[ ]:


# %load solutions/1-2.py


# #### Set - Unordered collection of unique hashable items

# In[ ]:


john_classes = {'monday', 'tuesday', 'wednesday'}
eric_classes = set(['wednesday', 'thursday'])


# In[ ]:


john_classes


# In[ ]:


eric_classes


# In[ ]:


union = john_classes | eric_classes
union


# Set Exercises

# In[ ]:


# Print out a list of all the unique destinations ryanair flies to from czech republic


# In[ ]:


# %load solutions/1-3.py


# # Python Brno - Part 1B - Pandas

# ### Pandas
# 
# - Provides high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
# 
# 
# - `pandas.DataFrame`
#   - 2D size-mutable data structure with labeled row index and labeled column index.
# - `pandas.Series`
#   - 1D size-mutable data structure with labeled row index and labeled column.
# 
# 
# - Fantastic documentation: http://pandas.pydata.org/pandas-docs/stable/

# ### The Titanic Survival Problem
# 
# Predict whether a passenger on the titanic will survive. 
# - Input: information about each passenger
# - Output: whether or not the passenger survived

# ### Exploring the Titanic Data

# Load the pandas library with alias `pd`

# In[ ]:


import pandas as pd
pd.options.display.max_rows = 8


# Load matplotlib

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# Read the data csv file into a pandas DataFrame

# In[ ]:


# Need to add data via Add Data->COmpetitions-->Titanic


# In[ ]:


df = pd.read_csv('../input/train.csv')


# See what's inside the DataFrame

# In[ ]:


df


# In[ ]:


df.Age.hist()


# In[ ]:





# Typing `df.` and pressing `Tab` we can see the methods available on this DataFrame

# In[ ]:


# df.


# ### Exercises
# 
# 1. Write each of these expressions in a separate cell and interpret what they do
# ```python3
# df.columns
# df.head()
# df.tail()
# df.shape
# df.info()
# df.dtypes
# ```
# 2. What is the type of each of the expressions above?
# 3. There's a function which provides summary statistics for a pandas DataFrame. Do a google search for 'pandas summary statistics', find out what the function name is and execute this function in a new cell.
# 4. Interpret each line of the output from this summary statistics function.

# In[ ]:





# ### Selection
# #### Column Selection

# In[ ]:


df['Name']


# In[ ]:


df.Name


# What is the type of this column?

# In[ ]:





# In[ ]:


# Try to select multiple columns


# #### Row Selection - Integer Index

# In[ ]:


df.iloc[0]


# #### Row Selection - Label Index 1

# In[ ]:


df.loc[0, :]


# What is the type of this row?

# In[ ]:





# #### Row Selection - Label Index 2

# In[ ]:


import numpy as np

# Date range creation function
i = pd.date_range(pd.Timestamp('2016-01-01'), pd.Timestamp('today'))

# Create a dataframe with a series of random numbers
date_df = pd.DataFrame(np.random.randn(len(i)), columns=['random'], index=i)

date_df.head()


# In[ ]:


date_df.loc['2016-01-04', :]


# We can do magical things with label based indexes (month based selection)

# In[ ]:





# #### Row + Column Selection

# In[ ]:


df.loc[5:8,'Fare']


# In[ ]:


# df.loc[0:5, 'PassengerId', 'Name']


# In[ ]:


df.loc[0:5, ['PassengerId', 'Name']]


# What is the type of this selection?

# In[ ]:





# In[ ]:





# In[ ]:





# ### Exercise
# 1. Select the ticket column of passengers 100 to 200 inclusive

# In[ ]:





# In[ ]:





# In[ ]:





# ### Filtering

# Determine which rows have value `male` in the `Sex` column.

# In[ ]:


df.Sex == 'male'


# What is the type of this column?

# In[ ]:


type(df.Sex == 'male')


# To select all the rows with Sex value male use the square bracket operator.

# In[ ]:


df[df.Sex == 'male']


# Compose complex filters using the `&` operator.

# In[ ]:





# ### Exercises
# 
# 1. Use the `>=` operator to select all passengers older than 40
# 1. Select all female passengers who survived and are between 20 and 30

# In[ ]:





# In[ ]:





# In[ ]:





# ### More functions

# In[ ]:


df.Sex.unique()


# In[ ]:


df.isnull()
#df[df.isnull().any(axis=1)]


# In[ ]:


df.Sex.value_counts()


# In[ ]:


df.count()


# ### Exercises

# How many people were in second class?

# In[ ]:





# How many different classes were there?

# In[ ]:





# In[ ]:


# %load solutions/1-6.py


# ### Group By: split-apply-combine
# - Reference: http://pandas.pydata.org/pandas-docs/stable/groupby.html
# 
# By "group by" we are referring to a process involving one or more of the following steps
# 
# - Splitting the data into groups based on some criteria
# - Applying a function to each group independently
# - Combining the results into a data structure

# In[ ]:


pd.options.display.max_rows = 20


# In[ ]:


df.groupby('Sex').count()


# What was the average age of females and males who survived?

# In[ ]:


# %load solutions/1-7.py


# In[ ]:





# In[ ]:





# In[ ]:





# # Python Brno - Part 2B - Pandas

# Notes:
# - Next level Pandas
#   - https://github.com/TomAugspurger/modern-pandas

# ### Pandas
# 
# - Provides high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
# 
# 
# - `pandas.DataFrame`
#   - 2D size-mutable data structure with labeled row index and labeled column index.
# - `pandas.Series`
#   - 1D size-mutable data structure with labeled row index and labeled column.
# 
# 
# - Fantastic documentation: http://pandas.pydata.org/pandas-docs/stable/

# ### The Titanic Survival Problem
# 
# Predict whether a passenger on the titanic will survive. 
# - Input: information about each passenger
# - Output: whether or not the passenger survived
# 
# The data we will use is located in the file `titanic_data.csv` and is similar to the set from https://www.kaggle.com/c/titanic/data
# 
# 
# 
# 

# ### Exploring the Titanic Data

# Load the pandas library with alias `pd`

# In[ ]:


import pandas as pd
pd.options.display.max_rows = 8


# Load matplotlib

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# Read the data csv file into a pandas DataFrame

# In[ ]:


df = pd.read_csv('../input/train.csv')


# ### Missing Data

# In[ ]:


df.info()


# ### The components of a DataFrame

# In[ ]:


df.values


# In[ ]:


df.index


# In[ ]:


df.columns


# ### More on groupby / stack / unstack
# 
# It's possible to groupby multiple variables.

# What's the difference in the fare between 1st class and 3rd class for females and males?

# In[ ]:


df.groupby(['Sex','Pclass'])['Fare'].mean()


# Here we want the difference between the first row and third row and the difference between the fourth row and sixth row. It's ugly to write code specifically targeting these rows. Instead we can transform the data.

# ### Use `unstack` to pivot index labels into column labels

# In[ ]:


df.groupby(['Sex','Pclass'])['Fare'].mean().unstack()


# ### Use `stack` to pivot column labels into index labels

# In[ ]:


df.groupby(['Sex','Pclass'])['Fare'].mean().unstack().stack()


# Now we just need to subtract the two columns to get our answer

# In[ ]:


avg_fare_groupedby_sex_v_class = df.groupby(['Sex','Pclass'])['Fare'].mean().unstack()
avg_fare_groupedby_sex_v_class


# In[ ]:


abs(avg_fare_groupedby_sex_v_class.loc[:,3] - avg_fare_groupedby_sex_v_class.loc[:,1])


# ### Exercise

# What was the average age of females and males who survived?

# In[ ]:





# In[ ]:





# In[ ]:





# ### Setting
# 
# Suppose we want to store this calculation in a new column.

# In[ ]:


avg_fare_groupedby_sex_v_class.loc[:,'abs_diff_1_3'] = abs(avg_fare_groupedby_sex_v_class.loc[:,3] - avg_fare_groupedby_sex_v_class.loc[:,1])
avg_fare_groupedby_sex_v_class


# Question: Does this change our original dataframe?

# In[ ]:





# In[ ]:





# In[ ]:





# ### Sorting

# In[ ]:


df.sort_index(axis=0, ascending=False)


# In[ ]:


df.sort_values(by='Pclass')


# In[ ]:





# ### Missing Data

# In[ ]:


df.info()


# To clear out any rows with missing data call `dropna`

# In[ ]:


df.dropna().info()


# Unfortunately this removes 80% of the observations in our dataset.

# In[ ]:


len(df.dropna()) / len(df)


# Instead we choose reasonable filler values for missing data based on inference or statistics.
# 
# Suppose we knew for example that any unspecified Cabin data meant that the passengers were staying in the Dorm room.

# In[ ]:


df.Cabin


# In[ ]:


df.Cabin.fillna(value='Dorm')


# ### Exercise
# 
# Two rows are missing data in the Port (Embarked) column. Talk with your neighbors about which port would be most appropriate to replace the missing data and then execute the appropriate command.
# 
# 1. Feed `df.Embarked.isnull()` into `df[  ]` as a filter see what the missing data rows are
# 2. Use `value_counts` to determine which values are the most common

# In[ ]:





# In[ ]:





# In[ ]:





# ### Manipulating data
# 
# - `apply` - execute function on a row / column of a DataFrame
#   - row based: df.apply(fn, axis=0)   # default
#   - row based: df.apply(fn, axis=1)
# - `applymap` - execute function elementwise on a DataFrame
# - `map` - execute function elementwise on a Series

# In[ ]:


import numpy as np


# In[ ]:


df_age_fare = df[['Age','Fare']]
df_age_fare.describe()


# **Here it's getting quite hard**

# Suppose we want to normalize some data between the values of 0 and 1. We can use a lambda function and `apply`

# In[ ]:


df_age_fare = df[['Age','Fare']]
df_norm_1 = df_age_fare.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
df_norm_1.describe()


# We can declare a separate function and pass it to `apply`

# In[ ]:


df_age_fare = df[['Age','Fare']]

def my_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

df_norm_2 = df_age_fare.apply(my_norm)
df_norm_2.describe()


# Or we can use pandas built in functions to get the same result

# In[ ]:


df_norm_3 = (df_age_fare - df_age_fare.min()) / (df_age_fare.max() - df_age_fare.min())


# In[ ]:


df_norm_3.describe()


# In[ ]:


df_age_fare.hist();
df_age_norm_3.hist();


# ### Exercise
# 
# 1. Determine how many rows are missing from the Age column
# 2. Plot the histogram of Age
# 3. It's common practice to fill in missing data with the mean of the variable.
# 4. Create a new column called age_filled_with_mean and set it's value to Age with missing items replaced by the mean
# 5. Plot the histogram of Age vs the histogram of age_filled_with_mean
# 6. Discuss with your neighbour whether this is a good or bad approach to filling in the missing data

# In[ ]:





# In[ ]:





# In[ ]:





# ### Time Series / rolling functions

# In[ ]:


# http://stackoverflow.com/questions/16734621/random-walk-pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def geometric_brownian_motion(T = 1, N = 100, mu = 0.1, sigma = 0.01, S0 = 20):        
    dt = float(T)/N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N) 
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    X = (mu-0.5*sigma**2)*t + sigma*W 
    S = S0*np.exp(X) ### geometric brownian motion ###
    return S

dates = pd.date_range('2012-01-01', '2016-02-22')
T = (dates.max()-dates.min()).days / 365
N = dates.size
start_price = 100
y = pd.Series(geometric_brownian_motion(T, N, sigma=0.1, S0=start_price), index=dates)
y.plot()
# plt.show()


# In[ ]:


type(y.index)


# In[ ]:


y


# Since we are using a DatetimeIndex we can slice it based on month and year

# In[ ]:


y.loc['2014-10']


# In[ ]:


y.loc['2014']


# To calculate a moving average we use rolling

# In[ ]:


y.rolling(window=30).mean().plot()


# In[ ]:


y.plot()
y.rolling(window=30).mean().plot()


# ### Exercise
# 
# Write a function which takes a DataFrame and a list of moving averages and returns the DataFrame with each of the moving averages calculated in a separate column

# In[ ]:


moving_averages = [30, 60, 200]

def calculate_moving_averages(df, moving_average_list):

    # Fill me in
    
    return df

# Uncomment the next line
# calculate_moving_averages(y, moving_averages)


# In[ ]:




