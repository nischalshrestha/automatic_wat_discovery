#!/usr/bin/env python
# coding: utf-8

# # Introduction
# I was looking for something a bit more interactive, and really like the interface here on Kaggle. So I thought I would build one  as a learning exercise. With my technical background in development and training, I thought this might be worth sharing. So I am making this public to see if anyone else finds it useful.
# 
# The goal here is to introduce pandas and related tools useful for machine learning, rather than a focus on the competition specifically. This notebook focuses on the DataFrame, the core table object in pandas. Appreciate any feedback you may have, and if this turns out to be useful I will try to create more of these.

# # Getting Started
# This notebook assumes you have explored kaggle enough to know what a notebook is and how to run through it. You can hit "Shift+Enter" as a keyboard shortcut to execute the current cell and move on to the next one. This is meant to be an interactive exercise, so I would recommend using it as follows:
# 1. Scan through the notebook top to bottom to get a sense of what is here.
# 1. If it looks interesting, give it a "thumbs up" to say you like it.
# 1. Work through the notebook and exercises. You can do this  separately using the Titanic dataset (in another tab, I would suggest), or simply press Fork Notebook at the top to create your own copy of this kernel.
# 
# There are a number of exercises, some of them repetative, as my own challenge has been using the tools well enough to understand and remember them later. It is one thing to read a guide or tutorial, and perhaps do a question or two. It is quite another to use the tools directly. Hopefully this hands-on approach will help you (and me) understand basic machine learning in a more logical way.
# 
# As you may be aware, most competition kernals start out with the below cell. Executing this cell shows the training and test sets for the competion, along with some tools that might be useful.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# We can load these into a pandas DataFrame with the `read_csv` method. Pandas includes a number of `read_` methods, from fixed width and json formats to Excel to sqlite files. We won't go into these here, though it is good to know they exist.

# In[ ]:


# Execute the prior cell before this one to load pandas and other modules
training_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
training_df.head(3)


# As you can see from these rows, some information is numerical, and some is textual. Generally, well-defined numerical values are needed for the machine learning models, so let's explore how we interact with DataFrame objects to both understand them in more detail and alter their contents.

# # Exploring DataFrames
# There are a number of ways to explore a DataFrame. Exploring is the recommended first step in analyzing a data set. The goal is to gain an understanding of the data, know what you have, what you might be missing, and what might be relevant for predictions.
# 
# A DataFrame is just a table. There are rows, columns, headings, indexes. If you just type the DataFrame name `training_df` you will see the entire table. Kaggle notebooks automatically truncate really long lists, although the Titanic data isn't that big so it may show the entire table. Try it out.

# In[ ]:


# Type the training_df DataFrame name here to see what it does


# ## DataFrame Attributes
# 
# There are a number of attributes that charactize the data, including the following:
# * `axes` returns the dimensional axes of the data, typically the index and columns. For the Titanic training data, this will show  an index with 891 entries and 12 columns.
# * `columns` returns the names of the columns in the table.
# * `dtypes` returns the type for each columns.
# * `index`returns the current index.
# * `ndim` returns the number of dimensions.
# * `shape` returns the size of the data as a tuple showing rows and columns. For the Titanic training data this is (891, 12).
# * `size` returns the total number of individual entries in the table.
# * `values` returns  all values in the table as an array.
# 
# Here is an example that shows the columns in the table. Execute this cell to see the columns for our table.

# In[ ]:


training_df.columns


# ### Attributes Exercises
# To become more familiar with these attributes, answer the following questions about the training data.
# 1. How many dimensions are in the training data? (Use `ndim`)
# 2. How does the `axes` attribute compare with the `index` and `columns` attributes? Execute these three attributes to determine how they differ or are similar.
# 3. If you multiply the number of rows and columns together (from `shape`) does it equal the total number of entries (from `size`)?
# 4. What is the data type of the PassengerId column? (hint: use `dtypes`)
# 5. What is the data type of the Age column?
# 1. What is the `shape` and `size` of the test data `test_df`?
# 1. What column is missing from the test data that is in the training data?
# 1. Pick one other attribute and execute it on the test data to see the results.

# In[ ]:


# Code area for Attributes Exercises


# ## DataFrame Columns
# 
# Pandas does a lot of work under the covers to simplify looking at row and column data. You can access any column as a `Series` using  brackets or as an attribute. The DataFrame converts an attribute name into a column automatically. So `training_df['Fare']` and `training_df.Fare` both return the Fare column. Column names are case-sensitive, so Pclass and pclass are not the same. So the passenger class column is accessible with `training.Pclass` or `training['Pclass']` but both `training.pclass` and `training.['pclass']` result in an error.
# 
# ### Column Exercises
# These will make more sense if you try the following exercises:
# 1. Another feature of pandas is to construct columns using the plus (+) operator. Use this to verify  that `training_df.Ticket` and `training_df['Ticket']` return the same series by adding these together, placing the string `' equals '` in between them.  That is, use the code `training_df.Ticket + ' equals ' + training_df['Ticket']` to see the two values side by side.
# 2. Previously we noticed that the Age column is a float type. Display this column to see that the values are indeed decimal values.
# 3. Does the Cabin column have any null, or missing, values? Display this column to see.  Null values are shown as `NaN` in the list.
# 4. What are the two values used in the Sex column?
# 5. What happens if you enter an invalid name? Use both `training_df['pclass']` and `training_df.age` to see. Are the error messages the same for both?

# In[ ]:


# Code area for Column Exercises


# ## DataFrame Rows
# Rows are a bit tricker. The `values` attribute we saw previously returns an array of row arrays. So `training_df.values[3]` will show the fourth row (remember, indexes start at 0). There are also two attributes `loc` and `iloc` that access specific locations within the table. The `iloc` attribute is purely integer based (`i` is for integer), while the `loc` attribute is index and column name based (or boolean values, but that is for another time).
# 
# If you look at the table, Age is the sixth column (index 5), so to obtain the Age of the fourth row (index 3) you can do this with any of the following:
# > `training_df.values[3][5]`   
# >  `training_df.iloc[3,5]`   
# >  `training_df.loc[3,'Age']`   
# 
# In the Titanic data the index is the same as the row number, so it isn't immediately clear in this example that the 3 used in `iloc` and `loc` and very different.  We'll see a better example of this in the exercises.
# 
# ### Row Execises
# This may seem a little confusing, so work through the following exercises for a better understanding.
# 1. Display the three Age values in the above example to verify that they all return the same value. If you want to be clever, write the three examples on the same line separated by commas. You will get a three-number tuple showing the three identical values.
# 2. What does the `values` attribute look like without an index? To find out, evalute `training_df.values`to see the result.
# 3. How does a row appear using `values`compared with `iloc`? Display row 42 using both attributes (`values[41]` and `iloc[41]`) to see the difference.
# 

# In[ ]:


# Code area for Row Exercises


# To really see how `iloc` and `loc` differ, let's assign a named index to a DataFrame. There is a method `set_index()` that does this. Like most operations in pandas, this returns a new DataFrame by default so the original DataFrame `training_df` is preserved. Uncomment and execute the code in the following cell to create a new DataFrame called `mydata` and view the first few rows. Note how the Ticket column is now shown as the index for the table.

# In[ ]:


# Uncomment the following statements to create the mydata DataFrame
# mydata = training_df.set_index('Ticket')
# mydata.head()


# After the prior cell executes, we have a new DataFrame `mydata` where the index is very different than the row number. Let's continue our exercises.
# 4. Use the `axes` attribute to examine the new table structure. Is Ticket still in the list of columns?
# 5. Find the passenger names for ticket 113803 with the code `mydata.loc['113803', 'Name']`. Would it be easy to find this information with the `iloc` attribute?
# 6. Find the cabin for the passengers with ticket 113803 (hint: use `mydata.loc`).
# 7. Show the two rows with ticket 113803 using a single value in the `loc` attribute, namely`loc['113803']`.
# 8. Look at the PassengerId of these rows (4 and 138). Display these two rows using the `iloc` attribute by passing in a list for the first and only argument. That is, as `mydata.iloc[[3,137]]`. Note the double brackets. What happens if you pass these same values to the `iloc` attribute in our `training_df` table?
# 8. In the `mydata` table, the Cabin column is tenth. Show the cabin name for our two passengers with ticket 113803 using the `iloc` attribute (hint: your indexes should look like this: `[[3,137],9]`).
# 9. Now show these same two cabin names with the `training_df` DataFrame, first using the `iloc` attribute and then using the `loc` attribute. Remember you are using the row index with `iloc` and the index name with `loc`.
# 10. We can show a range of rows with `iloc` using a standard python range, like 11:20. Show only the name and cabin columns for the passengers in rows 10 through 19 (hint: your last row should be Mrs. Fatima Masselmani with a `NaN` cabin).
# 11. Do the same with the `loc` attribute. That is, show only the name and cabin columns for passengers in rows 10 through 19 (hint: use `training_df` for this exercise).

# In[ ]:


# Code area for more Row Exercises


# The last exercise is a bit of a trick question. Using `training_df.loc` the row numbers are the index value so you can use a range just like you did for `iloc` in the prior exercise. However, you will discover that the numbers required for `training_df.loc` are different than those for `training_df.iloc`. This is because `iloc` and `loc` handle such ranges (slices) differently, because one is meant to be a row number, and the other an index name. The `iloc` attribute follows the python convention of excluding the right-hand value, so with `iloc` the slice 4:10 translates to 4, 5, 6, 7, 8, 9. The `loc` attribute includes the right-hand index name, so with `loc` the slice 4:10 translates to 4, 5, 6, 7, 8, 9, 10.
# 
# As a result, the two answers for the final exercise should be:
# > `training_df.iloc[11:20,[3,10]]   # shows rows 11 through 19`   
# > `training_df.loc[11:19,['Name','Cabin']]    # shows rows with index 11 through 19`

# ## DataFrame Methods
# There are perhaps a hundred or more methods for DataFrame objects, so a complete discussion is not really possible here. I have somewhat arbitrarily broken some of the more common methods into some groups to make them easier to discuss. Let's start with methods that display data from the table.

# ### Display Methods
# A number of methods provide basic information about the DataFrame, and are useful to better understand what you are looking at or explore some of the values in the table.
# * `filter(items)` returns a DataFrame filtered by the given criteria.
# * `head(n=5)` returns the first `n` rows of the DataFrame, , or the first 5 rows if `n` is not specified.
# * `info()` returns a consise summary of the DataFrame.
# * `nunique()` shows the number of unique observations (values) for each column
# * `sample(n=1)` displays a random sampling of `n` rows in the DataFrame, or one row by default.
# * `tail(n=5)` shows the last `n` rows of the DataFrame, or the last 5 rows if `n` is not specified.
# 
# #### Display Methods Exercises
# 1. Which columns in the training data (`training_df`) have null values (use `info()` method)?
# 1. Which columns in the test data (`test_df`) have null values?
# 2. Do the training and test data have the same number of passenger classes (Pclass)?  (hint: use `nunique()` method)?
# 1. Display the first 3 and last 3 rows of the training data. Note how the corresponding methods return a new DataFrame.
# 3. There is an `append()` method that appends a similar DataFrame to the current one.  Create a DataFrame with 6 rows by calling `append()` on the first 3 rows of the training data and passing in the last three rows of the training data.
# 1. The `filter()`method accepts a list of colums, as in `filter(['Name'])`. Use `filter()` and `head()` to show the first 10 rows of just the Name column.
# 1. The `filter()` method also accepts multiple columns. Use this to show a random set of 7 rows from just the Pclass and Fare columns in the training data. Based on this sample, does it appear that passengers in first or second class (1 or 2) paid more than passengers in third class?
# 1. Display a random 5 rows from the first 10 rows of the last 100 rows of the training data. If you call this correctly, the  result will show PassengerId values from 791 to 800, inclusive.

# In[ ]:


# Code area for more Display Methods Exercises


# ### Common Named Parameters
# Most pandas methods accept one or more optional (named) parameters. These default to reasonable values, and can be specified explicitly to alter how the method behaves. The following are fairly common and good to know.
# * `ascending` indicates whether to sort in ascending (`True`, the default) or descending (`False`) order. For example `training_df.mean().sort_values(ascending=False)` returns the mean of each column from largest to smallest.
# * `axis` indicates the axis along which an operation occurs, either `index` or `columns`, or `0` and `1` respectively. It's a bit confusing. In most methods `index` is the default, meaning that computations are column-wise (along the index). So `df.mean()` or `df.mean(axis='index')` calculates the mean of each column, while `df.mean(axis=1)` calculates the mean for each row.
# * `inplace` indicates whether to return a new DataFrame (`False`, the default) or modify the existing DataFrame (`True`). The default is `False` so that a new DataFrame is returned.
# 
# We'll use these parameters in some upcoming examples and exercises.

# ### Analysis Methods
# Now let's look at some methods that analyze the DataFrame in some manner.
# * `any()` returns whether any elements are `True`. There is an `all()` method that returns whether all elements are `True`.
# * `corr()` returns the pairwise correlation of numeric columns, excluding `NaN` values. A `corrwith()` method also exists to perform correlation against another DataFrame (or a column in the current DataFrame).
# * `describe()` returns statistics that summarize the shape of numerical columns.
# * `isnull()` returns a boolean same-sized DataFrame indicating which values are not defined (`NaN`)
# * `median()` returns the median value for each numeric column. Methods for `mean()`, `min()`, and `max()` also exist.
# * `sum()` adds (or concatenates, for strings) the columns or rows in the DataFrame.
# 
# #### Analysis Methods Exercises
# 1. Use the `any()` method with the DataFrame returned by `isnull()` to show which columns have null values in the training data.
# 2. Which numeric column correlates most postively with the Survived column? Use the `corr()` method on the training set to view the correlations on the numeric columns in the training data, and then look at the Survived column or row to see the answer. A higher positive correlation means that higher column values translate to higher survival chances. Does this make sense?
# 3. The `corrwith()` method makes the prior question 2 even easier, as it takes the target for comparison. Call `corrwith()` using the Survived column as the parameter (`training_df.Survived`) to view correlations against the Survived column.
# 1. Using the answer for question 3, what column most negatively correlates with survival? A negative correlation means that lower column values translate to higher survival chances. Does this make sense?
# 5. What are two ways to find the median value for the Age column? (hint: finding the 50% value using the `describe()` method is one of them).
# 6. Methods that result in a list, or Series, also accept a column name as an attribute to show that column's value. For example, `training_df.mean().Fare` will return the mean value for just the Fare column.  Use this feature to find  the difference between the median and mean value for the `Fare` column, writing your answer as a single expression.
# 1. What is the difference in Ages between the oldest and youngest passenger?
# 1. Is the median Age in the training set nearer in years to the youngest passenger or the oldest passenger?
# 7. How many passengers in the training set survived? Recall that the Survived column is 0 if a passenger did not survive and `1` if they did, so you can use the `sum()` method to determine this answer.
# 8. What is the minimum and maximum Fare paid by a passenger in the training set?

# In[ ]:


# code for Analysis Method Exercises


# ### Modify Methods
# Here are some methods that modify the DataFrame in some way. These accept the `inPlace` parameter to specify whether to modify the original DataFrame (`True`) or create a new one (`False`, the default).

# * `clip()` returns a DataFrame with values below a threshold set to this low threshold, and above another threshold set to this high threshold. There are `clip_lower()` and `clip_uppper()` methods as well.
# * `copy()` returns a copy of a DataFrame.
# * `drop(labels, axis)` returns a DataFrame with the given rows or columns removed. There are also methods to drop duplicate rows (`drop_duplicates()`) and missing values (`dropna()`).
# * `fillna(value)` returns a DataFrame with `NaN` values replaced with the given value.
# 
# As an example, one way to handle missing (null) values is to simply drop them from the table. The following would drop the rows with null values from the training set:
# > `training_df.dropna(inplace=True)` (or `training_df.dropna(axis=0,inplace=True)`)
# 
# Alternatively, the following would drop the columns with null values from the training set:
# > `training_df.dropna(axis=1,inplace=True)`
# 
# #### Modify Methods Exercises
# Let's get to know these methods a bit more with the following exercises:
# 1. Create a copy of the training set called `mydata` and call `mydata.shape` to verify it has the same size as the original set. Use this copy for the subsequent exercises.
# 2. Which removes more data from the `mydata` table, dropping rows with null values or columns with null values? Use `dropna()` to drop each axis in turn, as in the example, and use the `shape` or `size` attribute to determine which resulting table has more values.
# 1. Another way to handle null values is fill them with another value, using the `fillna()` method. Fill null values in the `mydata` table with 0 (zero) in place and then execute `mydata.isnull().any()` to verify there are no null values left in the table.
# 1. Refresh `mydata` with a fresh copy of the training set. Many DataFrame methods, including `fillna()`, can be invoked on a single column, as in `training_df.Fare.fillna(42)` to fill missing Fare values with 42. Fill the null values in the Age column for `mydata` (in place) to its median value.
# 1. Using the modified Age column in `mydata` from the prior question, set the minimum and maximum Age in the mydata table to 5 and 75, respectively (hint: use `clip()` and pass the lower and upper values). Use the `describe()` method to verify the result.
# 1. One standard use of `clip()` is to eliminate outliers that might otherwise skew the data. Look at the median and max value for the Fare column and you will see a rather large descrepancy. Modify the `mydata` table (in place) so the maximum value for the Fare column is 100 (hint: use the `clip_upper()` method on `mydata.Fare`). Show the `mean()` for the Fare column in both `mydata` and `training_df` to see the difference.
# 1. Create a filtered table from the training set including only the PassengerId, Fair, and Age columns. Clip this DataFrame with a lower threshold of 25 and upper threshold of 100, and show a sample of 20 rows from the result. Inspect this output to verify that the values for all columns are from 20 to 100.

# In[ ]:


# code for Modify Method Exercises


# ### Evaluate Methods
# There are a few methods that evaluate the table, or data in the table, in some manner, including the following:
# * `eval(expr)` returns the DataFrame that results from the given operation on the DataFrame columns. For example, `training_df.eval('Decade = Age/10')` returns a DataFrame with a new Decade column, where the Decade value for each passenger is set to that passenger's Age divided by 10.
# * `groupby(by)` returns an object grouped by the given columns (or other mechanism) suitable for additional processing. See the exercises for a better understanding of this method.
# * `query(expr)` returns the result of querying the DataFrame with the given boolean expression.
# * `sort_values(by)` sorts the DataFrame by the given column. For example, `training_df.sort_values('Age')` returns a DataFrame with the rows sorted by the passenger's age.

# #### Evaluate Methods Exercises
# These can get a little complicated, so the following exercises well help see how they might be used.. 
# 1. Sort the training set by Age from oldest to youngest (so `ascending=False`) and view the first 10 rows. Did these passengers tend to survive?
# 1. Did children less than 1 year old tend to survive? Use `query()` with the expression `'Age < 1'` to find out.
# 1. There are two passengers with age 30.5. Did they survive? 
# 1. The `eval()` method can be used to add new columns. Use `eval()` with the expression `'FamilySize = SibSp + Parch'` to return a DataFrame with a new FamilySize column that adds the Sibling and Parent/Children columns to obtain the total family size for each passenger. What is the minimum, maximum, and median value for this new column?
# 1. Make a copy of the training set called `mydata`. Add a new column IsFemale to this table (in place) that is `True` when a passenger is female. That is, when the passenger's Sex is equal to female (`== "female"`). Use this table for the remaining questions.
# 1. Does the new IsFemale column correlate well with the Survived column? What does this tell you?
# 1. What is the survival rate by passenger class? Group `mydata` by Pclass column (using `groupby()`) , and call `mean()` on the result to determine the answer. Which class is most likely to survive?
# 1. The `groupby()` method accepts a list. Look at the survival rate by class and sex by passing `['Pclass','IsFemale']` into this method, and calculate the `sum()` on the result to examine how many males and females survived in each class.
# 1. What is the median fare by passenger class?
# 1. Use `query()` and `shape` to find out how many passengers survived.
# 1. How many passengers were in third class? (hint: use a similar approach as the prior exercise)
# 1. How many people in the training data paid a Fare of more than 200?

# In[ ]:


# code for Evaluate Method Exercises


# ### Plotting and Output Methods
# The last group of methods relate to plotting and output.
# * `plot()` displays a graphic based on the given parameters.
# * `to_csv(path)` is one of many `to_` methods that writes a DataFrame into another format. This method saves the DataFrame into a csv file in the given path. Methods for saving as Excel, HTML, json and many other formats also exist. 
# * `to_pickle(path)` serializes a DataFrame into the given path as a [pickle](https://docs.python.org/3/library/pickle.html) object.
# 
# The `to_` methods are fairly self-explanatory. These methods accept the file location where the output should be written. We mention `pickle` here as this is a good way to transfer objects between environments. The `to_pickle()` method outputs a pickled object, which can then be read into another enviornment using the `read_pickle()` method.
# 
# The `plot()` method uses the 2D plotting package matplotlib to create graphs of the data.  A large number of parameters control what kind of plot to create, including `kind` for the kind of chart (bar, pie, scatter, and others), `x` and `y` for what to  show on each axis,  and `title` to set the chart title. A full discussion is beyond the scope of this discussion, but here are some examplse that show how this method works. The comment before each example is meant to illustrate the type of question the subsequent plot might answer.
# > `# View passenger age clusters for survived passengers (alpha sets transparency)`   
# > `training_df.plot(x='Age',y='Survived',kind='scatter', alpha=0.1)`    
# >
# > `# What is the median of each numeric column among survivors`   
# > `# Note that removing PassengerID from the table makes our chart more readable`   
# > `training_df.drop('PassengerId',axis='columns').median().plot(x='Survived',kind='bar')`   
# > 
# > `# What is the distribution of Fares among the passengers`   
# > `training_df.plot(y='Fare',kind='hist', title="Passenger Counts by Fare Groupings")`   
# > 
# > `# Display the average fare by passenger class`   
# > `training_df.groupby('Pclass').mean().plot(y='Fare', title="Average Fare by Class")`   
# 
# Feel free to try these and other parameters for the `plot()` method. More detailed documentation for this method can be found [in the pandas documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html)

# In[ ]:


# Try the plot() examples here


# ## Conclusion
# That is an overview of the DataFrame class. We looked at attributes and methods, and did a number of exercises aimed at highlighting different aspects of the Titanic training data. Since this notebook is based on a competition, we'll end with putting a submission together using the tools we have seen in this notebook.
# 
# With that in mind, the following lines assemble a training set for the competition. You should recognize most of the methods in this section from the exercises.

# In[ ]:


# X is the traditional training set, with y the solution set.
key_features = ['Pclass','Age','Fare','Sex','SibSp','Parch']
X = training_df.filter(key_features)
y = training_df['Survived']

# Modify the training set with some addition data
X.eval('IsFemale = (Sex == "female")', inplace=True)
X.eval('FamilySize = SibSp + Parch', inplace=True)
X.drop(['Sex','SibSp','Parch'], axis=1, inplace=True)
X.info()


# In[ ]:


# Only the Age column has null values, so we can fill based on the median Age
X.fillna(X.Age.median(), inplace=True)
X.corrwith(y).sort_values(ascending=False)


# Looking at the correllations, the new IsFemale column correlates reasonable well with survivorship. Using our  training set `X` and solution set `y` we can create a model for submission. The goal here is just to run through the submission steps, rather than create an optimal solution. We will use a RandomForestRegressor since these models do reasonably well on small data sets like the Titanic.

# In[ ]:


# Fit a RandomForestRegressor model to the data.
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X,y)


# We have a model! Next step is to adjust our test set to have the same set of columns, and fill in any missing values so we can generate a prediction.

# In[ ]:


test_X = test_df.filter(key_features)
test_X.eval('IsFemale = (Sex == "female")', inplace=True)
test_X.eval('FamilySize = SibSp + Parch', inplace=True)
test_X.drop(['Sex','SibSp','Parch'], axis=1, inplace=True)
test_X.isnull().any()


# In[ ]:


# We have two columns with missing values, so we will treat these separately.
test_X.Age.fillna(test_X.Age.median(), inplace=True)
test_X.Fare.fillna(test_X.Fare.median(), inplace=True)
test_X.isnull().any()


# Our final step is to run a prediction for the test set against our model. The prediction produces probability values between 0 and 1, so we adjust the results to integer values of 0 or 1 to produce our Survived column.

# In[ ]:


predict = np.around(rf_model.predict(test_X)).astype(int)
output = pd.DataFrame({'PassengerId': test_df.PassengerId,
                      'Survived': predict})

output.to_csv('submission.csv', index=False)
print('Ready to submit. Commit this notebook and go to the Output tab.')


# ## Submission
# There you have it. If you wish to submit the prediction to the competition. press the Commit button at the top of this page. Once complete, press the Open Version button to view the committed version. Then press Output to view your submission. Press the Submit to Competition button here, and you will submit your data.
# 
# You should find that our prediction is about 75% accurate. Not too bad for focusing only on DataFrame methods. This accuracy can be approved of course, and you will find other notebooks aimed at more accurate predictions on the competition Kernels page.
# 
# Thank you for reading through to the end. If you enjoyed this, click the Upvote button to show your appreciation. If there is interest I would like to create another kernel focused on the pandas Series object.  Until next time.
