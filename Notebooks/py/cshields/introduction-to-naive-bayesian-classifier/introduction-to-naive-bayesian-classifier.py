#!/usr/bin/env python
# coding: utf-8

# This notebook introduces Python, data analytics, and an online platform to practice them - Kaggle! We will go over the basics of data storage, manipulation, and visualization with Python. Then we will implement a statisical learning model to develop a classification predictor. To see how well our predictor performs, the answers from our predictor will be submitted to a Kaggle competition and scored. Links to more advanced data manipulation and analyses will be referenced thoroughout.
# 
# **Warning:** Our goal is provide a starting point for Python data analytics, not to teach Python from the ground up or to address statistical learning at a detailed level. There will be references included for those who wants more details.
# 
# The challenge is to predict survival on the Titanic. We will attempt this by implementing a statistical learning method called the Naive Bayesian Classifier. To implement and train our classifier we use  three Python packages: Numpy, Pandas, Seaborn, and Scikit Learn. Numpy includes tools for linear algebra, statistical anlysis, and matrix manipulation. Pandas helps us store and manipulate data. Seaborn is used to visualize the data stored in Pandas. Finally, Scikit Learn includes the data analysis tools that we will use develop and train our predictive model.
# 
# There are three sections of this notebook:
# 1. Data wrangling
#     1. Load data
#     2. FIll in missing data
# 2. Feature visualization
# 3. Modeling
#     1. Load model
#     2. Train model
#     3. Create predictions

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


# **Section 1A: Load data**
# 
# FIrst we need to load the data with Pandas
# 
# pd.read_csv(file_name) reads data from a csv file to create a DataFrame that stores data in a table.
# Pandas has other read functions, e.g. read_xlsx()

# In[ ]:


#The train data has features (independent variables) and targets (dependent variable).
#Feature examples are Name, Age, or Fare. The target is if the passenger survived.
train=pd.read_csv('../input/train.csv')
#We print the first 5 rows of the train dataset to show what this looks like:
ntrain = train.shape[0] #this gets the number rows in the traning dataset
print("Training data (",ntrain,"rows)")

#Display the data in Pandas: .head(n_rows) shows the first n rows of the DataFrame
display(train.head(10))

#The test dataset is used to test how well the classifier performs
#Test data only has features, the targets are empty and must be predicted
test=pd.read_csv('../input/test.csv')

#Lets looks at the test data...
ntest = test.shape[0]
print("Test data (",ntest,"rows), notice that the survived column (target) is missing!")
display(test.head(10))


# After the train and test data are loaded, we combine them into a single DataFrame so we can inspect the data and fill in missing values.
# 
# pd.concat() concatenates multiple DataFrames into a new DataFrame

# In[ ]:


df_all=pd.concat([train,test],axis=0)


# **Section 1B: Fill in missing data**
# 
# Now we inspect our data to determine how much data we have, what our features are, and the amount of missing data.
# 
# DataFrame.info() shows the number of entries (rows) in a DataFrame, the number of features (columns), the type of data in each feature, and number of null (presumably missing) entries in each feature.

# In[ ]:


#Use DataFrame.info() and print the results.
print(df_all.info())


# In this notebook, we will only use the Age, Sex, and PClass features so it is important to know the way these features are recorded in the table and how much of that data is available. To investigate, let's take a look at four lines in the info() output:
# > Int64Index: 1309 entries, 0 to 417'
# 
# > Age            1046 non-null float64
# 
# >Sex            1309 non-null object
# 
# > Pclass         1309 non-null int64
# 
# The line "Int64Index..." tells us that there are 1309 entries in the data set and the values of those entries range from 0 to 417. Here, we are concerned with the number of entries and will ignore the range. 
# 
# "Age..." shows that the feature labeled Age has 1046 entires that are not null values. This means that there are 263 missing entries for Age! We can also see that Age is "float64" data type - the entries are numbers and can have decimals. Before we can use the Age feature, we need to fill in these missing values.
# 
# The other parts of the output provide similar information. The line for the feature "Sex" shows that there are 1309 entries for this feature, which means it does not have any missing data! Sex is an "object" datatype - in this case a string for 'male' or 'female' . Later, we will convert these categorical labels to numbers so that they can be interpreted by our model.
# 
# The line for the passenger class feature, "PClass," also has 1309 entries,  The datatye for PClass is "Int64" - it can only be a real number without decimals. Lucky us, we don't need to do anything with this feature!

# **Fill in Age values**
# 
# Now we fill in the missing Age values with the median age in the dataset. To do this we use a combination of functions:
# * DataFrame.fillna(value): this function fills the missing values in the DataFrame with the input value. 
# * DataFrame.median(): this function gets the median value from columns in a DataFrame.

# In[ ]:


#First we get the median age by calling DataFrame.median() on the 'Age' column
age_med=df_all['Age'].median()

#Print the Median Age
print('Median Age = {}'.format(age_med))

#Fill in the missing ages with the median values and overwrite previous column
df_all['Age']=df_all['Age'].fillna(age_med)

display(df_all.head(10))


# **Convert Sex to numeric values**
# 
# The Sex feature must be converted to numeric values so that it can be used in model. To encode 'male' and 'female' into numbers we replace 'female' with 1 and 'male' with 0. This denotes the Sex feature as a binary variable (1 or 0 variable). We do this with the replace function:
# * DataFrame.replace(value_to_replace, replacement_value):  this is pretty straight-forward, it replaces a value in the dataframe with another value.
# 

# In[ ]:


#Replace the strings in the 'Sex' column with numbers
df_all['Sex']=df_all['Sex'].replace('male',0)
df_all['Sex']=df_all['Sex'].replace('female',1)

#Look at the changed column
display(df_all.head())


# **Step 2: Feature visualization**
# 
# We have our three features - Age, Sex, PClass - formatted for analysis. But first let's visualize the these features to see what they tell us about passenger survival. We visualize using the Python package, Seaborn. Specifically, the function pairplot plots the pairwise relationships in a dataset:
# * sns.pairplot(dataset, vars=variables_to_plot, hue=comparison_aspects): this plots the pairs of  variables_to_plot in the dataset separated by the their value in terms of the hue variable. See https://seaborn.pydata.org/generated/seaborn.pairplot.html for examples and documentation.
# 
# 

# In[ ]:


#Import the package
import seaborn as sns


# In[ ]:


#Visualize the training data to differentiate between passengers who survived
#Split the data
train=df_all[:ntrain]

#Plot the data
sns.pairplot(train,vars=['Age','Sex','Pclass',],hue='Survived',)


# *Remember! Died is Survived=0 and Lived is Survived=1*
# 
# The plot above has features on the x- and y- axes and the values are color-coded by if the passenger survived or not. The plots on the diagonal show the histogram of each feature separated by the the value of Survived. The off-diagonal plots show the pairwise comparison of each feature and are color-coded by the value of Survived. 
# 
# Based on the historgrams on the diagonals, we see that Sex and Pclass appear to be related to the chance of survival. Males died more often than females and third class passengers died more often than first and second class passengers. We also see in Age that younger passengers appear to survive more often. As they say, "Rich women and children first!"
# 
# Now we will use these features to train a model and predict the survival of passengers in the test set.

# **Step 3: Model**
# 
# Now that we have the features we care about, we must create a model to predict survival based off of those features. Survival is categorical, passengers either survived or did not, so we use a classifier model. We use a Naive Bayesian Classifier in this model. This requires three steps:
# 1. Load the model from Scikit learn
#     1. See http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html for the model details.
#     2. Refer to "*The elements of statistical learning* " (2017) p. 210-211 for a description of the Naive Bayesian Classifier.
# 2. Train the model with the training data
# 3. Predict survival for the test data

# In[ ]:


#Load the model from the sklearn package
#See http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
from sklearn.naive_bayes import GaussianNB

#Now call the model so we can train it
clf=GaussianNB()


# In[ ]:


#Train the model
#First, separate the features and target
y_train=train['Survived'].values
x_train=train[['Age','Sex','Pclass']].values

#Now train the model on the features and target
clf.fit(X=x_train,y=y_train)

#Check accuracy on training set
#Technically, we should use cross-validation to check accuracy.
#Cross-validation helps to prevent overfitting the model by chasing training accuracy
print('Bayesian Classifier Score = {}'.format(clf.score(X=x_train,y=y_train)))


# In[ ]:


#Now we predict the test set values
#First we get the test values
test_df=df_all[ntrain:]
x_test=test_df[['Age','Sex','Pclass']].values

#Now predict our results
results=clf.predict(x_test)


# **Submit our results to the Kaggle competition!**
# 1. Convert our results to a .csv format required for submission
# 2. Click "Commit & Run" button at the top right of the webpage
# 3. Click the "<" arrow next to "Commit & Run"
# 4. In the right pane that opens up go to the "Versions" tab, then the "Output" tab, and click "Submit to Competition" 

# In[ ]:


#Convert the results to int datatypes (real numbers)
results=[int(i) for i in results]

#Get passenger id's from test set with the .iloc command
results_id=df_all['PassengerId'].iloc[ntrain:].values

#Create a dataframe for submission
submission=pd.DataFrame({'PassengerId':results_id,'Survived':results})

#Check what the submission looks like
display(submission.head(10))

#Save the dataFrame as a .csv (save to Kaggle)
submission.to_csv('submisison.csv',index=False)

#Now complete steps 2, 3, and 4 to submit for scoring!


# In[ ]:




