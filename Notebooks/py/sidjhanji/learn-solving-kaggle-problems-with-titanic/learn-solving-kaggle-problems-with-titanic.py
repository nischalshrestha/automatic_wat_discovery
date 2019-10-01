#!/usr/bin/env python
# coding: utf-8

# **Getting Started **

# *Hey guys, thank you for visiting my kernel.
# This kernel is for all the beginners  and machine learning enthusiasts who face the difficulty and want to learn how to get started with the dataset. 
# If you are an expert reading this and would like to give your inputs. Please comment the same below. We really appreciate your guidance.* 

# ***Happy Learning :D***

# In this competition, we have a data set of different information about passengers onboard the Titanic, and we see if we can use that information to predict whether those people survived or not.

# The training set contains data we can use to train our model. It has a number of feature columns which contain various descriptive data, as well as a column of the target values we are trying to predict:  ***Survival***

# The testing set contains all of the same feature columns, but is missing the target value column. Additionally, the testing set usually has fewer observations (rows) than the training set.

# **Importing the dataset**

# In[ ]:


# Importing the libraries
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Importing the training dataset
train = pd.read_csv("../input/train.csv")

# Viewing the number of rows and columns in the training dataset
train_shape = train.shape
print(train_shape)


# In[ ]:


# Importing the testing dataset
test = pd.read_csv("../input/test.csv")

# Viewing the number of rows and columns in the training dataset
test_shape = test.shape
print(test_shape)


# In[ ]:


train.info()


# **The dataset has below columns :**
# * PassengerID - A column added by Kaggle to identify each row and make submissions easier
# * Survived - Whether the passenger survived or not and the value we are predicting (0=No, 1=Yes)
# * Pclass - The class of the ticket the passenger purchased (1=1st, 2=2nd, 3=3rd)
# * Sex - The passenger's sex
# * Age - The passenger's age in years
# * SibSp - The number of siblings or spouses the passenger had aboard the Titanic
# * Parch - The number of parents or children the passenger had aboard the Titanic
# * Ticket - The passenger's ticket number
# * Fare - The fare the passenger paid
# * Cabin - The passenger's cabin number
# * Embarked - The port where the passenger embarked (C=Cherbourg, Q=Queenstown, S=Southampton)

# In[ ]:


# The first 5 rows of the training dataset are below:
# Index in python starts with 0.
train.head()


# The type of machine learning we will be doing is called **classification**, because when we make predictions we are classifying each passenger as survived or not. More specifically, we are performing **binary classification**, which means that there are only two different states we are classifying.

# In any machine learning exercise, thinking about the topic you are predicting is very important. We call this step acquiring domain knowledge, and it's one of the most important determinants for success in machine learning.
# 
# In this case, understanding the Titanic disaster and specifically what variables might affect the outcome of survival is important. Anyone who has watched the movie Titanic would remember that women and children were given preference to lifeboats (as they were in real life). You would also remember the vast class disparity of the passengers.
# 
# This indicates that *Age*, *Sex*, and *PClass* may be good predictors of survival. We'll start by exploring Sex and Pclass by visualizing the data.

# **Visualizations**
# 
# There are a lot of different ways to visualize the data but lets start with the simple ones.
# 
# Because the *Survived* column contains 0 if the passenger did not survive and 1 if they did, we can segment our data by sex and calculate the mean of this column.

# In[ ]:


# We can use DataFrame.pivot_table() to easily do this
# Importing the library for plotting
import matplotlib.pyplot as plt
# Calling the pivot_table() function for Sex
sex_pivot = train.pivot_table(index = "Sex", values = "Survived")
sex_pivot.plot.bar()
plt.show()


# We can immediately see that females survived in much higher proportions than males did.
# 
# Let's do the same with the Pclass column.

# In[ ]:


# Calling the dataframe.pivot_table() function for Pclass
pclass_pivot = train.pivot_table(index = "Pclass", values = "Survived")
pclass_pivot.plot.bar()
plt.show()


# You can see that a passenger belonging to class 1 has a better chance of surviving in comparison to class 2 & 3 passenger.

# The *Sex* and *PClass* columns are what we call **categorical** features. That means that the values represented a few separate options (for instance, whether the passenger was male or female).

# In[ ]:


# Let's take a look at the Age column using Series.describe()
train["Age"].describe()


# The *Age* column contains numbers ranging from *0.42* to *80.0* (If you look at Kaggle's data page, it informs us that Age is fractional if the passenger is less than one). The other thing to note here is that there are 714 values in this column, fewer than the 891 rows we discovered that the train data set had earlier in this mission which indicates we have some missing values.

# All of this means that the *Age* column needs to be treated slightly differently, as this is a continuous numerical column. One way to look at *distribution of values in a continuous numerical set* is to use **histograms** . We can create two histograms to compare visually the those that survived vs those who died across different age ranges:

# In[ ]:


# Contains the details of the passengers who survived
survived = train[train["Survived"] == 1]
survived["Age"].plot.hist(alpha=0.5, color="red", bins=50)


# In[ ]:


# Contains the details fo the passengers who died
died = train[train["Survived"] == 0]
died["Age"].plot.hist(alpha=0.5, color="blue", bins=50)


# In[ ]:


# Viewing them combined
survived["Age"].plot.hist(alpha=0.5, color="red", bins=50)
died["Age"].plot.hist(alpha=0.5, color="blue", bins=50)
plt.legend(["Survived","Died"])
plt.show()


# The relationship here is not simple, but we can see that in some age ranges more passengers survived - where the red bars are higher than the blue bars.
# 
# In order for this to be useful to our machine learning model, we can separate this continuous feature into a categorical feature by dividing it into ranges. We can use the *pandas.cut()* function to help us out.
# 
# The pandas.cut() function has two required parameters - the column we wish to cut, and a list of numbers which define the boundaries of our cuts. We are also going to use the optional parameter labels, which takes a list of labels for the resultant bins. This will make it easier for us to understand our results.
# 
# Before we modify this column, we have to be aware of two things. Firstly, any change we make to the train data, we also need to make to the test data, otherwise we will be unable to use our model to make predictions for our submissions. Secondly, we need to remember to handle the missing values we observed above.

# In[ ]:


# Create a function to process the Age column to different categories
def process_age(df, cut_points, label_names):
    # use the pandas.fillna() method to fill all of the missing values with -0.5
    df["Age"] = df["Age"].fillna(-0.5)
    # cuts the Age column using pandas.cut()
    df["Age_categories"] = pd.cut(df["Age"], cut_points, labels=label_names)
    return df

# Cut the Age column into seven segments: Missing, from -1 to 0 Infant, from 0 to 5 Child, from 5 to 12 Teenager, from 12 to 18 
# Young Adult, from 18 to 35 Adult, from 35 to 60 Senior, from 60 to 100
cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
label_names = ["Missing", "Infant", "Child", "Teenager", "Young Adult", "Adult", "Senior"]

train = process_age(train, cut_points, label_names)
test = process_age(test, cut_points, label_names)    


# In[ ]:


# Use the pivot_tables() function to plot with Age_categories column
age_categories_pivot = train.pivot_table(index="Age_categories", values = "Survived")
age_categories_pivot.plot.bar()
plt.show()


# Before we build our model, we need to prepare these columns for machine learning. Most machine learning algorithms can't understand text labels, so we have to convert our values into numbers.
# 
# Additionally, we need to be careful that we don't imply any numeric relationship where there isn't one. If we think of the values in the Pclass column, we know they are 1, 2, and 3.

# In[ ]:


# value_counts() function is used to get the count of occurence unique values present in the column of the dataset.
train["Pclass"].value_counts()


# While the class of each passenger certainly has some sort of ordered relationship, the relationship between each class is not the same as the relationship between the numbers 1, 2, and 3. For instance, class 2 isn't "worth" double what class 1 is, and class 3 isn't "worth" triple what class 1 is.
# 
# In order to remove this relationship, we can create dummy columns for each unique value in Pclass.

# In[ ]:


# pandas.get_dummies() function will generate columns for us.
def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df,dummies], axis=1)
    return df

train = create_dummies(train, "Pclass")
test = create_dummies(test, "Pclass")

train.head()


# In[ ]:


# Similarly for Sex & Age Categories Column
train = create_dummies(train, "Sex")
test = create_dummies(test, "Sex")

train = create_dummies(train, "Age_categories")
test = create_dummies(test, "Age_categories")

train.head()


# In[ ]:


# Calling the pivot_table() function for Embarked
embarked_pivot = train.pivot_table(index="Embarked", values="Survived")
embarked_pivot.plot.bar()
plt.show()


# In[ ]:


# Calling the pivot_table() function for SibSp
SibSp_pivot = train.pivot_table(index="SibSp", values="Survived")
SibSp_pivot.plot.bar()
plt.show()


# In[ ]:


# Calling the pivot_table() function for Parch
parch_pivot = train.pivot_table(index="Parch", values="Survived")
parch_pivot.plot.bar()
plt.show()


# In[ ]:


train.columns


# In[ ]:


# View the details of the other columns
columns = ["SibSp", "Parch", "Fare", "Cabin", "Embarked"]
train[columns].describe(include="all", percentiles=[])


# Of these,* SibSp, Parch and Fare * look to be standard numeric columns with no missing values. Cabin has values for only 204 of the 891 rows, and even then most of the values are unique, so for now we will leave this column also. Embarked looks to be a standard categorical column with 3 unique values, much like PClass was, except that there are two missing values. We can easily fill these two missing values with the most common value, "S" which occurs 644 times.
# 
# Looking at our numeric columns, we can see a big difference between the range of each. SibSp has values between 0-8, Parch between 0-6, and Fare is on a dramatically different scale, with values ranging from 0-512. In order to make sure these values are equally weighted within our model, we'll need to rescale the data.
# 
# Rescaling simply stretches or shrinks the data as needed to be on the same scale, in our case between 0 and 1.

# In[ ]:


# the preprocessing.minmax_scale() function allows us to quickly and easily rescale our data
from sklearn.preprocessing import minmax_scale
# rescale the SibSp, Parch, and Fare columns
# Added 2 backets to make it a dataframe. Otherwise you will get a type error stating cannot iterate over 0-d array.
train["SibSp_scaled"] = minmax_scale(train[["SibSp"]])
train["Parch_scaled"] = minmax_scale(train[["Parch"]])
train["Fare_scaled"] = minmax_scale(train[["Fare"]])
train.head()


# In[ ]:


# Checking the details with the test data
test[columns].describe(include="all",percentiles=[])


# In[ ]:


# Fare column has a missing value, we will replace the missing value with the mean
test["Fare"] = test["Fare"].fillna(train["Fare"].mean())
test["Fare"].count()


# In[ ]:


# Applying the same rescaling to the test dataset 
test["SibSp_scaled"] = minmax_scale(test[["SibSp"]])
test["Parch_scaled"] = minmax_scale(test[["Parch"]])
test["Fare_scaled"] = minmax_scale(test[["Fare"]])
test.head()


# In[ ]:


# Analyzing & Fixing the Embarked Column
train[columns].describe(include="all", percentiles=[])


# In[ ]:


# We have 2 missing values in the training data of the Embarked column 
# S is the most common value occuring 644 time. So we will replace the missing value with S.
train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"].describe()


# In[ ]:


# Checking with the testing dataset
test[columns].describe(include="all", percentiles=[])


# In[ ]:


test.shape
# We have no missing value for Embarked in the test data


# In[ ]:


# Creating dummy columns for Embarked columns
train = create_dummies(train, "Embarked")
test = create_dummies(test, "Embarked")
train.head()


# In[ ]:


test.head()


# In order to select the best-performing features, we need a way to measure which of our features are relevant to our outcome - in this case, the survival of each passenger. One effective way is by training a logistic regression model using all of our features, and then looking at the coefficients of each feature.
# 
# The scikit-learn LogisticRegression class has an attribute in which coefficients are stored after the model is fit, **LogisticRegression.coef_**. We first need to train our model, after which we can access this attribute.

# In[ ]:


# Defining all the featured columns
lrColumns = ['Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'SibSp_scaled', 'Parch_scaled', 'Fare_scaled']

print(lrColumns)


# In[ ]:


# Applying Logistic Regression
from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression()
logisticRegression.fit(train[lrColumns], train["Survived"])
coefficients = logisticRegression.coef_
print(coefficients)


# The coef() method returns a NumPy array of coefficients, in the same order as the features that were used to fit the model. To make these easier to interpret, we can convert the coefficients to a pandas series, adding the column names as the index:

# In[ ]:


feature_importance = pd.Series(coefficients[0], index=lrColumns)
print(feature_importance)


# In[ ]:


# Plotting as a horizontal Bar chart
feature_importance.plot.barh()
plt.show()


# The plot we generated shows a range of both positive and negative values. Whether the value is positive or negative isn't as important in this case, relative to the magnitude of the value. If you think about it, this makes sense. A feature that indicates strongly whether a passenger died is just as useful as a feature that indicates strongly that a passenger survived, given they are mutually exclusive outcomes.
# 
# To make things easier to interpret, we'll alter the plot to show all positive values, and have sorted the bars in order of size:

# In[ ]:


ordered_feature_importance = feature_importance.abs().sort_values()
ordered_feature_importance.plot.barh()
plt.show()


# In[ ]:


# We'll train a model with the top 8 scores
predictors = ['Age_categories_Infant', 'SibSp_scaled', 'Sex_female', 'Sex_male',
       'Pclass_1', 'Pclass_3', 'Age_categories_Senior', 'Parch_scaled']

lr = LogisticRegression()
lr.fit(train[predictors], train["Survived"])
predictions = lr.predict(test[predictors])
print(predictions)


# In[ ]:


# Calculating the accuracy using the k-fold cross validation method with k=10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lr, train[predictors], train["Survived"], cv=10)
print(scores)


# In[ ]:


# Taking the mean of all the scores
accuracy = scores.mean()
print(accuracy)


# In[ ]:


# # Submitting the result
# submission = pd.DataFrame({
#         "PassengerId": test["PassengerId"],
#         "Survived": predictions
#     })
# submission.to_csv('submission1.csv', index=False)


# This is just the 1st submission, many more to come.

# A lot of the gains in accuracy in machine learning come from Feature Engineering. Feature engineering is the practice of creating new features from your existing data.
# 
# One common way to engineer a feature is using a technique called **binning**. Binning is when you take a continuous feature, like the fare a passenger paid for their ticket, and separate it out into several ranges (or 'bins'), turning it into a categorical variable.
# 
# This can be useful when there are patterns in the data that are non-linear and you're using a linear model (like logistic regression). We actually used binning when we dealt with the Age column, although we didn't use the term.

# In[ ]:


print(train["Fare"])


# In[ ]:


import numpy as np
np.histogram(train["Fare"])


# In[ ]:


# Creating a frequency table
from collections import Counter
fare_count = Counter(train["Fare"])
# fare_labels, fare_values = zip(*Counter(train["Fare"]).items())
# print(fare_labels)
print(fare_count)


# In[ ]:


plt.hist(train["Fare"], bins = range(150))
plt.show()


# In[ ]:


# print(fare_values)


# In[ ]:


# indexes = np.arange(len(fare_labels))
# print(indexes)


# In[ ]:


# Plotting a bar chart
# width = 6
# plt.bar(indexes, fare_values, width)
# plt.show()


# In[ ]:


# Plotting a histogram
# plt.hist(fare_values, bins=10)
# plt.show()


# In[ ]:


import seaborn as sns
sns.distplot(train["Fare"])


# In[ ]:


survived = train[train["Survived"]==1]
died = train[train["Survived"]==0]
survived["Fare"].plot.hist(alpha=0.5, color = "red", bins=range(150))
died["Fare"].plot.hist(alpha=0.5, color = "blue", bins=range(150))
plt.legend("Survived","died")
plt.show()


# Looking at the values, it looks like we can separate the feature into four bins to capture some patterns from the data:
# 
# * 0-12
# * 12-50
# * 50-100
# * 100+

# In[ ]:


# Creating functions as created for Age to perform binning on the Fare column
def process_fare(df, cut_points, label_names):
    df["Fare_categories"] = pd.cut(df["Fare"], cut_points, labels = label_names)
    return df

fare_cut_points = [0, 12, 50, 100, 1000]
fare_label_names = ["0-12", "12-50", "50-100", "100+"]

process_fare(train, fare_cut_points, fare_label_names)
print(train["Fare_categories"])


# In[ ]:


# for the test dataset
process_fare(test, fare_cut_points, fare_label_names)
print(test["Fare_categories"])


# In[ ]:


# Calling the create_dummies function to convert the categories into columns
train = create_dummies(train, "Fare_categories")
train.head()


# In[ ]:


test = create_dummies(test, "Fare_categories")
test.head()


# Another way to engineer features is by extracting data from text columns. Earlier, we decided that the **Name** and **Cabin** columns weren't useful by themselves, but what if there is some data there we could extract? Let's take a look at a random sample of rows from those two columns:

# In[ ]:


train[["Name","Cabin"]].head(10)


# While in isolation the cabin number of each passenger will be reasonably unique to each, we can see that the format of the cabin numbers is one letter followed by two numbers. It seems like the letter is representative of the type of cabin, which could be useful data for us. We can use the pandas Series.str accessor and then subset the first character using brackets:

# In[ ]:


train.head()["Cabin"]


# In[ ]:


train.head()["Cabin"].str[0]


# In[ ]:


# Creating a new column Cabin_type to store these values and fill all the missing values with unknown
train["Cabin_type"] = train["Cabin"].str[0]
train["Cabin_type"] = train["Cabin_type"].fillna("Unknown")
train["Cabin_type"].head()


# In[ ]:


# Doing the same for test dataset
test["Cabin_type"] = test["Cabin"].str[0]
test["Cabin_type"] = test["Cabin_type"].fillna("Unknown")
test["Cabin_type"].head()


# Looking at the Name column, There is a title like 'Mr' or 'Mrs' within each, as well as some less common titles, like the 'Countess'.  By spending some time researching the different titles, we can categorize these into the below categories:

# In[ ]:


titles = {
    "Mr" :         "Mr",
    "Mme":         "Mrs",
    "Ms":          "Mrs",
    "Mrs" :        "Mrs",
    "Master" :     "Master",
    "Mlle":        "Miss",
    "Miss" :       "Miss",
    "Capt":        "Officer",
    "Col":         "Officer",
    "Major":       "Officer",
    "Dr":          "Officer",
    "Rev":         "Officer",
    "Jonkheer":    "Royalty",
    "Don":         "Royalty",
    "Sir" :        "Royalty",
    "Countess":    "Royalty",
    "Dona":        "Royalty",
    "Lady" :       "Royalty"
}


# We can use the Series.str.extract method and a regular expression to extract the title from each name and then use the Series.map() method and a predefined dictionary to simplify the titles.

# In[ ]:


extracted_titles = train["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
train["Title"] = extracted_titles.map(titles)
train["Title"].head()


# In[ ]:


extracted_titles = test["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
test["Title"] = extracted_titles.map(titles)
test["Title"].head()


# In[ ]:


# Using the create dummies function to convert these values into categories
for column in ["Title","Cabin_type"]:
    train = create_dummies(train,column)
    test = create_dummies(test,column)
train.head()


# In[ ]:


test.head()


# We now have 34 possible feature columns we can use to train our model. One thing to be aware of as you start to add more features is a concept called **collinearity**. Collinearity occurs where more than one feature contains data that are similar.
# 
# The effect of collinearity is that your model will overfit - you may get great results on your test data set, but then the model performs worse on unseen data (like the test set).
# 
# One easy way to understand collinearity is with a simple binary variable like the **Sex** column in our dataset. Every passenger in our data is categorized as either male or female, so 'not male' is exactly the same as 'female'.
# 
# As a result, when we created our two dummy columns from the categorical Sex column, we've actually created two columns with identical data in them. This will happen whenever we create dummy columns, and is called the **dummy variable trap**. The easy solution is to choose one column to drop any time you make dummy columns.

# Collinearity can happen in other places, too. A common way to spot collinearity is to plot correlations between each pair of variables in a heatmap. 

# In[ ]:


import seaborn as sns
correlations = train.corr()
sns.heatmap(correlations)
plt.show()


# It is very difficult to make inferences from this heatmap.
# 
# Lets create a custom function for the same.

# In[ ]:


# custom function to set the style for heatmap
def plot_correlation_heatmap(df):
    corr = df.corr()
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
    
# Columns to use    
heatmap_columns = ['Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'SibSp_scaled', 'Parch_scaled', 'Fare_categories_0-12',
       'Fare_categories_12-50','Fare_categories_50-100', 'Fare_categories_100+',
       'Title_Master', 'Title_Miss', 'Title_Mr','Title_Mrs', 'Title_Officer',
       'Title_Royalty', 'Cabin_type_A','Cabin_type_B', 'Cabin_type_C', 'Cabin_type_D',
       'Cabin_type_E','Cabin_type_F', 'Cabin_type_G', 'Cabin_type_T', 'Cabin_type_Unknown']

plot_correlation_heatmap(train[heatmap_columns])


# We can see that there is a high correlation between **Sex_female/Sex_male** and **Title_Miss/Title_Mr/Title_Mrs.**
# 
# We will remove the columns Sex_female and Sex_male since the title data may be more nuanced.
# 
# Apart from that, we should remove one of each of our dummy variables to reduce the collinearity in each. We'll remove:
# 
# * Pclass_2
# * Age_categories_Teenager
# * Fare_categories_12-50
# * Title_Master
# * Cabin_type_A

# In an earlier step, we manually used the logit coefficients to select the most relevant features. An alternate method is to use one of scikit-learn's inbuilt feature selection classes. We will be using the feature_selection.**RFECV** class which performs recursive feature elimination with cross-validation.
# 
# *The RFECV class starts by training a model using all of your features and scores it using cross validation. It then uses the logit coefficients to eliminate the least important feature, and trains and scores a new model. At the end, the class looks at all the scores, and selects the set of features which scored highest.*
# 
# Like the LogisticRegression class, RFECV must first be instantiated and then fit. The first parameter when creating the RFECV object must be an estimator, and we need to use the cv parameter to specific the number of folds for cross-validation.

# In[ ]:


from sklearn.feature_selection import RFECV

predictor_columns = ['Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Young Adult',
       'Age_categories_Adult', 'Age_categories_Senior', 'Pclass_1', 'Pclass_3',
       'Embarked_C', 'Embarked_Q', 'Embarked_S', 'SibSp_scaled',
       'Parch_scaled', 'Fare_categories_0-12', 'Fare_categories_50-100',
       'Fare_categories_100+', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
       'Title_Officer', 'Title_Royalty', 'Cabin_type_B', 'Cabin_type_C',
       'Cabin_type_D', 'Cabin_type_E', 'Cabin_type_F', 'Cabin_type_G',
       'Cabin_type_T', 'Cabin_type_Unknown']

all_X = train[predictor_columns]
all_y = train["Survived"]

lr = LogisticRegression()
selector = RFECV(lr, cv = 10)
selector.fit(all_X, all_y)

optimized_predictors = all_X.columns[selector.support_]
print(optimized_predictors)


# The RFECV() selector returned only four columns.
# 
# Let's train a model using cross validation using these columns and check the score.

# In[ ]:


all_X = train[optimized_predictors]
all_y = train["Survived"]

lr = LogisticRegression()
lr.fit(all_X, all_y)
scores = cross_val_score(lr,all_X,all_y, cv=10)
print(scores)


# In[ ]:


accuracy = scores.mean()
print(accuracy)


# In[ ]:


predictions = lr.predict(test[optimized_predictors])


# In[ ]:


# Submitting the result
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('submission1.csv', index=False)

