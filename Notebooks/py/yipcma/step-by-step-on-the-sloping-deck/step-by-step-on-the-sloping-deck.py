#!/usr/bin/env python
# coding: utf-8

# ## Preface
# You love coding. You create apps that users get to create and update certain data through various interfaces. You are a transducer of data. But how does that data look as a whole? And what insights would you gain from this data to improve your web product? In this tutorial, we will take a first dive into Data Science: an art and science to gain insights from data.
# 
# There are two parts to this tutorial: Exploratory Data Analysis and Model Building.
# 
# This tutorial assumes no pre-requisites other than basic coding literacy and an open-mind to learn as-you-go, the most important attribute of a data science enthusiast.
# 
# ## Take the helm
# Welcome aboard this *HMS Kernel*. I'm your First Mate, Andrew, to sail alongside with you on your maiden voyage (aka beginner tutorial) braving the seas of exploratory data analysis and predictive model building in python. We'll be using various popular libraries such as pandas (data handling), matplotlib and seaborn (basic plotting), bokeh (interactive visualization), and scikit-learn (machine learning). We'll take everything step-by-step, and let things slowly "sink-in".  *cough*
# 
# Before we leave port, let's import.

# In[ ]:


import pandas as pd
df = pd.read_csv('../input/train.csv')


# # Inspect the Data
# 
# We have just loaded the data. You must have realized by now that we have, in fact, two data files: `train.csv` and `test.csv`. They are exactly what their names suggest: the *train* dataset will be used to train our model, and the *test* dataset will be used to test how well our model performs. For now, let's focus on the train dataset which we just loaded.  The CSV (comma-separated-value) format is very popular for storing tabular data, data with rows and columns. You can also open the file and inspect it in any spreadsheet applications. But since we have python, we don't have to. Instead, use the following commands to get an idea of what we are dealing with.
# 
# Remember: we are interested in who lived, and who didn't.
# 

# In[ ]:


df.columns
df.info()
df.head()
df.tail()
df.describe()
get_ipython().system(u'ls ../input')


# ## EDA
# Now we have an idea about the shape of the data and what variables it entails, let's make some plots to have a sense of how the variables relate.
# A great way to getting a bird's eye view is a pairplot. On the diagonal, it shows the distribution of a variable, and other entries, a pairplot would plot the paired values of the two variables (scatter plot). We color-code the points so that we can see whether that observation (passenger) survived (1) or not (0). 
# 
# Note: to do this we need to make sure there are no missing values in the data. We check this by `df.isnull().sum()`, and drop the variables that we don't look at for now.

# In[ ]:


# from pandas.plotting import scatter_matrix
# scatter_matrix(X_train, c=y_train);
import seaborn as sns
sns.pairplot(df.drop(['PassengerId'], axis=1).dropna(axis=1), hue='Survived', plot_kws={'alpha':0.3});


# ### Initial insights from the Pairplot
# A simple pairplot on the data already gives us some interesting insights. [TODO]
# 
# However, categorical/discrete variables such as "Sex" were not visualized in the pairplot.
# 
# If you recall from the movie Titanic, it seems that first-class female might have a better chance of survival than a third-class male. Let's try to see if this intuition is supported by the data.
# 
# To do this, we will try to estimate the distribution the passengers along the categories of gender and cabin-class and inspect the (average) estimates on survival.

# In[ ]:


sns.barplot(x='Sex', y='Survived', hue='Pclass', data=df);


# ### Insights from barplot
# The data tells us that most first-class females indeed survived, while the survival rate of a third-class male is below 20%. You also see a vertical black-bar on each of the combination of categories which indicates the *confidence interval*. For now, you may interpret that as a range of values (95% of the values) of the same experiment you would get if you repeat it many times (1000 here) by drawing passenger repeatedly from the lot and report the mean survival.
# 
# This leads to another interesting question: does it mean that the more you pay for the ticket, the more likely you survive?
# 
# To answer this, we will do a swarmplot with "Sex" and "Fare", and color-code the passenger with its survival.

# In[ ]:


sns.swarmplot(x="Sex", y="Fare", hue="Survived", data=df, alpha=0.3);


# ### Insights from the swarmplot
# It is unclear how Fare plays into the survival. We see some mixed signals especially at the top, where "Sex" seems to make a difference.

# ### Correlations
# We can also look at the how the numeric variables relate by the Pearson correlation coefficients, where 1 implies they are perfectly positively related, and -1 the opposite, with 0 meaning there's no correlation in terms of how the two variables vary.

# In[ ]:


### Investigate the correlations between variables
sns.heatmap(df.drop('PassengerId', axis=1).corr(), annot=True, cmap="coolwarm", center=0, square=True);


# ## Predictive Model
# Now that we have gained some preliminary understanding of the data underlying the problem, let's get all hands on deck  and build our first predictive model. Our task is to predict whether a passenger survived or not, in other words, we are tackling a *Classification Problem*, where the variables we use for predictions are the *features* and the labels we are trying to predict are *targets*. There are 4 general steps:
# 1. Prepare the data
# 2. Train/fit the model
# 3. Do some prediction and see how well the model performs
# 4. Repeat and improve the model
# 
# ### Prepare the Data
# We are blessed with a well formatted data here on Kaggle. The real world is way messier than this: you might need to pull together a dataset from various sources of different formats, you might need to patch holes in the data, and in some worse cases, you need to figure out where the problems are in the data and patch them.
# 
# In this maiden voyage into model building, let's stay on the safe side, and take variables/features from the data that look harmless and require relatively little effort to handle. First, look at the variable names and inspect if they have missing values.

# In[ ]:


df.info()


# ### Initial thoughts on choosing features
# Out of the 12 columns, only 3 of them have missing values, which is good news! As a first drill, let's use only the numeric columns as features and "Survived" as target. Note that we will drop PassengerId as it is an identifier of the passengers added after the fact and is arbitrary.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# X = df.select_dtypes(["int64", "float64"]).dropna(axis=1).drop(["PassengerId", "Survived"], axis=1).values
X = df.select_dtypes(["int64", "float64"]).fillna(df.median()).drop(["PassengerId", "Survived"], axis=1).values
y = df['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# # Drawing a line in the sand
# 0.6 does not look like a very accurate result (i.e., you get 3 out of 5 right in your predictions). If ML were to live up to its hype, surely it/we can do better than this.

# ## Now, your turn: fork this notebook and try another model to see if you can do better than this.
