#!/usr/bin/env python
# coding: utf-8

# _Newbe to Kaggle and datascience in general. As I am creating my first kernel for the Titanic Survival prediction model (in python) I wrote down everything that I thought was unclear -for me at least- at first. The goal I set for myself was to get everythng working, create a model that predicted at least somewhat better than chance alone, and upload it to Kaggle.<br> 
# <br>
# I hope this is useful to newbe kagglers like myself._
# <br>
# ***
# <br>
# ### 1 Decisions to make before starting your first datascience project (kaggle titanic..)
# - _First: the decision to either work 'within' Kaggle (the kaggle kernel) or use use your own downloadable platform. _
# <br>
# - _Second: 'Python or R?'_
# <br>
# <br>
# 
# #### First decision:
# What is meant by 'own platform'?
# You can do all of this outside of Kaggle and then upload (/copy paste code) to kaggle when done. Obviously this requires some setup but the advantages are that you know where you stand when doing datascience projects oustide of kaggle. The setup is made quite easy by means of an application called 'anaconda navigator' which also has other benefits so I stronlgy suggest using this if going for your own setup.
# <br>
# <br>
# #### Second decision:
# Discussion has been going forever. Rivalling the tabs vs spaces debate :P #piedpiper Being horrible at the javascript syntax I choose python because of its syntaxical ease but both have their advantages and disadvantages. 
# Most important take-away; eiter will be fine, if starting choose the language your (desired) job/company uses.
# <br>
# <br>
# 
# 
# ### 2 Getting started:
# - 2A: Orienting Yourself
# - 2B: Having a looksy at the data (or Exploratory Data Analysis (EDA))
# <br>
# <br>
# 
# ### 3 Data prepping & further visualization
# <br>
# ### 4 Modelling
# <br>
# ### 5 Submission
# <br>
#  ##### _(6 reiterate)_
#  <br>
#  <br>
# ***

# ### 1A: Getting started: Orienting Yourself
# First things first. Even before looking at your data you want to orient yourself _(at least you want to do this if you are new to jupyter notebook and kaggle kernels like me)._ See 'where are working from' and if necesary change your working directory to where you have saved your data files (csv's). <br>To do this you need:
# <li>import os   _#to be able to use the os. commands_
# <li>os.getcwd()   _#to find were you are working from now (like 'pwd' in unix)_
# <li>os.chdir('....path..')   _#to change from you current direcotory to you desired directory (e.g. where your data csv's are). Like 'cd' in unix._<br>
# 
# _If you are working fully within a kaggle kernel you can skip this. But is might be good to do this anyway for potential troubleshooting purposes in the future._

# In[ ]:


import os
os.getcwd()


# So we see we are currently set in kaggle/working. <BR>
# For kaggle competitions this is fine. <br>
# <br>
# If you are working 'locally' with Jupyter notebook, you can change this location with 'os.chdir('/...path...')
# So I look up were the folder is that contains the CSV's (train.csv & test.csv downloaded from kaggle) I intend to use.
# You do this outside Jupyter, by just looking on your computer and looking at the path. <br>
# For me it is /Users/steven/Documents/Kaggle/Titanic so this will be used in the following command.
# <br>
# #### This is case sensitive so pay attention to whether your folders on your PC start with or without uppercase!
# <br>
# _### If you are working in a kaggle kernel, skip the next command._

# In[ ]:


# os.chdir('/Users/steven/Documents/Kaggle/Titanic')


# <br>
# Now we check by using same command as before _(and we see it is correct because it prints out the directory we wanted)_:
# <br>
# <br>

# In[ ]:


os.getcwd()


# ***
# <br>
# ### 1B: Getting started: Having a looksy at the data ("EDA")
# 
# So now you are ready to start. You want to look a bit at the data. Two (there are a lot more) basic ways are;
# - 1: open your data files (csv's) with excel, save a copy as .xls and in excel go to the data tab and use the text to data wizard to use the comma's to seperate fields.<br>
# _the advantage of this is that after having a quick look you can you pivot tables functionality in excel to look deeper_
# - 2: show data in your jupyter notebook
# 
# 
# Let's assume you have already done number 1 (opening in excel and looking in the data, ideally using a pivot table). If you are starting at kaggle I am going to assume you are familiar with excel basics..
# For number 2 (data in jupyter notebooke/kaggle kernel) first step in to import a library that helps you work with csv files called 'pandas' _(this is your 'csv-reader' and allows you to create dataframes_ :
# <br>
# <br>

# In[ ]:


import pandas as pd


# In[ ]:


get_ipython().magic(u'pylab inline')
# the %pylab statement here is just something to make sure visualizations we will make later on 
# will be printed within this window..


# _Before continuing; in kaggle your data-files are located at  '../input' . You can check this by navigating to them by scrolling up and using the foldout button with the '+' logo called 'Input Files'_

# In[ ]:


train_df = pd.read_csv('../input/train.csv', header=0)
#above is the basic command to 'import' your csv data. Df is the new name for your 'imported' data 
#(df is short for dataframe, you can name this anyway you want, but including 'df' in your name is convention)

test_df = pd.read_csv('../input/test.csv', header=0)
#you don't have to use the test set but I am doing this to eveluate the model without uploading. You can slip this.

#Other options for this (splitting dataset to train part and test part) involve importing 'train_test_split' from sklearn. 
#I have not used this option, but perhaps it is 'easier'..

train_df.head(2)
#with df.head(2) we can 'test' by previewing the first(head) 2 rows(2) of the dataframe(df)
#You can see the final x by using 'df.tail(x)  (replace x with number of rows)


# In[ ]:


train_df
#show full dataset (df). (if very large this can be very inconvenient but with our trainset it's ok)
#notice it adds a rows total and columns total underneath (troubleshooting: if you do not see these totals
# you can seperatly create this by using 'df.shape')


# In[ ]:


#let's get slightly more meta. (data about the data, like what type is eacht variable ('column')?)
train_df.info()
#especially the information on the right is usefull at this point (the clomuns with values 'int64 and 'object' etc)
# These values describing each variable should be identical to that of the testset (which in this case being
# the Titanic datasets from Kaggle) they are. To test this you could repeat this procedure but use the test set instead
# of the train set.


# <br>
# _About these datatype names:_
# - int64 => whole numbers (can still be categorical)
# - object => string (can be categorical)
# - float64 => numeric with decimals (continous)
# <br>
# <br>
# 
# _There is a lot of ambiguity when expressing what type a variables is. 
# I think this is in part because 'datatypes' (above) are not the same as 'measurement levels'. In statistics a measurement level hierarchy is used to help you decide which analysis methods are appropriate. Added confusion arises because in some fields 'categorical' is the overlapping clustering contaning (a.o.) nominal, ordinal and ratio as subcategories, while in other fields categorical is synonomous for the ordinal measurement level._
# <br>
# <br>
# 
# - nominal (groups)
# - ordinal (groups with hierarchy)
# - interval (numbers with equal differences beteen them)
# - ratio (numbers with equal differences but also a absolute zero-point)

# In[ ]:


# More, More more!
#Let's fully dive in this meta data description of the variables:
train_df.describe()

# Notice that the decribe function only gives back non-'object' (7 out of the 12) variables..


# <br>
# Looking at our 'dependent variable' (i.e. 'survived') we see the average (mean) of .38 survival rate.<br>
# We know from the introduction on the problem (the kaggle description) _"...killing 1502 out of 2224..."_ so this seems about right because _1 - 1502/2224 = roughly 1/3 (.33)._ <br>
# Knowing our .38 is based on the training part of the data set while the given description is based on the total set it is close enough to state this is an honest sample.
# 
# Let's say you'd want the actual total of people who survived and the total of people who died (without calculating this back from the mean):
# <br>
# <br>

# In[ ]:


train_df.Survived.value_counts()
#the variable name (Survived) is with a capital letter because it has a capital letter in data set.
#'value_counts' is the 'smart' part, the function.


# _(So we know in our train data set, high over survival change = 342/(342+549) = .38 )_

# In[ ]:


train_df.Sex.value_counts().plot(kind='bar')
#you can replace the variable with any of the 12 (for some with more visual succes than for others..)


# <br>
# So we have come some way now.<br>
# It is time to really start segmenting and we have already made a start with gender (Sex).<br>
# We choose 'Sex'  to start with because A) it's easy and B) by looking at the data in excel(see beginning) we should have a reasonable suspicion that survival rate is not equal for men and woman. <br>Making this a sensible start for segmenting.
# <br>
# <br>
# Let's show the data for woman only:
# <br>
# <br>

# In[ ]:


train_df[train_df.Sex=='female']
# double == because were making a comparison not setting up for creating)


# In[ ]:


#before continuing let's do a quick check for 'missing values' (rows where gender is unknown)
# by using the 'isnull' function:
train_df[train_df.Sex.isnull()]


# _This shows up empty which is good news, saves us time _

# In[ ]:


# Let's visualize the number of survival amongst woman and later the number of survival amongst men to campare.
train_df[train_df.Sex=='female'].Survived.value_counts().plot(kind='bar', title='Survival among female')


# _Now we copy and past the command from above, and delete the 'fe' from 'female' (don't forget title) :_

# In[ ]:


train_df[train_df.Sex=='male'].Survived.value_counts().plot(kind='bar', title='Survival among male')


# ### Note that the bars are swapped. Our suspicion is confirmed (womand are more likely to have survived)
# _(meaning that in female segment survival occurred far more often than death. In male segment it is the other way around.)_
# <br>
# <br>

# In[ ]:


# The same can be done for age. Here it can also be interesting to combine age with sex;
train_df[(train_df.Age<11) & (train_df.Sex=='female')].Survived.value_counts().plot(kind='bar')
# '11' is just an arbitrarily chosen number as value of age.


# <br>
# _As you can see the combination of a low age and female (i.e. little girls) have a quite different (higher) survivalrate compared to the total trainset average survival rate (.38)_
# 
# Let's see if childeren regardless of gender ('Sex') also have better chances of survival:
# <br>
# <br>

# In[ ]:


train_df[(train_df.Age<11)].Survived.value_counts().plot(kind='bar')


# We can clearly see childeren(< 11 years) in general have better chances than the overal population (trainset) but not as good as childeren (< 11 years) that are also girls.

# <br>
# ### Visualize further using Seaborn
# <br>
# Import the python library seaborn (as sns)<br>
# <br>
# _You can also choose to do this in the beginning when importing pandas, so you get a list in the beginning with every library to be imported. To do it at the beginning is convention (and makes it easy for outsiders to see al used libraries at once) but for the purpose of this getting started tutorial I think it is better like this._

# In[ ]:


import seaborn as sns
# I don't know why seaborn is abbreviated as sns but you can choose anything you like as long as it is not used
# by anything else. Sns seems to be convention.


# <br>
# With seaborn we can more easily make barplots that show us more at once. For example we can take the variable Pclass and defining it as the x-axis, while makeing our dependent variable 'Survived' the y-axis, while differentiating between men and women (defining Sex as hue):
# <br>
# <br>

# In[ ]:


sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train_df);


# In[ ]:


#If we don't mind stereotypes ;p we could change the colors so that we don't have to look at the legend
# to remind us of the colorcoding for Sex:
#Just use the same command but add ' palette={"male": "blue", "female": "pink"} '


# In[ ]:


sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train_df, palette={"male": "blue", "female": "pink"});


# In reality we would have to repeat these visualisation command for all variables to see which variables might be intereseting for our model. <br>
# 
# For now we let's say we have repeated this for all variables. <br>
# This will result in you wanting to keep (at least):
# Sex, Age, Pclass, Cabin & Fares (and passengerID)

# <br>
# ## 3:  Data prepping & further visualization
# <br>
# Now that we have gotten familiarized with our data, we have to get the data in such a shape that we can use it.
# <br>
# <br>
# 
# This means dropping some features (variables) we don't want to use (Ticket, Name and Embarked), creating bins for other variables (Age and Fares) or changing some values of a variable to only the first letter (Cabins).
# <br>
# <br>
# 
# 

# In[ ]:


# Let's firs remove the variables we don't want:
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)


# In[ ]:


# make bins for ages and name them for ease:
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df


# In[ ]:


#keep only the first letter (similar effect as making bins/clusters):
def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df


# In[ ]:


# make bins for fare prices and name them:
def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df


# In[ ]:


# createa all in transform_features function to be called later:
def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = drop_features(df)
    return df


# In[ ]:


# create new dataframe with different name:
train_df2 = transform_features(train_df)
test_df2 = transform_features(test_df)


# <br>
# _Let's see what it looks like, see if everything has gone as planned:_
# <br>
# <br>

# In[ ]:


train_df2.head()


# Looks fine :)
# <br>
# <br>
# ### Now let's do some seaborn visualizations with our new dataset:
# _(3vars per plot)_

# In[ ]:


sns.barplot(x="Age", y="Survived", hue="Sex", data=train_df2, palette={"male": "blue", "female": "pink"});


# In[ ]:


sns.barplot(x="Cabin", y="Survived", hue="Sex", data=train_df2, palette={"male": "blue", "female": "pink"});


# In[ ]:


sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train_df2, palette={'male': 'blue', 'female': 'pink'});


# In[ ]:


from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
train_df2, test_df2 = encode_features(train_df2, test_df2)
train_df2.head()


# In[ ]:


train_df2.info()


# In[ ]:


X_train = train_df2.drop(["Survived", "PassengerId"], axis=1)
Y_train = train_df2["Survived"]
X_test  = test_df2.drop("PassengerId", axis=1).copy()




# I initially did not drop PassengerID. Keeping 8 variables ('features') in x-train and x-test. However, later on
# (during the modelling part) this resulted in an accuracy (for the random forests and classification trees) of 1.00
# Most likely I think it keeping PassengerId in this manner caused some form of label leakage. 
# After dropping this in both sets accuracy results were more realistic..

X_train.shape, Y_train.shape , X_test.shape


# <br>
# _Make sure that X-train and X-test have the same amount of variables ('features'; in this example '7').._
# <br>
# <br>
# ### Almost ready to try some models  ('sci-kitlearn')
# <br>
# <br>
# Being: <br>
# 1) regression 
# 2) decision tree and 
# 3) random forests.<br>
# <br>
# <br>
# #### _first a quick looksy at the X-train and Y-train_:

# In[ ]:


X_train.head()


# In[ ]:


Y_train.head()


# ***
# <br>
# ## 4 Modelling
# <br>
# <br>
# Ready to try some models:
# <br>
# <br>
# - 1) logistic regression 
# - 2) decision tree
# - 3) random forests
# <br>
# <br>

# In[ ]:


# Logistic Regression

# Import from the the scikit-learn library (sklearn is the abbreviation for scikit-learn)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


# Decision Tree

# Import from the the scikit-learn library (sklearn is the abbreviation for scikit-learn)
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


# Random Forest

# Import from the the scikit-learn library (sklearn is the abbreviation for scikit-learn)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest


# In[ ]:


#Creating a csv with the predicted scores (Y as 0 and 1's for survival)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

# But let's print it first to see if we don't see anything weird:


# In[ ]:


submission.describe()


# ***
# <br>
# ## 5 Submission
# <br>
# <br>
# #### All looks fine. Let's turn it into a CSV and save it in a logical local place and upload it to Kaggle to find out real score (i.e. the score on their test set)
# <br>
# <br>

# In[ ]:


os.getcwd()


# In[ ]:


#submission.to_csv('../pathhere../submission.csv', index=False)


# <br>
# Aaaaaaand we have a 0.7512 score. Not too bad for a very first model.<br>
# _(Not too well either because guessing at random would not be .5 but .62. ('mean as model'))_<br>
# <br>
# <br>
# One of the advantages of starting out with a very basic model is that all feature engineering and inmprovement can be measured in accuracy gains against time spent..
# <br>
# <br>
# Since we haven't done any futher optimization like feature engineering yet, now it would be the time to start improving this 'base' model. _(// the titles in the passenger name variable would be a good start.)_
# <br>
# <br>
# ***
# <br>
# _Please feel free to comment below, I will read and if possible incorporate them and please give tips were you think needed. I will try to learn from users' suggestions and tips:)_
# <br>

# In[ ]:




