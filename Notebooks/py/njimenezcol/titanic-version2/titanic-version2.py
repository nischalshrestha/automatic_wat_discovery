#!/usr/bin/env python
# coding: utf-8

# This notebook is a slightly modified version of the code published by Jonathan Bechel in "Mastering the Basics on the RMS Titanic! 

# **Step 1:  Load the Data.**
#  
# Given that later on I concatenate the training and testing data sets, the 'PassengerId' column is used as the index to correctly identify the passengers

# In[ ]:


#Data analysis
import pandas as pd
from pandas import Series,DataFrame

import numpy as np

#Graphics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

#Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm


# In[ ]:


#Read files into the program
test = pd.read_csv("../input/test.csv", index_col='PassengerId')
train = pd.read_csv("../input/train.csv", index_col='PassengerId')


# In[ ]:


print ("Basic statistical description:")
train.describe()


# In[ ]:


train.info()


# ##Some graphical analysis

# In[ ]:


#Age
#Survived vs not survived by age
Age_graph = sns.FacetGrid(train, hue="Survived",aspect=3)
Age_graph.map(sns.kdeplot,'Age',shade= True)
Age_graph.set(xlim=(0, train['Age'].max()))
Age_graph.add_legend()

# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = train[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)


# ##Fare variable
# Graphical analysis and imputation

# In[ ]:


# Since there is a missing value in the "Fare" variable, I imputed using the median
test["Fare"].fillna(test["Fare"].median(), inplace=True)

# convert from float to int
train['Fare'] = train['Fare'].astype(int)
test['Fare']  = test['Fare'].astype(int)

# Fare for passengers that survived & didn't survive  
fare_not_survived = train["Fare"][train["Survived"] == 0]
fare_survived     = train["Fare"][train["Survived"] == 1]

# Average and std for survived and not survived passengers
avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])

# Histogram 'Fare'
train['Fare'].plot(kind='hist', figsize=(8,3),bins=100, xlim=(0,50))


# We're going to do three things to start off:
# 
#  - Store the 'Survived' column as its own separate series and delete it from the 'Train' dataset.
#  - Concatenate the training and testing set to fill in and parse all the
#    data at once. 
#  - Drop two columns: 'Embarked' and 'Ticket.'

# In[ ]:


y = train['Survived']
del train['Survived']


# ##Concatenate the training and testing data sets

# In[ ]:


train = pd.concat([train, test])


# In[ ]:


#Drop variables that we will not included in the model: (6)'Embarked' and (9) 'Ticket.'
train = train.drop(train.columns[[6,9]], axis=1)


# #Categorical Data: Encoding and Feature Generation

# ##Handling categorical data with sklearn
# 
# **Using sklearn Lablel Encode** 
#    
#  - It assigns numeric values against each categorical variable in the data and add the column wise in the data frame. 
#  - Sklearn label encoder can handle numeric categories, while Pandas can  also handle strings. Pandas get_dummies function
# 
# Let's do that for PClass and Sex:

# In[ ]:


#fit_transform Encode labels with value between 0 and n_classes-1. 
train['Sex'] = LabelEncoder().fit_transform(train.Sex)
train['Pclass'] = LabelEncoder().fit_transform(train.Pclass)


# For the 'Cabin' feature we're going to first do a little bit of data transformation.
# To see we're going to extract the first letter of each passenger's cabin (if it exists) using the 'lambda x' feature in Python, and then encode it.
# We change the np.nan values to 'X' so all data is the same type, allowing it to be labeled.

# In[ ]:


train['Cabin'] = train.Cabin.apply(lambda x: x[0] if pd.notnull(x) else 'X')
train['Cabin'] = LabelEncoder().fit_transform(train.Cabin)


# In[ ]:


train[['Sex','Pclass', 'Cabin']][0:3]


# **Missing data**

# In[ ]:


train.info()


# *****************

# Clearly there's an important amount of missing data in the 'Age' category. To fill it we're going to use the median age of that passengers Class and Sex, which will be accessed via the groupby method in Pandas:

# **String processing**

# Extract the labels associated with a person's greeting using "string processing"  though a for loop by using the Python method split() to break up each greeting

# In[ ]:


#Used to create new pd Series from Name data that extracts the greeting used for their name to be used 
#as a separate variable
def greeting_search(words):
    for word in words.split():
        if word[0].isupper() and word.endswith('.'): #name into an array of "words" 
                                  #These are evaluate using the isupper() and endswith() methods in a for loop
            return word


# In[ ]:


# apply the greeting_search function to the 'Name' column
train['Greeting']=train.Name.apply(greeting_search)
train['Greeting'].value_counts()


# In[ ]:


#greetings that occur 8 or less times and classify them under the moniker 'Rare',
train['Greeting'] = train.groupby('Greeting')['Greeting'].transform(lambda x: 'Rare' if x.count() < 9 else x)

del train['Name']   

#tranform the data and drop the 'Name' series since it's no longer needed.
train['Greeting'] = LabelEncoder().fit_transform(train.Greeting)


# In[ ]:


train.info()


# There is missing data for the 'Age' and 'Cabin' variables

# ##Missing Data
# Considering that there is a lot of missing information for the 'Age' variable we are going to 
# impute it using the median age of  Greeting and Sex.

# In[ ]:


#This will be accessed via the groupby method in Pandas:
train.groupby(['Greeting', 'Sex'])['Age'].median()


# In[ ]:


#set using Lambda x
train['Age'] = train.groupby(['Greeting', 'Sex'])['Age'].transform(lambda x: x.replace(np.nan, x.median()))


# As we wil see next, there is a NaN value for the 'Fare' variable in row 1043 which corresponds to PassengerId '1044'. 

# In[ ]:


train[1042:1044]


# In[ ]:


train.iloc[1043, 5] = 7.90 #Imputation of Fare using iloc
train[1042:1044]


# Feature Generation: Family Size and Greeting

# In[ ]:


train['Family_Size'] = train.SibSp + train.Parch


# In[ ]:


train['Family_Size'][0:15]


# **Categorical coding** use Pandas **pd.get_dummies**

# CONTINUOUS ORDER has a precise hierarchy to it. Someone who paid  50foraticketdefinitelypaidmorethansomeonewhopaid50foraticketdefinitelypaidmorethansomeonewhopaid 30.
# So what we want to do is re-code a Series into a package of yes/no decisions demarcated as 0 or 1 depending on which option they were.
# Ie, Someone's Passenger class should be denoted as [0, 0, 1], [1, 0, 0], or [0, 1, 0] depending on which of the three classes they are.
# Pandas has a useful tool to do this called pd.get_dummies, which takes a series of encoded and then unpacks it into the appropriate number of yes/no columns.
# For example, we can take the 'Pclass' series and use pd.get_dummies like this:

# In[ ]:


Pclass = pd.get_dummies(train['Pclass'], prefix='Passenger Class', drop_first=True)


# In[ ]:


Pclass.head(5)


# Important: You might notice there's an option called 'drop_first' which is set to 'True.'
# That means the first variable in the series is excluded, which is important for avoiding something called collinearity, which you can read more about here.
# To be honest, probably not that important for this dataset, but a useful habit to keep in mind, especially if you work with Time Series data.

# In[ ]:


Greetings = pd.get_dummies(train['Greeting'], prefix='Greeting', drop_first=True)
Cabins = pd.get_dummies(train['Cabin'], prefix='Cabin', drop_first=True)


# In[ ]:


train.info()


# **Standardizing your Data**

# It's good practice to standardize the data in order to allow different data sets to be comparable. When standardizing your data the idea is to compute the mean and subtract it from your data. Then divide the results by the standard deviation. By doing so, we can compare the data distribution with a normal distribution (N(0,1)N(0,1))
# 
# [What is the purpose of subtracting the mean from data when standardizing?][1]
# 
# 
#   [1]: https://math.stackexchange.com/questions/317114/what-is-the-purpose-of-subtracting-the-mean-from-data-when-standardizing

# In[ ]:


#Scale Continuous Data
train['Family_scaled'] = (train.Family_Size - train.Family_Size.mean())/train.Family_Size.std()
train['Age_scaled'] = (train.Age - train.Age.mean())/train.Age.std()
train['Fare_scaled'] = (train.Fare - train.Fare.mean())/train.Fare.std()


# **Final steps**
# 
#  1. Drop the columns that we rae not gonna use in the analysis
#  2. Concatenate the dataframes that were created with pd.get_dummies()
#  3. split the data back into its training and test sets 

# In[ ]:


train.info()


# In[ ]:


train = train.drop(train.columns[[0,2,3,4,5,6,7,8]], axis=1)
#Varibles that I dropped: Pclass, Age,SisSp, Parch, Fare, Cabin, Greeting,Family Size
#Pclass	Sex	Age	SibSp	Parch	Fare	Cabin	Greeting	Family_Size	Family_scaled	Age_scaled	Fare_scaled


# In[ ]:


train.info()


# In[ ]:


#Concat modified data to be used for analysis, set to X and y values
data = pd.concat([train, Greetings, Pclass, Cabins], axis=1)


# In[ ]:


data.info()


# In[ ]:


#Split the data back into its original training and test sets
test = data.iloc[891:]
X = data[:891]


# Cross-validation

# In[ ]:


clf = LogisticRegression()


# In[ ]:


def find_C(X, y):
    Cs = np.logspace(-4, 4, 10)
    score = []  
    for C in Cs:
        clf.C = C
        clf.fit(X_train, y_train)
        score.append(clf.score(X, y))
  
    plt.figure()
    plt.semilogx(Cs, score, marker='x')
    plt.xlabel('Value of C')
    plt.ylabel('Accuracy on Cross Validation Set')
    plt.title('What\'s the Best Value of C?')
    plt.show()
    clf.C = Cs[score.index(max(score))]
    print("Ideal value of C is %g" % (Cs[score.index(max(score))]))
    print('Accuracy: %g' % (max(score)))


# In[ ]:


find_C(X_val, y_val)


# In[ ]:


#Create cross - validation set 
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.6)


# In[ ]:


clf = LogisticRegression()


# In[ ]:


def find_C(X, y):
    Cs = np.logspace(-4, 4, 10)
    score = []  
    for C in Cs:
        clf.C = C
        clf.fit(X_train, y_train)
        score.append(clf.score(X, y))
  
    plt.figure()
    plt.semilogx(Cs, score, marker='x')
    plt.xlabel('Value of C')
    plt.ylabel('Accuracy on Cross Validation Set')
    plt.title('What\'s the Best Value of C?')
    plt.show()
    clf.C = Cs[score.index(max(score))]
    print("Ideal value of C is %g" % (Cs[score.index(max(score))]))
    print('Accuracy: %g' % (max(score)))


# In[ ]:


find_C(X_val, y_val)


# In[ ]:


answer = pd.DataFrame(clf.predict(test), index=test.index, columns=['Survived'])
answer.to_csv('answer.csv')


# In[ ]:


coef = pd.DataFrame({'Variable': data.columns, 'Coefficient': clf.coef_[0]})
coef

