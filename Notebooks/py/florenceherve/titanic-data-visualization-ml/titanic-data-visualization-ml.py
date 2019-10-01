#!/usr/bin/env python
# coding: utf-8

# This notebook is part of the final project of the Udacity course "Intro to Data Analysis". In this notebook I'll apply the concepts learned in class to visualize the Titanic data and test some hypotheses. In a second part, I'll set up a machine learning algorithm to try to predict the survival of a passenger using the information we have in the database.
# I used the following notebooks for help... thanks a ton to their authors !!
# 1. 'A Journey through Titanic' https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic/
# 2. 'Machine Learning for Survival Prediction' https://www.kaggle.com/skywalkerhc/titanic/machine-learning-for-survival-prediction-2
# 3. 'Kaggle-Titanic-001' https://www.kaggle.com/michielkalkman/titanic/kaggle-titanic-001

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# **1. Set Up**

# In[ ]:


# Imports
import pandas as pd
from pandas import Series,DataFrame

# unicodecsv
import unicodecsv

# numpy, matplotlib, seaborn
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
import numpy as np

# machine learning
import sklearn
from time import time
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn import datasets, linear_model
from sklearn.naive_bayes import GaussianNB #for Naive Bayes
from sklearn import svm #for SVM
from sklearn.svm import SVC #for SVM with 'rbf' kernel
from sklearn import tree #for decision trees


# In[ ]:


# Choose color palette for the seaborn graphs for the rest of the notebook:
# Import color widget from seaborn:
sns.choose_colorbrewer_palette (data_type='sequential', as_cmap=False)
# Set color palette below:
sns.set_palette("YlGnBu", n_colors=5, desat=1, color_codes=False)

# Matplotlib color codes can be found here: http://stackoverflow.com/questions/22408237/named-colors-in-matplotlib


# Read the file and clean the data

# In[ ]:


titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")
# Preview the data
titanic_df.head()


# In[ ]:


titanic_df.info()


# From this I see that we have 891 rows in total, with 3 fields that are not populated everywhere: Age (714 rows filled), Cabin (only 204 rows) and Embarked (889 rows). Not all fields are in the right format as well ('Sex' for instance is an object), I will take care of this too in the next few steps.
# 
# From an overview of the columns and fields available in titanic_df, it looks like the information in some columns is not relevant to our analysis of predicting the survival of a passenger. I would consider those fields to be irrelevant for now:
# 1. Name
# 2. Ticket
# 3. Cabin
# 4. Embarked
# 
# I am going to drop those fields in the next step. Then I'll proceed to standardize the remaining data in the dataset before starting the analysis.

# In[ ]:


# Removing useless fields:
titanic_df = titanic_df.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
test_df = test_df.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
titanic_df.info()


# In[ ]:


# Changing the 'Sex' field from string to integer: 0 for male and 1 for female
titanic_df.loc[titanic_df["Sex"] == "male", "Sex"] = 0
titanic_df.loc[titanic_df["Sex"] == "female", "Sex"] = 1
test_df.loc[test_df["Sex"] == "male", "Sex"] = 0
test_df.loc[test_df["Sex"] == "female", "Sex"] = 1
# Convert to int
titanic_df['Sex'] = titanic_df['Sex'].astype(int)
test_df['Sex'] = test_df['Sex'].astype(int)


# The biggest question now is what to do with the 'Age' field. 177 rows are not populated; excluding them from the analysis will greatly reduce the size of the dataset and possibly impair the training and testing of the machine learning classifiers. On the other hand, assigning random values to those fields might create a bias in the results. After reading through previous submissions I decided to create a new column, 'Age_filled', and fill the empty rows or N/As there with the median age of the passengers. I'll use this column for the classifiers in part 3., but the data visualization could still be done with the original values if needed.

# In[ ]:


# Duplicate 'Age' column (for train dataset only)
titanic_df['Age_filled'] = titanic_df['Age']

# Fill N/As in 'Age_filled' column with median values
titanic_df['Age_filled'] = titanic_df['Age_filled'].fillna(titanic_df['Age'].median())
test_df['Age_filled'] = test_df['Age'].fillna(test_df['Age'].median())
print ("The median age for Titanic passengers is: ", titanic_df['Age'].median())

# Convert from float to int
titanic_df['Age_filled'] = titanic_df['Age_filled'].astype(int)
test_df['Age_filled'] = test_df['Age_filled'].astype(int)


# In[ ]:


# The two missing values in the 'Fare' column can also be filled with median values. 
# Fill N/As in 'Fare' column with median values
print ("The median fare for Titanic passengers is: ", titanic_df['Fare'].median())
titanic_df['Fare'] = titanic_df['Fare'].fillna(titanic_df['Fare'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

titanic_df.info()


# Query the data & Getting used to Panda Dataframes

# In[ ]:


# Survival function for counting survivors:
def Survival(dataframe, field, value):
    count = 0
    for index, row in dataframe.iterrows():
        if field == 0 and value == 0:
            if row['Survived'] == 1:
                count += 1
        if row[field] == value and row['Survived'] == 1:
            count +=1
    return count

print (Survival(titanic_df, 0, 0))
print (Survival(titanic_df, "Sex", 0))


# In[ ]:


# Looking for Jack & Rose:
potential_Jacks = []
potential_Roses = []

for index, row in titanic_df.iterrows():
    if row['Sex'] == 0 and row['Pclass'] == 3 and row['Survived'] == 0 and row['Age'] == 20:
        potential_Jacks.append(row)
    if row['Sex'] == 1 and row['Pclass'] == 1 and row['Survived'] == 1 and row['Age'] == 17:
        potential_Roses.append(row)
print ("There are %s potential Jacks in our dataset" % (len(potential_Jacks)))
print ("There are %s potential Roses in our dataset" % (len(potential_Roses)))
print ("---------")

# Printing out different passengers:
oldest_passenger = titanic_df.loc[titanic_df['Age'].idxmax()]
print ("The oldest passenger was %s years old." % oldest_passenger['Age'])
youngest_passenger = titanic_df.loc[titanic_df['Age'].idxmin()]
print ("The youngest passenger was %s years old." % youngest_passenger['Age'])
print ("---------")

# Main fares:
print ("The cheapest fare cost %s dollars." % titanic_df.loc[titanic_df['Fare'].idxmin()]['Fare'])
print ("The most expensive one was %s dollars." % titanic_df.loc[titanic_df['Fare'].idxmax()]['Fare'])
print ("The mean fare was %s dollars." % titanic_df['Fare'].mean())


# **2. Data Analysis and Visualization**

# I would consider the following fields to be good predictors of survival on the Titanic - Let's investigate those one by one:
# Gender "Sex"
# Class "Pclass"
# Age
# Number of siblings on board "SibSp"
# Number of children on board "Parch"

# *a. Gender*

# In[ ]:


female_live = Survival(titanic_df, "Sex", 1)
male_live = Survival(titanic_df, "Sex", 0)
female_all = 0
male_all = 0

for index, row in titanic_df.iterrows():
    if row['Sex'] == 1:
        female_all +=1
    if row['Sex'] == 0:
        male_all +=1

female_die = female_all - female_live
male_die = male_all - male_live
print ('Survival rate / female: ', float(female_live) / float(female_all))
print ('Survival rate / male: ', float(male_live) / float(male_all))


# In[ ]:


# Original tutorial on http://matplotlib.org/examples/pylab_examples/bar_stacked.html

# Version 1: show bars as 'Survived' and 'Did not survive'
N = 2
women_data = (female_live, female_die)
men_data = (male_live, male_die)
ind = np.arange(N)    # the x locations for the groups
width = 0.45       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, men_data, width, facecolor='darkslateblue', edgecolor='white', align='center')
p2 = plt.bar(ind, women_data, width, facecolor='teal',edgecolor='white', bottom=men_data, align='center')

plt.ylabel('# of people')
plt.xticks(ind + width/2., ('Survived', 'Did not survive'))
plt.legend((p1[0], p2[0]), ('Men', 'Women'), loc="upper left")
plt.show()

# Version 2: show bars as 'Men' and 'Women'
N = 2
live_data = (female_live, male_live)
die_data = (female_die, male_die)
ind = np.arange(N)    # the x locations for the groups
width = 0.45       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, live_data, width, facecolor='darkslateblue', edgecolor='white', align='center')
p2 = plt.bar(ind, die_data, width, facecolor='teal', edgecolor='white', bottom=live_data, align='center')

plt.ylabel('# of people')
plt.xticks(ind + width/2., ('Women', 'Men'))
plt.legend((p1[0], p2[0]), ('Survived', 'Did not survive'), loc="upper left")
plt.show()


# There seem to be a much better survival rate among women than among men (74.2% vs 18.9%). Two possible explanations are: 1) "Women and children first" policy on lifeboats and 2) A higher proportion of women on 1st and 2nd class (which I anticipate have higher survival rates than 3rd class). Let's now investigate the survival rates within classes.

# *b. Passenger Class*

# In[ ]:


#Divide the Titanic data set into different classes
def divide_pclass(dataframe, x):
    new_pclass = 0
    for index, row in dataframe.iterrows():
        if row['Pclass'] == x:
            new_pclass+= 1
    return new_pclass

first_class = divide_pclass(titanic_df, 1)
second_class = divide_pclass(titanic_df, 2)
third_class = divide_pclass(titanic_df, 3)

#Check that all rows in the data set have been allocated to a class set:
print ("First class: ", first_class, " Second class: ", second_class, " Third class: ", third_class)
print ("Total number of passengers (should match to 891): ", first_class + second_class + third_class)


# In[ ]:


from decimal import *
getcontext().prec = 4

# Print out basic distribution and survival rates accross classes:
first_survived = Survival(titanic_df, "Pclass", 1)
second_survived = Survival(titanic_df, "Pclass", 2)
third_survived = Survival(titanic_df, "Pclass", 3)

ratio_first_survival = Decimal(first_survived) / Decimal(first_class)
ratio_second_survival = Decimal(second_survived) / Decimal(second_class)
ratio_third_survival = Decimal(third_survived) / Decimal(third_class)

# Note: use %s (and not %d) to show decimal places in survival rates
print ("%s of the first class passengers lived, meaning a survival rate of %s" % (first_survived, Decimal(ratio_first_survival)))
print ("Regarding the second class, %s lived, meaning a survival rate of %s" % (second_survived, Decimal(ratio_second_survival)))
print ("Finally, %s of the third class passengers lived, meaning a survival rate of %s" % (third_survived, Decimal(ratio_third_survival)))


# In[ ]:


# Simple bar charts from kaggle notebook https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic/
fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

# Version 1: simple count
sns.countplot(x='Pclass', data=titanic_df, order=[1,2,3], ax=axis1)

# Version 2: with mean
# In dataframe: group by class, and get the mean for survived passengers for each value in Class
pclass_surv_mean = titanic_df[["Pclass", "Survived"]].groupby(['Pclass'],as_index=False).mean()
# Plot the total number of passengers per class
sns.barplot(x="Pclass", y="Survived", data=pclass_surv_mean, label="Total number of passengers", order=[1,2,3], ax=axis2)


# *c. Age*

# In[ ]:


# I'm using the 'Age' column (with NAs) since it is the original data
mean_age = titanic_df['Age'].mean()
median_age = titanic_df['Age'].median()
std_age = titanic_df['Age'].std()
print ("Key metrics:")
print ("The mean age for Titanic passengers is %s; the median is %s and the std is %s."% (mean_age, median_age, std_age))

# Plot Age values on an histogram
titanic_df['Age'].hist(bins=80) #bins=80 as ages range from 0 to 80 years old

plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# *d. Number of siblings and children*

# I'll combine the 'SibSp' and 'Parch' datapoints into a 'Family' indicator. The goal here would be to identify if having a family on board made you more or less likely to survive.

# In[ ]:


# Create a new 'Family' column in the dataframe:
titanic_df['Family'] =  titanic_df['Parch'] + titanic_df['SibSp']
test_df['Family'] =  test_df['Parch'] + test_df['SibSp']

# This column sums up the values in 'Parch' and 'SibSp' columns. 
#I would like to make it a dummy variable, 0 for no family onboard and 1 for one.
titanic_df.loc[titanic_df['Family'] >= 1, 'Family'] = 1
titanic_df.loc[titanic_df['Family'] == 0, 'Family'] = 0
test_df.loc[test_df['Family'] >= 1, 'Family'] = 1
test_df.loc[test_df['Family'] == 0, 'Family'] = 0
# Convert to int
titanic_df['Family'] = titanic_df['Family'].astype(int)
test_df['Family'] = test_df['Family'].astype(int)


# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

# Version 1: simple count
sns.countplot(x='Family', data=titanic_df, order=[1,0], ax=axis1)

# Version 2: with mean
# In dataframe: group by class, and get the mean for survived passengers for each value in Class
family_surv_mean = titanic_df[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
# Plot the total number of passengers per class
sns.barplot(x="Family", y="Survived", data=family_surv_mean, order=[1,0], ax=axis2)

axis1.set_xticklabels(["Family","Alone"], rotation=0)


# Those conclusions could be drawn from the 'Women and children first' rules for the lifeboats. Next I decide to isolate the types of passengers less likely to survive - that is, adult men - and see if the adult men with a family onboard were indeed more likely to survive.

# In[ ]:


# Dividing men above 18 years old into two groups:
men_alone = 0
men_family = 0
men_alone_1 = 0
men_family_1 = 0

for index, row in titanic_df.iterrows():
    if row['Sex'] == 0 and row['Family'] == 0 and row['Age'] >= 18:
        men_alone += 1
        if row['Survived'] == 1:
            men_alone_1 += 1
    if row['Sex'] == 0 and row['Family'] == 1 and row['Age'] >= 18:
        men_family += 1
        if row['Survived'] == 1:
            men_family_1 += 1
print ("There are %s men alone and %s men with family in our dataset." % (men_alone, men_family))
print ("---------")

# Survival rates in the two groups:
print ('Survival rate / men alone: ', float(men_alone_1) / float(men_alone))
print ('Survival rate / men with family: ', float(men_family_1) / float(men_family))


# **3. Machine Learning**

# Dividing the data into test and train datasets

# In[ ]:


"""
Dividing the data between training and testing datasets: 
1. Removing 'Survived' from the features (as this is what we are trying to predict)
2. Removing 'Age' column to keep only 'Age_filled'
""" 
features_train = titanic_df.drop(["Survived", "Age"],axis=1)
labels_train = titanic_df["Survived"]
features_test  = test_df.drop("Age", axis=1).copy()


# *a. Logistic Regression*

# In[ ]:


"""
from sklearn import datasets, linear_model

logreg = linear_model.LogisticRegression()
logreg.fit(features_train, labels_train)

#print('Coefficient:', logreg.coef_)
print('Intercept:', logreg.intercept_)
print('Score on train data:', logreg.score(features_train, labels_train))
#print('Score on test data:', logreg.score(features_test, labels_test))

# get Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(titanic_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
print (coeff_df)
"""


# *b. Naive Bayes*

# In[ ]:



"""
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print ("prediction time:", round(time()-t0, 3), "s")
print ("---------")

print (clf.score(features_train, labels_train))
#print ("Accuracy score: ", accuracy_score(pred, labels_test))

#precision = precision_score(labels_test, pred)
#print ("Precision: ", precision)

#recall = recall_score(labels_test, pred)
#print ("Recall: ", recall)

# Calculate F1 score:
#f1 = 2 * (precision * recall) / (precision + recall)
#print ("The F1 score is: ", f1)
"""


# *c. SVM*

# In[ ]:


"""
from sklearn import svm

clf = svm.SVC(kernel = 'linear')
t0 = time()
clf.fit(features_train, labels_train) 
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print ("prediction time:", round(time()-t0, 3), "s")
print ("---------")

print (clf.score(features_train, labels_train))
"""


# *d. Decision Tree*

# In[ ]:


# Adjusting min_samples_split

from sklearn import tree

clf = tree.DecisionTreeClassifier(min_samples_split=10)

t0 = time()
clf = clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print ("prediction time:", round(time()-t0, 3), "s")
print ("---------")

print (clf.score(features_train, labels_train))


# In[ ]:


"""
#print clf.feature_importances_
important_features = []
for x,i in enumerate(clf.feature_importances_):
    if i>0.2:
        important_features.append([str(x),str(i)])
        #print (x,i)
    print (important_features)
"""


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": pred
    })
submission.to_csv('titanic.csv', index=False)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": pred
    })
submission.to_csv('titanic.csv', index=False)

