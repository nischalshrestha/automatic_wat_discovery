#!/usr/bin/env python
# coding: utf-8

# # Introduction #
# 
# This is my first Kaggle notebook.  So I chose the practice competition [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) .  I used  python for this and used **Random Forest** model.    
# 
# # Content #
#   1. Understanding Data
#   2. Data Management and Visualisation
#   3. Modelling And Implementation
#   4. Conclusion
# 
# ## Understanding Data ##
# ### 1.1. Variable Description
# There are total eleven variables(or column) in titanic data-set.
# 
# + Survived : 
#    0 = No and 1 = Yes
# + pclass :
#            Passenger class is a proxy for socio-economic status (SES)\
#            1st ~ Upper;  2nd ~ Middle;  3rd ~ Lower 
# + name : 
#             Name of passenger
# + sex : 
#            Sex of passenger
# + age :
#            Age of passenger. Age is in Years.Fractional if Age less than One (1)
# + sibsp :
#             Number of Siblings/Spouses Aboard
# + parch :
#             Number of Parents/Children Aboard
# + ticket :
#             Ticket Number
# + fare :
#             Passenger Fare
# + cabin :
#             Cabin of ship
# + embarked : 
#             Port of Embarkation  
#                 (C = Cherbourg; Q = Queenstown; S = Southmpton)

# In[3]:


import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')
import warnings
warnings.filterwarnings('ignore')

# For loading the data
train_data = pandas.read_csv('../input/train.csv', low_memory = False)
test_data = pandas.read_csv('../input/train.csv', low_memory = False)

# Set Pandas to show all columns in DataFrames
pandas.set_option('display.max_columns', None)

# Set Pandas to show all rows in DataFrames
pandas.set_option('display.max_rows', None)

# upper-case all DataFrame column names
train_data.columns = map(str.upper, train_data.columns)
test_data.columns = map(str.upper, test_data.columns)

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%f'%x)

# train_data and test_data both are subset of the titanic data.
# So getting some basic information of the data from train_data as both of them are similar

# Getting heading of columns of data
print(train_data.head())

# Getting some details of data
print(train_data.describe())  


# ### 1.2. Variable Selection
# Out of eleven variables, 'Survived' is target variable. Now to choose explanatory variables, we will choose only pclass, sex, age, embarked, sibsp and parch\.
# 
# Passenger class is a proxy for socio-economic status. So upper class being rich or powerful have easy access to lifeboats and hence more chance of survival compare to other class\.
# 
# Females are more likely to get lifeboat than the male passenger hence more probability of survival. Similarly children are more likely to be boarded on lifeboats while old peoples who have already lived most of their life are less likely to board the lifeboats\.
# 
# Importance of embarked,parch and sibsp variables will be visualised in next section\.
# 
# Including variable 'name' in explanatory variable will not make a sense because we can not interpret survival of parson from their names excluding some particular names of celebrities,politician or wealthy businessman. Similarly including variable 'ticket' in to explanatory variable will not make sense as mostly ticket numbers will be unique and also their are many missing data which is very difficult to fill in\.
# 
# Cabin can be included but it have many missing data and filling those missing data is very tough\.
# 
# Variable 'fare' seems like that it should be included in explanatory variable but if you analyse this variale deeply, you will realise that including this variable will result in bad model as 'fare' is  confounded by 'age', 'embarked', 'pclass'  or may be 'sex'. Fare of a passenger will depend on age (children have low fare), port of embarkation(longer you travel more you have to pay),  passenger class(higher is the class more is the money) and also there can be a case that females were given some discounts. So 'fare' contains the information other variable which we have already included. This hypothesis can be proved by Inferential analysis but I am not going in that much detail.

# In[4]:


# Setting variable to numeric value as mostly all pandas operation/method work on numeric variable
train_data['SURVIVED'] = pandas.to_numeric(train_data['SURVIVED'], errors='coerce')
train_data['PCLASS'] = pandas.to_numeric(train_data['PCLASS'], errors='coerce')
train_data['AGE'] = pandas.to_numeric(train_data['AGE'], errors='coerce')
train_data['SIBSP'] = pandas.to_numeric(train_data['SIBSP'], errors='coerce')
train_data['PARCH'] = pandas.to_numeric(train_data['PARCH'], errors='coerce')

# Making new data set from given dataset with variables(column) which seem relevent in predicting the chance of survival
sub = train_data[['SURVIVED', 'PCLASS', 'AGE', 'SIBSP', 'PARCH', 'EMBARKED', 'SEX']]
print(sub.describe())


# ## Data Management and Visualisation ##
# ### 2.1. Data Management
# There are some missing data in variable 'age' which are set to the median value of age. Similarly missing data in 'embarked' are set to 'S' as 'S' has highest frequency.
# Quantitative variable age is categorised in to five categories - CHILDREN, ADOLESCENTS, ADULTS, MIDDLE AGE, OLD for better visualisation.
# 

# In[5]:


# Replacing unknown age by median of 'AGE' variable
sub['AGE'] = sub['AGE'].replace(numpy.nan, sub['AGE'].median())

# EMBARKED variable has some missing data. So filling it with category 'S' as this has most frequency
sub['EMBARKED'] = sub['EMBARKED'].fillna('S')

# Creating new variable for better visualization of age vs survival
def AGEGROUP(row):
    if (row['AGE'] > 60) :
        return 5        
    elif (row['AGE'] > 45) :
        return 4        
    elif (row['AGE'] > 19) :
        return 3   
    elif (row['AGE'] > 9) :
        return 2 
    elif (row['AGE'] >= 0) :
        return 1    
sub['AGEGROUP'] = sub.apply(lambda row: AGEGROUP (row),axis=1) 


# ### 2.2. Data Visualisation

# In[6]:


# Now visualization of 'SURVIVED'
# Printing counts and percentage of diffrent survival
print(sub['SURVIVED'].value_counts(sort=False))
print(sub['SURVIVED'].value_counts(sort=False,normalize=True))

# Making variable  categorical 
sub['SURVIVED'] = sub['SURVIVED'].astype('category')

# Visualising counts of survival with bar graph
seaborn.countplot(x="SURVIVED", data=sub);
plt.xlabel('Survival Status')
plt.ylabel('Frequency')
plt.title('Count of Survival')

# Converting 'SURVIVED' to numeric value for further operations
sub['SURVIVED'] = pandas.to_numeric(sub['SURVIVED'], errors='coerce')


# Approx 62% of passenger died. 

# In[10]:


# Printing counts and percentage of category of Age Group
print(sub['AGEGROUP'].value_counts(sort=False))
print(sub['AGEGROUP'].value_counts(sort=False,normalize=True))

# Making variable AGEGROUP categorical and naming category
sub['AGEGROUP'] = sub['AGEGROUP'].astype('category')
sub['AGEGROUP'] = sub['AGEGROUP'].cat.rename_categories(["CHILDREN", "ADOLESCENENTS", "ADULTS", "MIDDLE AGE", "OLDS"])

# Visualising counts of Age Group with bar graph
seaborn.countplot(x="AGEGROUP", data=sub);
plt.xlabel('Age Group')
plt.ylabel('Frequency')
plt.title('Age group Distribution   ')

# Showing proportion of survival of different groups by plot
seaborn.factorplot(x="AGEGROUP", y="SURVIVED", data=sub, kind="bar", ci=None)
plt.xticks(rotation=90)
plt.xlabel('AGE GROUP')
plt.ylabel('Survive Percentage')
plt.title('Survive v/s Age Group')


# 50% of passenger are in adult category(20 - 45 years).  7% of passengers are children(0 - 9 years)  out of which 60% has survived while 2.5% of passenger are old( more 60 years) of which only 20% survived. So this variable matching my assumption as discussed in Variable selection section.

# In[ ]:


# Now visualization of 'SIBSP'
# Printing counts and percentage of number of siblings and spouse
print(sub['SIBSP'].value_counts(sort=False))
print(sub['SIBSP'].value_counts(sort=False,normalize=True))

# Making variable categorical 
sub['SIBSP'] = sub['SIBSP'].astype('category')

# Visualising counts of siblings and spouse number with bar graph
seaborn.countplot(x="SIBSP", data=sub);
plt.xlabel('No. of Siblings and Spouse')
plt.ylabel('Frequency')
plt.title('Count of  number of Siblings and Spouse')

# Showing proportion of survival for different number of siblings and spouse
seaborn.factorplot(x="SIBSP", y="SURVIVED", data=sub, kind="bar", ci=None)
plt.xlabel('No. of Spouse and Siblings')
plt.ylabel('Survive Percentage')
plt.title('Survive v/s No. of Siblings and Spouse')


# 68% passengers travelled alone without spouse and siblings. 55% of passenger with one spouse or sibling survived. I think male sacrificed for their spouse as female survival rate is high and here 55%s passengers with one spouse or sibling survived. This also happened in famous movie 'Titanic'.

# In[ ]:


# Now visualization of 'PARCH'
# Printing counts and percentage of number of children and parent
print(sub['PARCH'].value_counts(sort=False))
print(sub['PARCH'].value_counts(sort=False,normalize=True))

# Making variable categorical 
sub['PARCH'] = sub['PARCH'].astype('category')

# Visualising counts of children and parent number with bar graph
seaborn.countplot(x="PARCH", data=sub);
plt.xlabel('No. of Children and Parent')
plt.ylabel('Frequency')
plt.title('Count of  number of Children and Parent')

# Showing proportion of survival for different number of parent and children
seaborn.factorplot(x="PARCH", y="SURVIVED", data=sub, kind="bar", ci=None)
plt.xlabel('No. of Children and Parent')
plt.ylabel('Survive Percentage')
plt.title('Survive v/s No. of Children and Parent')


# 78% passengers travelled alone without parent and children.  Parents sacrificed for their children that's why more than 50% passengers having PARCH = 1,2,3 survived.

# In[ ]:


# Now visualization of 'Gender'
# Printing counts and percentage of male and female
print(sub['SEX'].value_counts(sort=False))
print(sub['SEX'].value_counts(sort=False,normalize=True))

# Making variable  categorical 
sub['SEX'] = sub['SEX'].astype('category')

# Visualising counts of Gender with bar graph
seaborn.countplot(x="SEX", data=sub);
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Count of Gender')

# Showing proportion of survival for different type of gender
seaborn.factorplot(x="SEX", y="SURVIVED", data=sub, kind="bar", ci=None)
plt.xlabel('Gender')
plt.ylabel('Survive Percentage')
plt.title('Survive v/s Sex')


# 65% percent of passenger were males of which only 20% survived.  75% of females survived. Output is consistent with my assumption which was made earlier.

# In[ ]:



# Now visualization of 'PCLASS'
# Printing counts and percentage of diffrent passanger class
print(sub['PCLASS'].value_counts(sort=False))
print(sub['PCLASS'].value_counts(sort=False,normalize=True))

# Making variable  categorical 
sub['PCLASS'] = sub['PCLASS'].astype('category')

# Visualising counts of diffrent passanger class with bar graph
seaborn.countplot(x="PCLASS", data=sub);
plt.xlabel('Passanger Class')
plt.ylabel('Frequency')
plt.title('Count of different passenger class')

# Showing proportion of survival for different types of passanger class
seaborn.factorplot(x="PCLASS", y="SURVIVED", data=sub, kind="bar", ci=None)
plt.xlabel('Passanger Class')
plt.ylabel('Survive Percentage')
plt.title('Survive v/s Passanger Class')


# 55% passenger travelled in third class as it is cheaper of which only 25% survived . 62% of  first class passenger survived. Output is consistent with my assumption as discussed in  variable selection section.

# In[ ]:


# Now visualization of 'EMBARKED'
# Printing counts and percentage of diffrent points of embarkation
print(sub['EMBARKED'].value_counts(sort=False))
print(sub['EMBARKED'].value_counts(sort=False,normalize=True))

# Making variable  categorical 
sub['EMBARKED'] = sub['EMBARKED'].astype('category')

# Visualising counts of diffrent points of embarkation with bar graph
seaborn.countplot(x="EMBARKED", data=sub);
plt.xlabel('Embarkation Point')
plt.ylabel('Frequency')
plt.title('Count of different embarkation point')

# Showing proportion of survival for different points of embarkation
seaborn.factorplot(x="EMBARKED", y="SURVIVED", data=sub, kind="bar", ci=None)
plt.xlabel('Embrakation Point')
plt.ylabel('Survive Percentage')
plt.title('Survive v/s Embarkation Points')


# 72% of people boarded at Southampton as it was starting point of which only 35% survived . 55% of passengers who boarded at Cherbourg survived.

# ## Modelling  ##
# ### 3.1. Random Forest
# Random forest classifier with max_depth(Depth of Tree) = 10, min_samples_split=2 and  n_estimators(No. of Tree) = 100 is implemented.
# 
# ### 3.2. Data Prepration

# In[ ]:


# Convert the male and female groups to integer form
sub['SEX'] = sub['SEX'].astype('category')
sub['SEX'] = sub['SEX'].cat.rename_categories([0,1])

# Convert the Embarked classes to integer form
sub['EMBARKED'] = sub['EMBARKED'].astype('category')
sub['EMBARKED'] = sub['EMBARKED'].cat.rename_categories([0,1,2])

# Replacing unknown age by median of 'AGE' variable of test data
test_data['AGE'] = test_data['AGE'].replace(numpy.nan, test_data['AGE'].median())

# In test dataset EMBARKED variable has some missing data. So filling it with category 'S' as this has most frequency
test_data['EMBARKED'] = test_data['EMBARKED'].fillna('S')

# Convert the male and female groups to integer form
test_data['SEX'] = test_data['SEX'].astype('category')
test_data['SEX'] = test_data['SEX'].cat.rename_categories([0,1])

# Convert the Embarked classes to integer form
test_data['EMBARKED'] = test_data['EMBARKED'].astype('category')
test_data['EMBARKED'] = test_data['EMBARKED'].cat.rename_categories([0,1,2])

# Create the target and features numpy arrays: target, features_one
target = sub['SURVIVED'].values
features = sub[['PCLASS', 'AGE', 'SIBSP', 'PARCH', 'EMBARKED', 'SEX']].values


# ### 3.3. Implementation

# In[ ]:


# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features,target)

# Print the score of the fitted random forest
print(my_forest.score(features, target))

# Compute predictions on our test set features then print the length of the prediction vector 
pred_forest = my_forest.predict(features)


# ### 3.4.  Predicting Output and saving it to pred.csv for submission

# In[ ]:


# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =numpy.array(test_data["PASSENGERID"]).astype(int)
pred_forest1 = pandas.DataFrame(pred_forest, PassengerId, columns = ["Survived"])

# Check that your data frame has 418 entries
print(pred_forest1.shape)

#Write your solution to a csv file with the name my_solution.csv
pred_forest1.to_csv("pred.csv", index_label = ["PassengerId"])


# ## 4. Conclusion
# First data was analyse for suitable variable and data visualisation was done and finally random forest algorithm was implemented to predict survival of passenger\.   
# I am beginner in data science and python. Your feedback and suggestions are welcome. 
