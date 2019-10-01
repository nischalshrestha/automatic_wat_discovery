#!/usr/bin/env python
# coding: utf-8

# ## Introduction :
# *Kumar Harshvardhan*
# 
# This work is inspired by many wonderful work already presented.
# 
# ### The complete solution goes through the below stages:
# 
# 1. Problem definition & understanding
# 2. Importing Libraries & Data set
# 3. Understanding & exploring the data
# 4. Data cleaning 
#  	1. Checking for missing values
# 	2. Treating missing values
# 5. Data Wrangling
# 	1. Creating new columns
# 	2. Feature selection
# 6. Plotting & visualization
# 7. Model application & Prediction
# 8. Model evaluation
# 9. Submission
# 
# We may combine some stages and may not perform them in the same sequence for this problem.i.e For treating missing value Deletion will be performed earlier but Imputation will be performed after feature engineering.  

# ### 1. Problem definition & understanding
# From the competition description :
# There were 2224 passengers and crew on Titanic. 
# The Titanic sank after colliding with an iceberg, killing 1502 out of 2224 people
# Some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# ##### We have to do analysis of what were the main factors that contibuted to the survival of some people more than others
# We have to identify which passengers were more likely to survive on the basis of their attributes . So this is a classification problem.<br>
# ([From WIKI](https://en.wikipedia.org/wiki/Statistical_classification))In machine learning and statistics, classification is the problem of identifying to which of a set of categories (sub-populations) a new observation belongs, on the basis of a training set of data containing observations (or instances) whose category membership is known.
# 

# 2. Importing Libraries & loading training and test Data set

# In[45]:


# For data analysis,reading and wrangling
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import random as rnd

#For data visualization
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# For applying machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# In[46]:


# read test data and assign it to test_data pandas dataframe
test_data=pd.read_csv('../input/test.csv')
# read train data and assign it to train_data pandas dataframe
train_data=pd.read_csv('../input/train.csv')
# Print the first 10 rows of the test_data dataframe
train_data.head(10)


# #### Categorical Features
# Categorical: Survived,Sex,Embarked <br>
# Ordinal: Pclass
# #### Numerical Features
# Continous: Age, Fare. <br>
# Discrete: SibSp, Parch

# ### 3. Understanding & exploring the data
# To get better understanding of data we will combine the test data & train data

# In[47]:


full_data=pd.concat([test_data,train_data], axis=0, ignore_index=True)
# To see total number of rows and count and type for each attributes
full_data.info()
# There are TOTAL 1309 records and 12 columns present 
# Age column has 263 missing values and can be in decimal
# Cabin has 77% missing values. So not a good candidate for feature selection.
# Embarked has 2 missing values.
# Fare has 1 missing value.
# PassengerId is a identity column. Not usuful from analytics.
# Survived has 891 values . These values are from test dataset. We have to predict this value for train dataset.


# In[48]:


# to get statistical information about numerical features in the dataframe
full_data.describe()
# Mostly people are less than 39 years.
# Huge variation in Fare . May be due to outliers?
# Mean value for Parent + children is 0.385027 , but max value is 9 .May be due to outliers?
# Mean value for Sibling + Spouse is 0.498854 , but max value is 8. May be due to outliers?
# PassengerId,Survived are not exactly numerical value. These are categorical values so no need to analyze them here.


# In[49]:


# to get information about categorical features in the dataframe
full_data.describe(include=['O'])

#Cabin values have many missing & dupicates value (count=295 - unique=186 =109 duplicate). 
#Embarked takes three possible values. S is the mode for this category (top=S)
#No duplicate Name.count=unique=891
#Sex variable has two possible values. Top=male, freq=843/count=1309 , % male=64.4 , % female=35.6
#Ticket has many dupicates value (count=1309 - unique=929 =380 duplicate). 


# ### Further analyse of data
# 
# We have :
# 1. One missing value for Fare column.
# 2. Two missing value for embarked column.
# 
# Also we need to analyze:
# 1. 380 duplicate values in Ticket column can not be coincidence.
# 2. Too much variation in Fare values.
# 3. Is fare amount is per person or per ticket?
# 
# 
# My assumption is fare amount is per ticket. Which explains duplicate ticket value , too much variation in Fare.
# So if a family of 4 persons are travelling together and total cost of ticket is 400 , they will be issued same ticket number. Also same fare 400 will be mentioned against each person.
# But if a single person is travelling from the same Emabrked and Pclass his/her Fare should be 100.
# 
# To test this assumption:
# 1. For same class and same embarked  , Fare value should increase as the family size increases.
# 2. If Family size(SibSp + Parch + 1 (for self)) >1 , than persons having same ticket number should :<br>
#    a. Belong to same family.<br>
#    b. They should have same embarked , pclass and if cabins is mentioned than same cabins.<br>
#    c. Same Fare mentioned against each of them<br>
#    d. Same Family Size (There can be exception in some cases , like Two sisters & daughter of one travelling together , in this case daughter's family size will be 2 including her , Mother's will be 3 , Another sister's again 2 . But they all can have same ticket number)<br>
#    
#    If Family size=1 & more than one person have same ticket number ,that can be the case where a wealthy  man travelling with     steward, or elderly people travelling with caretaker or freinds travelling together.<br>
#    a. They should have same embarked , pclass and  cabins can be different<br>
#    b. Same Fare mentioned against each of them<br>
#    c. Family size will be usually 1 <br>

# In[50]:


# we will create Fsize for family size  
full_data["Fsize"] = full_data['SibSp'] + full_data['Parch'] + 1


# In[51]:


# this will display the actual variation of Fare with family size for different Embarked & Pclass
sns.lmplot(x='Fsize',y='Fare',data=full_data,col='Pclass',hue='Embarked',palette='coolwarm',aspect=0.6,size=8)
# Fare except for Pclass= 1 , emabrked=Q , for which it is constant , Fare value increases as Fsize increase.


# In[52]:


# To test the assumption that same ticket number people are travelling together , and Fare is same for all the people having same ticket number.
# We have taken all the repeated tickets values in a separate dataframe.
# We have randomly selected a ticket from this dataframe , and displayed all the details of people having same ticket number

Repeatedticket=pd.DataFrame()
Repeatedticket=full_data[full_data.duplicated(['Ticket'], keep=False)]
Repeatedticket.reset_index(drop=True, inplace=False)
Repeatedticket[Repeatedticket['Ticket']==Repeatedticket.iloc[np.random.randint(1, 595)]['Ticket']]

# We can notice that people with same ticket number have same embarked , pclass , same fare , usually same Family size


# ### Data cleaning 
# After data understanding we will work on Train dataset & Test dataset seprately. 
# #### Total records in Train & Test Dataset

# In[53]:


len(train_data)
# There are 891 records in Train data set


# In[54]:


len(test_data)
# There are 418 records in Test data set


# ### Checking for missing values

# In[55]:


train_data.isnull().sum()


# In[56]:


test_data.isnull().sum()


# #### Treating missing values
# #### 1. Deletion

# In[57]:


### If there are any records for which all the values are missing drop it.
train_data.dropna(axis=0,how='all')
test_data.dropna(axis=0,how='all')


# In[58]:


#train_data[train_data['Fare']==0]
# We are droping records which has Fare value as 0
train_data=train_data[train_data['Fare']!=0]
test_data=test_data[test_data['Fare']!=0]
# Not exactly related with analytics , but the people having fare value as 0 are like below , so I don't think their survival or not is any way related with their other attributes.
# Chisholm, Mr. Roderick Robert Crispin :Roderick was one of the nine-strong "guarantee group" of Harland and Wolff employees chosen to oversee the smooth running of the Titanic's maiden voyage.
#Ismay, Mr. Joseph Bruce : served as chairman and managing director of the White Star Line
#his valet Richard Fry and his secretary William Henry Harrison. 
#Cunningham, Mr. Alfred Fleming : Alfred was one of the nine-strong "guarantee group" of Harland and Wolff employees chosen to oversee the smooth running of the Titanic's maiden voyage.


# #### 2. Imputation
# a) Will fill the missing values in Embarked column with the mode (most frequent occured value)<br>
# b) Will fill the missing values in Age column with the mean values 

# In[59]:


train_data['Embarked'].fillna(train_data['Embarked'].value_counts().index[0], inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].value_counts().index[0], inplace=True)
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)


# ### Data Wrangling
# #### 1. Creating new columns
# We will create a new column Family to identify if the traveller was travelling with no family value as 0 ,small family(<4) value as 1 or large family(>3) value as2

# In[60]:


train_data['Famly'] = train_data.apply(lambda x:0  if((x['SibSp'] + x['Parch']==0))  else (1 if((x['SibSp'] + x['Parch']<4))  else 2)  , axis=1)
test_data['Famly'] = test_data.apply(lambda x:0  if((x['SibSp'] + x['Parch']==0))  else (1 if((x['SibSp'] + x['Parch']<4))  else 2)  , axis=1)


# #### 2. Selecting columns

# In[61]:


# Fare (as seen earlier)is dependent on Pclass & Embarked ,So we will keep the independent variable(Pclass & Emabrked) & will drop the dependent variable (Fare)
# will drop 'SibSp'&'Parch' (not important after introducing Famly ) 
# will drop PassengerId,Ticket , Cabin ,Name (not important for analytics)
train_data=train_data.drop(['PassengerId','SibSp','Parch','Fare','Name','Cabin','Ticket'],axis=1)
test_data=test_data.drop(['SibSp','Parch','Fare','Name','Cabin','Ticket'],axis=1)


# In[62]:


# Some models require that none of the columns have missing  value so checking again , that in our selected subset no null value is present
train_data.isnull().any()


# In[63]:


test_data.isnull().any()


# ### 5. Plotting & visualization
# We will plot each attributes against survived to see what factors played the major role in the survival.

# In[64]:


sns.countplot(x='Survived', hue="Pclass",data=train_data)
# Pclass =1 has more chance of survival and Pclass=3 least , for Pclass=2 its almost 50%


# In[65]:


sns.countplot(x='Survived', hue="Sex",data=train_data)
# Female has more chance of survival


# In[66]:


sns.set_context("talk")
fig, axs = plt.subplots(ncols=2)
sns.distplot(train_data['Age'],kde=False,bins=30,ax=axs[0])
g = sns.FacetGrid(train_data, col='Survived')
g = g.map(sns.distplot, "Age",kde=False,bins=30,ax=axs[1])
g = g.map(sns.kdeplot, "Age")

# Except for lower age group 12-15 years(approx), almost all the age group has not survival chance more than survival chance
# Not survvival chance is maximum aroud 30-35 age group (approx)
# Person more than 65 year (approx) has no chance to survive


# In[67]:


g = sns.FacetGrid(train_data, row='Sex', col='Survived', hue='Survived', palette='coolwarm')
g.map(plt.hist, 'Age')
# Age alone did not reveal too much information , so we plot age along with sex to see impact on survival


# In[68]:


sns.countplot(x='Survived', hue="Embarked",data=train_data)
# persons who emabrk at C have more chance to survive


# In[69]:


sns.countplot(x='Survived', hue="Famly",data=train_data)
# Persons having family with size less than 3 , have more chance to survive , than person having no family or having family more than 3 size


# ### Model application & Prediction
# #### Preparing data for models

# In[70]:


# we want to subsitute values in age as range i.e if age lies between 0 to 15 than 0-15 
for index, row in train_data.iterrows():
            if(row['Age']<= 15 ):
                  train_data.set_value(index,'AgeRange','0-15') 
            elif((row['Age']> 15) & (row['Age']<= 30)): 
                  train_data.set_value(index,'AgeRange','15-30')
            elif((row['Age']> 30) & (row['Age']<= 45)): 
                  train_data.set_value(index,'AgeRange','30-45')
            elif((row['Age']> 45) & (row['Age']<= 60)): 
                  train_data.set_value(index,'AgeRange','45-60')
            else:
                  train_data.set_value(index,'AgeRange','>60')


# In[71]:


for index, row in test_data.iterrows():
            if(row['Age']<= 15 ):
                  test_data.set_value(index,'AgeRange','0-15') 
            elif((row['Age']> 15) & (row['Age']<= 30)): 
                  test_data.set_value(index,'AgeRange','15-30')
            elif((row['Age']> 30) & (row['Age']<= 45)): 
                  test_data.set_value(index,'AgeRange','30-45')
            elif((row['Age']> 45) & (row['Age']<= 60)): 
                  test_data.set_value(index,'AgeRange','45-60')
            else:
                  test_data.set_value(index,'AgeRange','>60')


# In[72]:


# we could have done this step above also were we were deleting other not required columns
train_data=train_data.drop(['Age'],axis=1)


# In[73]:


test_data=test_data.drop(['Age'],axis=1)


# In[74]:


# we are creating dummy numerical values for all the categorical values as machine learning models require numerical values and also we want to keep the values replaced in name for easy to identify
train_data=pd.get_dummies(train_data, columns=['AgeRange', 'Pclass','Embarked','Famly','Sex'], prefix=['AgeRange', 'Pclass','Embarked','Famly','Sex'])


# In[75]:


test_data=pd.get_dummies(test_data, columns=['AgeRange', 'Pclass','Embarked','Famly','Sex'], prefix=['AgeRange', 'Pclass','Embarked','Famly','Sex'])


# In[76]:


# define training and testing sets

X_train = train_data.drop("Survived",axis=1)
Y_train = train_data["Survived"]
X_test  = test_data.drop("PassengerId",axis=1).copy()


# In[77]:


# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg_score = round(logreg.score(X_train, Y_train) * 100, 2)

logreg_score


# In[78]:


# Naive Bayes
naive = GaussianNB()

naive.fit(X_train, Y_train)

Y_pred = naive.predict(X_test)

naive_score = round(naive.score(X_train, Y_train) * 100, 2)

naive_score


# In[79]:


# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest_score = round(random_forest.score(X_train, Y_train) * 100, 2)

random_forest_score


# In[80]:


# K neighbours
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

knn_score = round(knn.score(X_train, Y_train) * 100, 2)

knn_score


# In[81]:


# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

decision_tree_score = round(decision_tree.score(X_train, Y_train) * 100, 2)

decision_tree_score


# In[82]:


#  Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(train_data.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df['Coefficient Estimate'] = pd.Series(logreg.coef_[0])
# to see which features contributes most in survival 
coeff_df.sort_values(['Coefficient Estimate'], ascending=[0])
# Famly_1 is having family with size <=3 ,Famly_0 is having NO family,Famly_2 is having family with size >=4


# ### Model evaluation
# Rank the models on the basis of their score to choose the best one for this problem

# In[83]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes','Random Forest','KNN','Decision Tree'],
    'Score': [logreg_score,naive_score,random_forest_score,knn_score,decision_tree_score]})
models.sort_values(by='Score', ascending=False)


# ### Submission

# In[84]:


submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic_result.csv', index=False)


# **Its funny , 3 months ago I did not even know the difference between classification or regression . And now I am submitting my first kernel ( still lot of scope for improvement). <br>
# Thank you everyone for teaching me through your wonderful work .<br>
# 
# *Thank you for reading. Hope you find it useful.***
# 
# 
# 
