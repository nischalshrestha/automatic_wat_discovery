#!/usr/bin/env python
# coding: utf-8

# # Task Description-
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy

# # Solution Steps-
# 
# 1) A CLOSER LOOK AT DATA
# 
#     1.1) Import libraries & Load training data into pandas dataframe<br>
#     1.2) Look at what features we have here<br>
#     1.3) Findings
#     
# 2) FEATURE ENGINEERING
# 
#     2.1) Validate Importance of each feature<br>
#     2.2) Imputing missing values<br>
#     2.3) Converting data into numerical format
#     
# 3) Final Predictions
#     
#     3.1) Check for correlation
#     3.2) Build Model
#     3.3) Accuracy
#  

# # 1) A CLOSER LOOK AT DATA
#     
#    # 1.1) Import libraries & Load training data into pandas dataframe

# In[ ]:


#importing libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


#load the training data
train_data = pd.read_csv("../input/train.csv")
#lets see few rows of our data
train_data.head(6)


# # 1.2) Look at what features we have here

# In[ ]:


#lets what features we have
train_data.info()


# # 1.3) Findings
# 
# We have total 11 features, 1 independent variable  & 891 rows of data.
# 
# PassengerId- Its the the id of passenger travelling<br>
# Survived- Passenger Died or Survived(This is independent variable for whcih we have to train our algorithm)<br>
# Pclass- In which class passenger travelled<br>
# Name- Name of passenger<br>
# Sex- Male or Female<br>
# Age- Age of passenger<br>
# Sibsp- Number of siblings or spouses<br>
# Parch- Number of parents/children<br>
# Ticket- Ticket Number<br>
# Fare- Fare</br> 
# Cabin- Cabin of passenger</br>
# Embarked- Port of embark
# 
# -----------------------------------
# Missing Values-
# 
# Age- 891-714 = 177 values missing from age feature<br>
# Cabin- 891-204 = 687 WOW!!! 687 values missing from cabin feature<br>
# Embarked- 891-889 = 2 only two values missing from feature embarcation

# # 2) Feature Engineering
# # 2.1) Validate Importance of each feature with respect to 'Survived'

# # PassengerID

# In[ ]:


# PassengerID is just a sequence number so we can delete it which do not have any impact on Survival
# Lets delete passengerid
del train_data["PassengerId"]


# # Pclass

# In[ ]:


# Pclass - Its is a numerical catorgircal feature with order, lets plot graph & see its relevance 
sns.factorplot(x="Pclass",y='Survived',data=train_data)


# # Survial chances if passenger is sitting in Class1>>Class2>Class3

# # Name

# In[ ]:


# Name - well name shouldn't affect the survival of the passenger but it can be an important feature 
# what extra information i can get from passenger's name ???
train_data['Name'].head()
#hmm!! we can see below that we can fetch the family Names of passengers it could be a usefull feature to answer other
# questions like "Ethnicity" of the passengers survived. since it is not usufull in our main prediction task 
# i am leaving it for now . 
del train_data['Name']


# # Sex

# In[ ]:


# Sex- it would be interseting to see this feature's relation with 'Survived' feature
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
sns.barplot(x='Sex',y='Survived',data=train_data,ax=ax2)
sns.countplot(train_data["Sex"],ax=ax1)


# # As we can see that 'gender affects survival chances' so this feature will be important

# # Age

# In[ ]:


# age- Lets what how age impacts the chances of survival
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,5))
sns.factorplot(x="Survived",y="Age",data=train_data,ax=ax1)
sns.boxplot(x="Survived",y="Age",data=train_data,ax=ax2)
sns.regplot(x='Age',y='Survived',data=train_data,ax=ax3)
plt.close(2)


# # As we can see from above graphs feature 'Age' is in realtion with 'Survived' so it would be usuful while trianing our algorithm

# # Sibsp(siblings/spouse) & Parch(parents/children) abord

# In[ ]:


# we can add these features to create new feature called "Fam_Size"
train_data['Fam_Size']= train_data['SibSp'] + train_data['Parch']
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,8))
sns.factorplot(x="SibSp",y="Survived",data=train_data,ax=ax1)
sns.factorplot(x="Parch",y="Survived",data=train_data,ax=ax2)
sns.factorplot(x="Fam_Size",y="Survived",data=train_data,ax=ax3)

plt.close(2)
plt.close(3)
 


# # As we can see from above graphs that passengers with family size of (1,2,3) have higher chances of survival so this feature also be useful

# In[ ]:


# since we are goint to work with Fam_Size lets delete Sibsp & Parch
del train_data["SibSp"]
del train_data['Parch']


# # Ticket

# In[ ]:


train_data["Ticket"].head(10)


# # This feature(ticket number) is alpha numbric & random, we could have analye it's pattern find cabin assignment but we already have cabin details so lets skip this feature

# In[ ]:


del train_data["Ticket"]


# # Fare

# In[ ]:


# IT would be interesting to see Fare & Survival realtion-ship
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,5))
sns.boxplot(x='Survived',y='Fare',data=train_data,ax=ax1)
sns.factorplot(x='Survived',y='Fare',data=train_data,ax=ax2)
sns.regplot(x='Fare',y='Survived',data=train_data,ax=ax3)
plt.close(2)


# # Higher the fare & Better chances of survival
# ---
# # Note- Features 'Pclass','Fare','Cabin' might be highly correalted so we will checking for Pearson's coefficient to validate the independance of features
# ---
# 

# # Cabin
# # This feature has 687 missing values & it is alpha-numeric so we will be handling this feature in Missing Values section of notebook

# ---

# # Embarked

# In[ ]:


# This feature has very few (only 2) missing values lets have a visualize this feature
#sns.countplot(train_data['Embarked'])
sns.factorplot(x='Embarked',y='Survived',data=train_data)


# # Survival rate is higher at 'C' we can use this info to find missing values in next section

# In[ ]:


# Lets look at our dataframe now
train_data.head()


# # 2.2) Impute Missing Values

# # Embarked

# In[ ]:


#only 2 missing values lets print rows with missing values
train_data[train_data["Embarked"].isnull()]


# In[ ]:


fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
sns.factorplot(x='Embarked',y='Survived',data=train_data,ax=ax1)
sns.boxplot(x='Embarked',y='Fare',data=train_data,ax=ax2)
plt.close(2)


# # As we can see that passengers with higher fare & high mean survival embarked from 'C' so lets fill our missing values with 'C' because our missing embarked rows have survived-1 & fare- 80

# In[ ]:


train_data['Embarked']=train_data['Embarked'].fillna('C')


# # Cabin

# In[ ]:


#687 missing values , lets fill them with the help of other features
train_data['Cabin'].head(5)


# In[ ]:


# As we can see above that cabin is alphnumeric , here we are not interted in the end digit of cabin lets just filter out
#first letter of cabin that way we will have a categorical variable(easy to analyze).
train_data['Cabin_Id'] = train_data['Cabin'].str[0]
#now we have our required data in cabin_id column we don't need 'Cabin' anymore
del train_data['Cabin']
train_data.head()


# # We will predict the missing values of Cabin_id by applying multiclass classifer algorithms on features 'Fam_Size','Fare','Pclass','Survived' .

# In[ ]:


new_train=train_data[train_data['Cabin_Id'].notnull()]
new_train.head()


# In[ ]:


# depedent variables
X = new_train.iloc[:,[0,1,4,6]].values
y = new_train.iloc[:,7].values


# # SVM

# In[ ]:


from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1)
clf.fit(X,y)
cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
scores = cross_val_score(clf, X, y, cv=4)
scores.mean()                                              


# # LogisticRegression

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X,y)
cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
scores = cross_val_score(clf, X, y, cv=4)
scores.mean()                                              


# # Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=10,random_state=0)
clf.fit(X,y)
cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
scores = cross_val_score(clf, X, y, cv=4)
scores.mean() 


# # KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X,y)
cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
scores = cross_val_score(clf, X, y, cv=4)
scores.mean() 


# # This is the best accurcy i could get for Missing values in "Cabin" Using KNN. Lets predict the values for remaining Nans in 'Cabin'

# In[ ]:


# here new_train will store all the data with Nan In Cabin
df=train_data[train_data['Cabin_Id'].isnull()]
df.head()
k = df.iloc[:,[0,1,4,6]].values
pred = clf.predict(k)


# In[ ]:


#now add these predicted values of Cabin_Id to datafraem
df['Cabin_Id'] = pred
train_data = new_train.append(df)
train_data.head()


# # Handling Missing Values in AGE
# - It is continuous variable so will be using Regression to fill the empty spots

# In[ ]:


# new dataframe- Contains the rows with known age
new_train = train_data[train_data["Age"].notnull()]
new_train.head()
# features
X = new_train.iloc[:,[0,1,4,6]].values
# Age 
y = new_train.iloc[:,3].values


# # Predicting Age Using Lasso

# In[ ]:


from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.20,random_state=0)
reg = linear_model.Lasso (alpha = 0.1)
reg.fit(X_train,y_train)
y_pred=reg.predict(X_val)
# Calculating Mean Square Error
mean_squared_error(y_val, y_pred) 


# # Predicting Age Using Ridge Regression

# In[ ]:


from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.20,random_state=0)
reg = linear_model.Ridge (alpha = 0.1)
reg.fit(X_train,y_train)
y_pred=reg.predict(X_val)
# Calculating Mean Square Error
mean_squared_error(y_val, y_pred) 


# # 165 is the mean squared error of our regression model for AGE lets compare it with mean squared error when we use Mean Instead of Regression predictions

# In[ ]:


#Compaer it with mean
y_mean =np.empty(143)
y_mean.fill(y_pred.sum()/len(y_pred))
mean_squared_error(y_val, y_mean) 


# # MSE if Age is filled by mean= 220
# # MSE if Age is filled by Regression Prediction = 165
# Now lets fill our missing Age with predicted values

# In[ ]:


df=train_data[train_data["Age"].isnull()]
# features
X = df.iloc[:,[0,1,4,6]].values
# Age 
y = reg.predict(X)
df['Age']=y
train_data = new_train.append(df)


# # Lets Look at our training data now- (No Missing Values)

# In[ ]:


train_data.info()


# # 2.3) Converting Categorical Features to Numeric data

# In[ ]:


train_data.head()
#We have Sex, Embarked, Cabin_id as categorical features


# # These features are non ordinal-categorical features , direct assignment of numeric valuse will lead to bad predictions so here we will use "dummies"
# ---
# # Embark

# In[ ]:


Embark_dummy = pd.get_dummies(train_data["Embarked"])
Embark_dummy.head(5)


# # Sex

# In[ ]:


# we do not need to do for featre "SEX" because it is anyways in 2 categoreis but just for understansding lets do it.
Sex_dummy = pd.get_dummies(train_data["Sex"])
Sex_dummy.head(5)


# # Cabin

# In[ ]:


Cabin_dummy = pd.get_dummies(train_data["Cabin_Id"])
Cabin_dummy.head(5)


# # In order to make above values independent we need to delete 1 column from each of them.

# In[ ]:


del Embark_dummy['S']
del Cabin_dummy['T']
del Sex_dummy['female']


# # Now lets add these featers to our train_data dataframe & delete old Embark,Cabin_id,Sex

# In[ ]:


train_data['Sex'] = Sex_dummy['male']
train_data['Embark_C'] = Embark_dummy['C']
train_data['Embark_Q'] = Embark_dummy['Q']
train_data['Cabin_A'] = Cabin_dummy['A']
train_data['Cabin_B'] = Cabin_dummy['B']
train_data['Cabin_C'] = Cabin_dummy['C']
train_data['Cabin_D'] = Cabin_dummy['D']
train_data['Cabin_E'] = Cabin_dummy['E']
train_data['Cabin_F'] = Cabin_dummy['F']
train_data['Cabin_G'] = Cabin_dummy['G']
del train_data['Sex']
del train_data['Embarked']
del train_data['Cabin_Id']


# # Below is our completely processed data.

# In[ ]:


train_data.head(10)


# # 3) Final Predictions
# ---
# # 3.1) check for correlation

# In[ ]:


corr = train_data.corr()
f, ax = plt.subplots(figsize=(25,16))
sns.plt.yticks(fontsize=18)
sns.plt.xticks(fontsize=18)

sns.heatmap(corr, cmap='inferno', linewidths=0.1,vmax=1.0, square=True, annot=True)


# # There is high correlation between 'Pclass' & 'Cabin' features. i will try our algorithms with 'both' features & by 'eliminating one of them' to see if we get any better results
# ---
# # 3.2) Applying Models
# # SVM

# In[ ]:


# features
X = train_data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13]].values
# dependent variable
y = train_data.iloc[:,0].values


# In[ ]:


from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.svm import SVC

clf = SVC(kernel='linear', C=1)
clf.fit(X,y)
cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
scores = cross_val_score(clf, X, y, cv=4)
scores.mean()                                              



# # RandomForest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=15,random_state=0)
clf.fit(X,y)
cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
scores = cross_val_score(clf, X, y, cv=4)
scores.mean() 


# # KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X,y)
cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
scores = cross_val_score(clf, X, y, cv=4)
scores.mean() 


# # 3.3) Accuracy
# ---
# ---
# 
# # KNN- 0.74
# # RandomForest- 81.6
# # Support Vector- 78.1

# ---
# ---
# # more to improve in this kernel :) will come up with updates
