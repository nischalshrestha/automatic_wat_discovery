#!/usr/bin/env python
# coding: utf-8

# We want to discover which passengers survived through the data.
# 
# This notebook contains:
# * Data analysis
# * Feature Engineer at:
#    * Gender, Embarked type, Name, Age ,Fare and Alone
# * Modeling with:
#    * LogisticRegression,KNeighborsClassifier, , DecisionTreeClassifier, RandomForestClassifier
#    * Score and Cross-Validation

# In[ ]:


import pandas as pd
import numpy as np 
from pandas import Series,DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')


# In[ ]:


titanic_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# making a list of both files

combined_df = [titanic_df,test_df]
combined_Data = pd.concat(combined_df,axis=0,sort=True)
titanic_df.head()


# In[ ]:


print('Traning Data -->',titanic_df.shape)
print('-'*58)
print('Testing Data  -->',test_df.shape)


# In[ ]:


titanic_df.info()


# Later,we have to deal with Null values in Age & Cabin 

# In[ ]:


titanic_df.describe()


# **Conclusion**
# 
# 1.Minumum value for the age is 0.4 . Look's like there are newborns.Keep that in mind when working with ages.
# 
# 2.There is huge variation in the fare column 

# Describe on categorical feature 
# 
# As you can see there are few columns missing from the above table

# In[ ]:


titanic_df.describe(include=['O'])


# This really comes in handly to see check the duplicacy 
# 
# 1.There are more Male on the ship.
# 
# 2.Most of people embarked from the S port and there are 3 unique port.

# ## Who are the passenger ?

# Let's differenciate them on gender basis

# In[ ]:


titanic_df['Sex'].value_counts()


# In[ ]:


plt.figure(figsize=(8,7))
sns.countplot(data=titanic_df,x='Sex')


# Let's see who survived

# In[ ]:


plt.figure(figsize=(9,7))
plt.style.use('seaborn')
sns.countplot(data=titanic_df,x='Sex',hue='Survived')


# So most of most the women survived.
# 
# Let's check out the average count of men & women who survived.

# In[ ]:


titanic_df['Survived'].groupby(titanic_df['Sex']).mean() *100


# So over 74.2% of women were saved but only 18.89% of men survived. 
# 
# It woulbe de intresting to see the number of women & men survived,

# In[ ]:


survived = titanic_df['Survived'].groupby(titanic_df['Sex']).value_counts().unstack()


# In[ ]:


survived


# ## Working with Pclass 

# In[ ]:


# check for NULL values

sum(titanic_df['Pclass'].isnull())


# In[ ]:


titanic_df['Pclass'].value_counts()


# In[ ]:


plt.figure(figsize=(8,6))
plt.style.use('seaborn')
sns.countplot(data=titanic_df,x='Pclass')


# In[ ]:


# Standard excel pivot style
titanic_df['Sex'].groupby(titanic_df['Pclass']).value_counts().unstack()


# In[ ]:


# let's Visualize  
plt.style.use('seaborn')
plt.figure(figsize=(8,7))
sns.countplot(data=titanic_df,x='Pclass',hue='Sex')


# So most of the passenger were in Pclass 3. 
# 
# If you look at the ratio of male & female in each class.
# 
# It will be intresting to see the percentage of male and female in the Pclass 3 

# ** Percentage of Male in Pclass**

# In[ ]:


(titanic_df['Sex'].groupby(titanic_df['Pclass']).value_counts().unstack()['male']/titanic_df['Pclass'].value_counts()[3]) *100


# Wow 70% of Male are in Pclas 3. Let's check out the Female percentage 

# In[ ]:


(titanic_df['Sex'].groupby(titanic_df['Pclass']).value_counts().unstack()['female']/titanic_df['Pclass'].value_counts()[3]) *100


# ## Working with Age

# In[ ]:


titanic_df['Age'].describe()


# **Remember there are few NULL values in age column**

# In[ ]:


# number of NULL values in age 

sum(titanic_df['Age'].isnull()) 


# In[ ]:


# number of Null values in test file

sum(test_df['Age'].isnull())


# There are Null values in both the file.Later we will figure out how to fill these Null values

# The minimum value for the age is 4 Months.It will be intresting to see how many newborn were there on the ship.
# 
# Also let's a change in the Sex column and <= 16 year as a child

# In[ ]:


# new borns

titanic_df[titanic_df['Age']<1]


# In[ ]:


len(titanic_df[titanic_df['Age']<=16])


# In[ ]:


# Distribution of age in each Pclass 

plt.figure(figsize=(100,80))
g = sns.FacetGrid(data=titanic_df,hue='Pclass',aspect=4)
g.map(sns.kdeplot,'Age',shade=True)

plt.style.use('seaborn')
g.add_legend()


# In the above KDE plot it's very hard to understand the distribution of ages in each class plus there are NULL values for the 
# age column as well.
# 
# It would be great to fill up those null values and divide the age into different age groups and then visualize it to get a better picture.
# 

# In[ ]:


# Let's check out the head of dataframe and see to which columns is more relateable to the those the age

titanic_df.head()


# We can fill the NULL values of Age column with the help of **Name** column as it also have the salutation along with the name 
# of passenger.
# 
# For this to work we need to do few things
# 
# 1.Check out the data in **Name** column.
# 
# 2.Look for duplicasy and NULL values.
# 
# 3.Create a new column and separate the salutation from the actual name.This often known as creating **new features** in data analysis.

# ## Creating New features

# In[ ]:


sum(titanic_df['Name'].isnull())


# In[ ]:


for df in combined_df:
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.',expand=True)


# In[ ]:


# check if every thing works fine
titanic_df['Title'].head()


# In[ ]:


titanic_df['Title'].value_counts()


# In[ ]:


plt.figure(figsize=(16,8))
plt.style.use('seaborn')
sns.countplot(data=titanic_df,x='Title')


# Let's assign a number for the title just like the Sex column

# In[ ]:


title_map = {'Mr':1,
             'Miss':2,
             'Mrs' :3,
             'Master':4,
             "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 4,"Countess": 4, "Ms": 4, 
            "Lady": 4, "Jonkheer": 4,"Don": 4, "Dona" : 4, "Mme": 4,"Capt": 4,"Sir": 4
             
            }


# In[ ]:


for df in combined_df:
    df['Title'] = df['Title'].map(title_map)


# Now check the mean age in each group and with the help of these fill Null vaulue in age column

# In[ ]:


titanic_df[['Title','Age']].groupby('Title').mean()


# Creating a fucntion

# In[ ]:


def compute_age(dataframe):
    Age = dataframe['Age']
    Title = dataframe['Title']
    
    if pd.isnull(Age):
        if Title == 1:
            return titanic_df['Age'][titanic_df['Title']==1].mean()
        if Title == 2:
            return titanic_df['Age'][titanic_df['Title']==2].mean()
        if Title == 3:
            return titanic_df['Age'][titanic_df['Title']==3].mean()
        if Title == 4:
            return titanic_df['Age'][titanic_df['Title']==4].mean()
        
    else:
        return Age
    
    


# In[ ]:


for df in combined_df:
    df['Age'] = df[['Age','Title']].apply(compute_age,axis=1)


# In[ ]:


# now check for null in age column

sum(titanic_df['Age'].isnull()),sum(test_df['Age'].isnull())


# In[ ]:


g=sns.FacetGrid(data=titanic_df,aspect=5)
plt.style.use('seaborn')
g.map(sns.kdeplot,'Age',shade=True)


# Now divide the age into group to get more clear picture and then we will visulize it.
# 
# There are couple of ways to do this but the way i like you do is make a function to create a new column and we use that column later for visualization. 

# In[ ]:


def age_group(age):
    if age <=16:
        return 1
    if (age > 16) & (age <=26):
        return 2
    if (age > 26) & (age <=36):
        return 3 
    if (age > 36) & (age <=46):
        return 4
    if (age > 46) & (age <=56):
        return 5
    if age > 56:
        return 6
      


# In[ ]:


for df in combined_df:
    df['Age_group'] = df['Age'].apply(age_group)


# In[ ]:


titanic_df['Age_group'].value_counts()


# In[ ]:


plt.figure(figsize=(10,8))
plt.style.use('seaborn')
sns.countplot(data=titanic_df,x='Age_group',hue='Survived')


# To make stacked countplot

# In[ ]:


survived_num = titanic_df[titanic_df['Survived']==1]['Age_group'].value_counts()
dead_num = titanic_df[titanic_df['Survived']==0]['Age_group'].value_counts()

survived_num,dead_num


# In[ ]:


plt.style.use('seaborn')
stacked_df = pd.DataFrame([survived_num,dead_num])

stacked_df.index = ['Survived','Dead']
stacked_df.columns = ['0-16','17-26','27-36','37-46','47-56','57+']

stacked_df.plot(kind='bar',stacked=True,figsize=(10,6))

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=1)


# In[ ]:


titanic_df.head()


# In[ ]:


titanic_df.isna().sum()


# In[ ]:


sns.heatmap(titanic_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


fig,ax = plt.subplots(1,2)

plt.title('Null values in Traning and Test Data')

sns.heatmap(titanic_df.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[0])
sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[1])


# In[ ]:


print('Traning Data -->',titanic_df.shape)
print('-'*58)
print('Testing Data  -->',test_df.shape)


# ## Working with Fare

# In[ ]:


# to check the Null Values and the describe Fare in each traning and tests dataset

for df in combined_df:
    print(sum(df['Fare'].isna()))
    print('--'*25)
    print(df['Fare'].describe())


# So, we have one Null Value in Test dataset. 
# 
# Since Fare is directly related to the Class to we can repalce the null values from the mean values of Classs

# In[ ]:


test_df[test_df['Fare'].isna()]


# In[ ]:


# Calculate the mean and replace this v
# Combined_Data is just the concacated datatframe of the traninng and test 

test_df['Fare']  = test_df['Fare'].fillna(combined_Data['Fare'][combined_Data['Pclass'] == 3].mean())


# In[ ]:


test_df[test_df['Fare'].isna()]  # this should be empty


# In[ ]:


sum(titanic_df['Fare'].isnull())   # Null Vlaues in Fare 


# In[ ]:


# test_df.loc[test_df['Fare']<17,'Fare']

# use this if you want to change the fare in its place
# we i want to keep the oroginal price column and create a new Fare_group column
# just like the i did above with Age

# for df in combined_df: 
#     df['Fare_Class'] = df.loc[df['Fare'] <10 ,'Fare'] == 1
#     df['Fare_Class'] = df.loc[ (df['Fare'] >= 10)  & (df['Fare'] < 30) ,'Fare'] == 2
#     df['Fare_Class'] = df.loc[ (df['Fare'] >= 30)  & (df['Fare'] < 70) ,'Fare'] == 3
#     df['Fare_Class'] = df.loc[ (df['Fare'] >= 70)  ,'Fare'] == 4



# In[ ]:


def fare_class(Fare):
    if Fare < 10 :
        return 1
    if (Fare >= 10)& (Fare < 30):
        return 2
    if (Fare >= 30) & (Fare < 70):
        return 3
    if (Fare >= 70):
        return 4


# In[ ]:


for df in combined_df:
    df['Fare_group'] = df['Fare'].apply(fare_class)


# ## Working with Cabin 

# In[ ]:


# Checking null values in

for df in combined_df:
    print('Number of null Values are ---> ' + str(sum(df['Cabin'].isna())))
    print('Sample data\n')
    print(df['Cabin'].tail())
    print('--'*25)
    


# In[ ]:


# Sepaarte the Cabin names to Cabin_name column

for df in combined_df:
    df['Cabin_Name'] = df['Cabin'].str.extract(r'([A-Za-z])')


# In[ ]:



for df in combined_df:
    print(df['Cabin_Name'].tail())
    print('--'*25)


# In[ ]:




fig,ax = plt.subplots(nrows=1,ncols=2)

sns.countplot(titanic_df['Cabin_Name'].sort_values(ascending=True),ax=ax[0])
sns.countplot(test_df['Cabin_Name'].sort_values(ascending=True),ax=ax[1])



# In[ ]:


cabin_dict = { 'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8}


# In[ ]:


for df in combined_df:
    df['Cabin_Number'] = df['Cabin_Name'].map(cabin_dict)


# In[ ]:


fig,ax = plt.subplots(nrows=1,ncols=2)

sns.countplot(titanic_df['Cabin_Number'].sort_values(ascending=True),ax=ax[0])
sns.countplot(test_df['Cabin_Number'].sort_values(ascending=True),ax=ax[1])



# Now Replace all Null values from Cabin to 0 which means missing Cabin info.
# 
# Replacing Nulll values with a number will help while traning our model.(Let's see)

# In[ ]:


titanic_df['Cabin_Name'].unique()


# In[ ]:


for df in combined_df:
    df.loc[df['Cabin_Number'].isna(),'Cabin_Number'] = 0
    df.loc[df['Cabin_Number'].isna(),'Cabin_Number'] = 0


# In[ ]:


titanic_df.head()


# ## Woking with Embarked

# In[ ]:


gender_dict = {'male' :1,'female':0}


# In[ ]:


titanic_df['Embarked'].unique()


# In[ ]:


titanic_df[titanic_df['Embarked'].isnull()]


# In[ ]:


titanic_df.groupby([titanic_df['Embarked']]).mean()


# In[ ]:


titanic_df['Embarked'] = titanic_df['Embarked'].fillna('C')


# In[ ]:


embarked_dict = {'S':1,'C':2,'Q':3}


# In[ ]:


for df in combined_df:
    df['Embarked'] = df['Embarked'].map(embarked_dict)
    df['Sex'] = df['Sex'].map(gender_dict)


# In[ ]:


titanic_df.head(2)


# ## Working with Alone Feature

# If you check the description for the data,there are two columns(Sibsp,Parch) which defines the passenger family relations.
# 
# I just copy paste the below lines from the description page. 
# 
# 1. **sibsp** column 
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# 2.**parch** column
#  Parent = mother, father
#  Child = daughter, son, stepdaughter, stepson
#  Some children travelled only with a nanny, therefore parch=0 for them.

# In[ ]:


plt.style.use('seaborn')
fig,ax = plt.subplots(nrows=1,ncols=2)


titanic_df['PassengerId'].groupby(titanic_df['SibSp']).count().plot(kind='bar',ax=ax[0])
titanic_df['PassengerId'].groupby(titanic_df['Parch']).count().plot(kind='bar',ax=ax[1])


# In[ ]:


def isAlone(cols):
    Parch = cols[0]
    SibSp = cols[1]
    
    if (Parch == 0) & (SibSp == 0):
        return 1
    else:
        return 0
    


# In[ ]:


titanic_df['Alone'] = titanic_df[['Parch','SibSp']].apply(isAlone,axis=1)


# ## Modeling 
# 
# 

# ## Linear Regression 
# 
# 

# Linear Regression would now work here as we want our predicted variable to be classified into the 2 groups (survived or not)
# 
# So, now let's jump on to the Logistics regression
# 

# 
# ## Logistics Regression

# In[ ]:


X = titanic_df[['Pclass','Age_group','Alone','SibSp','Parch','Fare_group','Age_group','Cabin_Number',
                   'Sex','Embarked']]
y = titanic_df['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train,y_train)


# In[ ]:


log_model_Predict = log_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print('Confusion Matrix \n',confusion_matrix(y_test,log_model_Predict))
print('\n')
print('Classsification Report \n',classification_report(y_test,log_model_Predict))


# ## KNN 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
knn = KNeighborsClassifier(n_neighbors=3)         # deafult value is 5
knn.fit(X_test,y_test)


# In[ ]:


knn_predict = knn.predict(X_test)
print('Confusion Matrix \n',confusion_matrix(y_test,knn_predict))
print('\n')
print('Classsification Report \n',classification_report(y_test,knn_predict))


# After checking the model accuracy on a random value for k (or your leave to the deafualt which is K=5) .Now let's check the Error Rate for the different values of k(number of neighbors). Choose the first value where error rate fall.

# In[ ]:


error_rate = []

for k in range(1,40):
    
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train,y_train)
    knn_predict_k = knn_k.predict(X_test)
    
    error_rate.append(np.mean(y_test != knn_predict_k))


# In[ ]:


plt.style.use('seaborn')
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',ls='dashed',lw=2,marker='o',markerfacecolor='red')


# I will update the notebook with more models and a summarize table of  their cross validation scores.
