#!/usr/bin/env python
# coding: utf-8

# #Titanic Survival Prediction For Beginners
# I am a newbie to machine learning. This project is my first attempt on any dataset (Titanic: Machine Learning from Disaster dataset). I will try my level best to way you through every step. If you find this useful please consider upvoting.

# ##Contents:
# 
# 
# 1.   Import Necessary Libraries
# 2.   Read and Explore the data
# 3.   Data Analysis
# 4.   Data Visualization
# 5.   Cleaning Data
# 6.   Selecting the Model
# 7.   Creating Submission File
# 
# ####I have used google's colab for my program. Any and all feedback is welcome!
# 
# 

# In[ ]:


#this is to access data set stored in google drive to use it uncomment the following lines
#!pip install pydrive
#from google.colab import auth
#auth.authenticate_user()

#from pydrive.drive import GoogleDrive
#from pydrive.auth import GoogleAuth
#from oauth2client.client import GoogleCredentials
#gauth = GoogleAuth()
#gauth.credentials = GoogleCredentials.get_application_default()
#drive = GoogleDrive(gauth)

#myfile = drive.CreateFile({'id': 'your_train_dataset_id'})
#myfile2 = drive.CreateFile({'id': 'your_test_dataset_id'})
#myfile.GetContentFile('train.csv')
#myfile2.GetContentFile('test.csv')


# # **1) Import Necessary Library**
# We will use numpy for numerical calculation, pandas for manupulation of data, matplotlib and seaborn for visualization. %matplotlib inline will make your plot outputs appear and be stored within the notebook

# In[ ]:


# data analysis library
import numpy as np
import pandas as pd

# Visuvalization library
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

#ignore warnings
import warnings 
warnings.filterwarnings('ignore')


# #2) Read And Explore Data
# we will use  pd.read_csv  function to read data from a csv file and store it in a data frame(train).
# View the first 3 entries of the tain using train.head(3) 

# In[ ]:


# import train and test CSV files
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#see the first 3 rows of train
train.head(3)


# In[ ]:


#Look into the training data
train.describe(include='all')


# #3) Data Analysis
# We are going to consider features in our dataset and see how complete they are.

# In[ ]:


# get a list of features
print(train.columns)


# In[ ]:


#see a sample of the dataset to get an idea of variables
train.sample(5)


# -  **Numeical Features:** Age(continuous), Fare(Continuous) SibSp(Discrete), Parch(Discrete)
# -  **Categorical Features:** Survived,Sex,Embarked,Pclass
# -  **Alphanumeric Features:** Tickets,Cabin

# ##what are the data type for each feature?
# -  Survived: int
# -  Pclass: int
# -  Name: string
# -  Sex: string
# -  Age: float
# -  SibSp: int
# -  Parch: int
# -  Ticket: string
# -  Fare: float
# -  Cabin: string
# -  Embarked: string

# In[ ]:


# See a summary of traning data
train.describe(include = 'all')


# ##Observation:
# -  there is a total of 891 passanger in traing dataset.
# -  The age feature is missing approximately 19.8% of its value.
# -  The Cabin feature is missing approx. 77.1% of its value
# -  The Embarked feature is missing 0.22% of its values

# In[ ]:


# checking of NaN values
print(pd.isnull(train).sum())


# ##Some confusing Feature
# -  **sibsp -** number of siblings/spouses aboard the titanic
# - **parch -** number of parents/children aboard the titanic
# - **Embarked -** Port of Embarkation  - - C = Cherbourg, Q = Queenstown, S = Southampton
# - **Pclass -** A proxy of socio-economic status(SES)  --- 1st = Upper, 2nd = Middle, 3rd = Lower 

# #4) Data Visualization
# its time to visualize our data to see which feature affect more on survival rate

# ##Sex Feature

# In[ ]:


#draw a bar plot of survival by sex
sns.barplot(x= 'Sex',y='Survived',data=train)

#print % of female vs. male that survived
print('Percentage of female who survived:',train['Survived'][train['Sex'] == 'female'].value_counts(normalize = True)[1]*100)

print('Percentage of male who survived:',train['Survived'][train['Sex'] == 'male'].value_counts(normalize = True)[1]*100)


# So, Female have a much higher chance of survival than males. The Sex feature is essential in our prediction.

# ##Pclass Feature

# In[ ]:


#draw a bar plot of survival by Pclass
sns.barplot(x='Pclass', y='Survived',data=train)

# Percentage of people that survived by Pclass
#class 1 = Upper
print('petcentage of Pclass=1 who survived:', train['Survived'][train['Pclass'] == 1].value_counts(normalize=True)[1]*100)

#class 2 = Middle
print('petcentage of Pclass=2 who survived:', train['Survived'][train['Pclass'] == 2].value_counts(normalize=True)[1]*100)

#class 3 = Lower
print('petcentage of Pclass=3 who survived:', train['Survived'][train['Pclass'] == 3].value_counts(normalize=True)[1]*100)


# People with higher socio-economic class had a higher rate of survival. Z

# ##SibSp Feature

# In[ ]:


#draw a bar plot for SibSp vs. Survival
sns.barplot(x='SibSp',y='Survived',data=train)

# The percentage of people survived
print("Percentage of SibSp = 0 who suvived:",train['Survived'][train['SibSp']==0].value_counts(normalize=True)[1]*100)

print("Percentage of SibSp = 1 who suvived:",train['Survived'][train['SibSp']==1].value_counts(normalize=True)[1]*100)

print("Percentage of SibSp = 2 who suvived:",train['Survived'][train['SibSp']==2].value_counts(normalize=True)[1]*100)

print("Percentage of SibSp = 3 who suvived:",train['Survived'][train['SibSp']==3].value_counts(normalize=True)[1]*100)

print("Percentage of SibSp = 4 who suvived:",train['Survived'][train['SibSp']==4].value_counts(normalize=True)[1]*100)


# In general, those with more siblings or spouses aboard were less likely to survive. However, people with no sibling or spouses were less likely to survive than those with one or two.

# ##Parch Feature

# In[ ]:


#draw a bar plot for parch vs.survival
sns.barplot(x='Parch',y='Survived',data=train)


# People with less than four parents or children aboard are more likely to survive than those with four or more. Again, people with no parents or children are less likely to survive than those with 1-3 parent or children

# ##Age Feature

# In[ ]:


#sort the ages into logical categories
#train["Age"] = train["Age"].fillna(-0.5)
#test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young adult', 'Adult', 'Elderly']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

# Draw a bar plot for age vs. survival
sns.barplot(x='AgeGroup',y='Survived',data=train)


# Babies are more likely to survive than any other group

# ##Cabin Feature

# In[ ]:


# to convert the alphanum values to integer for ploting
train['CabinBool'] = train['Cabin'].notnull().astype('int')
test['CabinBool'] = test['Cabin'].notnull().astype('int')

sns.barplot(x='CabinBool', y='Survived', data=train )

# Calculate percentage of CabinBool vs. Survived
print('percentage of CabinBool = 1 who survived:',train['Survived'][train['CabinBool']==1].value_counts(normalize=True)[1]*100)

print('percentage of CabinBool = 0 who survived:',train['Survived'][train['CabinBool']==0].value_counts(normalize=True)[1]*100)


# People with a recorded Cabin number are more likely to survive.

# ##Embarked Feature

# In[ ]:


# Draw a bar plot of port of embarkment vs. survival
sns.barplot(x='Embarked', y='Survived', data=train )

# Calculate percentage of CabinBool vs. Survived
print('percentage of Embarked = S who survived:',train['Survived'][train['Embarked']=='S'].value_counts(normalize=True)[1]*100)


print('percentage of Embarked = C who survived:',train['Survived'][train['Embarked']=='C'].value_counts(normalize=True)[1]*100)
      
      
print('percentage of Embarked = Q who survived:',train['Survived'][train['Embarked']=='Q'].value_counts(normalize=True)[1]*100)


# So people who boarded from C = Cherbourg are more likely to survive than people from  Q = Queenstown, S = Southampton.

# #5) Cleaning Data
# Now we will clean our data to account for missing values.

# ##Looking at the test data
# Let's see how our test data look

# In[ ]:


test.describe(include='all')


# 
# 
# *   We have a total of 418 pasengers.
# *   1 value from the Fare feature is missing.
# *    Around 78..2% of Cabin Feature is missing, we need to fill that in.
# 
# 

# ##Ticket Feature

# In[ ]:


# we can drop the Ticket feature since it's unlikely to yeild any useful information
train.drop(['Ticket'], axis = 1, inplace=True)
test.drop(['Ticket'], axis=1, inplace=True)


# ##Embarked Feature

# In[ ]:


#now we need to fill in the missing value in the embarked feature
print("Number of People embarking in Southampton(S):")
southampton = train[train['Embarked']== 'S'].shape[0]
print(southampton)
 
print("Number of People embarking in Cherbourg(C):")
cherbourg = train[train['Embarked']== 'C'].shape[0]
print(cherbourg)
       
print("Number of People embarking in Queenstown(Q):")
queenstown = train[train['Embarked']== 'Q'].shape[0]
print(queenstown)


# So its clear that majority of people embarked from southampton. Let's fill the missing values with S.

# In[ ]:


#replacing the missing values in the embarked feature with s
train = train.fillna({"Embarked":"S"})


# ##Age Feature
# 
# Next we'll fill in the missing values in age feature. Since a higher percemtage of values are missing, it would be illogical to fill all of them with the same value(as we did with embarked). Instead, let's try to find a way to pridict the missing age.

# First we will get the title of passenger from there name. then, we will try to map then into certain age group.

# In[ ]:


# create a combined group of both dataset
combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
  dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
pd.crosstab(train['Title'], train['Sex'])


# In[ ]:


# Replace various titles with more common names
for dataset in combine:
  dataset['Title'] = dataset['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
  dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'],'Royal')
  dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'],'Miss')
  dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()  
  
  


# In[ ]:


# Map each title to a numerical number
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5, 'Rare':6}
for dataset in combine:
  dataset['Title'] = dataset['Title'].map(title_mapping)
  dataset['Title'] = dataset['Title'].fillna(0)

train.head()


# In[ ]:


# group by Sex, Pclass, and Title
#combined = train.append(test,ignore_index=True)
grouped = train.groupby(['Sex','Pclass', 'Title'])
grouped2 = test.groupby(['Sex','Pclass', 'Title'])
# view the median Age by the grouped features 
grouped.Age.median()
grouped2.Age.median()


# Instead of simply filling in the missing Age values with the mean or median age of the dataset, by grouping the data by a passenger’s sex, class, and title, we can drill down a bit deeper and get a closer approximation of what a passenger’s age might have been. Using the grouped.Age variable, we can fill in the missing values for Age.

# In[ ]:


# apply the grouped median value on the Age NaN
train.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
test.Age = grouped2.Age.apply(lambda x: x.fillna(x.median()))


# In[ ]:


train.info()
print('-'*40)
test.info()


# ##Name Feature
# 
# we can drop name feature now as we've extracted the title.

# In[ ]:


#drop the name feature
train.drop(['Name'], axis = 1, inplace=True)
test.drop(['Name'],axis = 1, inplace=True)


# ##Sex Feature

# In[ ]:


# map each sex value to anumerical value
sex_mapping = {"male":0,"female":1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()


# In[ ]:


train.drop('AgeGroup',axis=1,inplace=True)
test.drop('AgeGroup',axis=1,inplace=True)


# In[ ]:


train.info()
test.info()


# ##Embarked Feature

# In[ ]:


# Map each Embarked value to a numerical value
embarked_mapping = {'S':1,'C':2,'Q':3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)


# In[ ]:


#Drop the cabin Feature
train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)


# ##Fare Feature
# We will seaperate the fare value into logical groups as well as filling in single missing value in the dataset.

# In[ ]:


#fill in the missing Fare value in Test set based on mean fare for that Pclass
for x in range(len(test['Fare'])):
  if pd.isnull(test['Fare'][x]):
    pclass = test['Pclass'][x]  #Pclass = 3
    test['Fare'][x] = round(train[train['Pclass'] == pclass]['Fare'].mean(), 4)
    
# maping Fare into Group of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train.drop(['Fare'],axis=1,inplace=True)
test.drop(['Fare'],axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# #6) Choosing the best model
# 
# ## Spliting the Training Data
# We will use part of our traing data(22% in this case) to test accuracy of our different models.
# 

# In[ ]:


from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train['Survived']
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)


# ##Model
# Here i will be using Support Vector Machine(SVM) to predict my output.

# In[ ]:


#SVM model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# #7) Creating Submission File
# it's time to create a submission.csv file to upload to the Kaggle Compettiton!
# 

# In[ ]:


#set ids as PassengeId and predict survival
ids = test['PassengerId']
predictions = svc.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({'PassengerId':ids, 'Survived':predictions})
output.to_csv('submission.csv', index=False)


# In[ ]:


#This give a list of folder and ids in your google drive
file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
for file1 in file_list:
  print('title: %s, id: %s' % (file1['title'], file1['id']))


# In[ ]:


# Copy the id of the folder where you want to save the submission file
file = drive.CreateFile({'parents':[{u'id': 'Your_folder_id_here'}]}) 
file.SetContentFile("submission.csv")
file.Upload()

