#!/usr/bin/env python
# coding: utf-8

# This is my exploration through the Titanic survival data. This is my first Kraggle Kernal and Competition as well. I am fairly new to Machine learning and Data Science, so my Main goal is to better my process and understanding.  
# 
# Lots of inspiration of this Kernal came from the following Kernals.
# 
# https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic
#  
# https://www.kaggle.com/oysteijo/titanic/titanic-passenger-survival

# **Importing Statements and Loading the Data**
# 
# Nothing to special at this moment, just loading in the Data into a Dataframe, which should help exploring the data a little easier. 

# In[ ]:


#import Statements:
import numpy as np
import random
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# loading the train & test csv files as a DataFrame
train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )


# To Start out, I just want to explore the data and get it to a point where I can run it to get a baseline. Not really exploring the features to much. Feature creation and exploration I have planned for later. 

# In[ ]:


# Looking at info of the data sets
train_df.info()
print("----------------------------")
test_df.info()


# From the Information above, it seems to get the data to a "Usable" state 
# we will have to get rid or fill in all the null values.  Then we will have to convert all Text based set into numerical data. 
# 
# To start lets first work on fill in the null values.  I like to start with the easiest things first. It looks like Embarked on has two missing values in the training set, so  I will start there. 

# In[ ]:


#Embarked
sns.countplot(x="Embarked", data=train_df)


# With the majority of people starting their journey from "S". I think it is safe to say we can fill in the the two Null values with "S" as well.  

# In[ ]:


train_df['Embarked'] = train_df['Embarked'].fillna(value='S')
train_df.info()
print("----------------------------")
test_df.info()


# Next step, Cabin. Cabin seems to have some many null values both in the training and testing test, I think the best plan of action it to just drop it from the data sets. 

# In[ ]:


#Cabin
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)


# Next  is Fare,  for Fare there is one missing value in the testing set, so lets add that in. 

# In[ ]:


#Fare
fare_Plot = test_df['Fare'].dropna(axis = 0)
fare_Plot=fare_Plot.astype(int)
sns.distplot(fare_Plot , bins= 20)
median_Fare = test_df['Fare'].median()
print (median_Fare)


# It seems looking at the graph and numbers it is okay to fill in the Null value with the median fare. 

# In[ ]:


#Fare
test_df['Fare'] = test_df['Fare'].fillna(value=median_Fare)

train_df.info()
print("----------------------------")
test_df.info()


# To fill in the values of age I will be taking a simple route and adding random ages to the test set between the 1st standard deviation, but before I do this I am wondering if there is at large difference in mean and deviation  between people who survived and people who didn't, so let look into that below. 

# In[ ]:


# Age
#Looking at the data
age_plot = train_df.loc[train_df['Survived'] == 0, 'Age'].dropna(axis = 0)
age_plot_survived = train_df.loc[train_df['Survived'] == 1, 'Age'].dropna(axis = 0)
age_STD = age_plot.std()
age_mean = age_plot.mean()
age_median = age_plot.median()
print (age_STD)
print (age_mean)
print ('-------------------')
age_STD_survived = age_plot_survived.std()
age_mean_survived = age_plot_survived.mean()
age_median_survived = age_plot_survived.median()
print (age_STD_survived)
print (age_mean_survived)


# From above it seem there isn't to much difference so I will just fill in the data not caring if they have survived or not. 

# In[ ]:


#Age
#Filling in the data.
train_null= train_df.loc[train_df['Age'].isnull() == True]
test_null= test_df.loc[test_df['Age'].isnull() == True]
train_index = train_null['Age'].index.tolist()
test_index = test_null['Age'].index.tolist()
min_age_range = age_mean - age_STD
min_age_range=int(min_age_range)
max_age_range = age_mean + age_STD
max_age_range = int(max_age_range)

train_filler =np.random.randint(min_age_range, high=max_age_range, size=len(train_null))
test_filler = np.random.randint(min_age_range, high=max_age_range, size=len(test_null))

train_Replace = pd.Series(train_filler, index=train_index)
train_df['Age']= train_df['Age'].fillna(train_Replace)

test_Replace = pd.Series(test_filler, index=test_index)
test_df['Age']= test_df['Age'].fillna(test_Replace)

train_df.info()
print("----------------------------")
test_df.info()


# Almost there! To make the model runn-able. I will now covert all the categorical data into process-able data for the model. 

# In[ ]:




from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(train_df['Sex'])
print(list(le.classes_))
train_df['Sex']=le.transform(train_df['Sex'])

le.fit(test_df['Sex'])
test_df['Sex']=le.transform(test_df['Sex'])


# Improvement #1
# From just looking at how well the the classifiers predicts the training data, my guess is that it is over-fitting a bit to the training data. I am going to implement cross validation so I can help keep an eye out for this.  
# 
# Also at the same time  I will dive into each feature to see what feature is valid.  

# In[ ]:


#Passenger
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
sns.distplot(train_df.loc[train_df['Survived'] == 0, 'PassengerId'] , bins= 20, ax=ax1)
sns.distplot(train_df.loc[train_df['Survived'] == 1, 'PassengerId'] , bins= 20, ax=ax2)


# In[ ]:


#Embarked
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
sns.countplot(x="Embarked", hue='Survived', data=train_df, ax = ax1)
sns.countplot(x="Embarked", hue='Sex', data=train_df, ax = ax2)


# From the above data and Graphs, I believe Passenger ID and Embarked, don't provide useful information to learn off of. So I will drop them.

# In[ ]:


#Ticket
plot_df = train_df
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(plot_df['Ticket'])
plot_df['Ticket'] = le.transform(plot_df['Ticket'])
    
count_plot = train_df['Ticket'].value_counts().head(20)
count_plot_index=count_plot.index.tolist()

ticket_count=[]
survived_count=[]
WC_count=[]
for x in range(0, len(count_plot_index)):
    new_ticket_count = 0
    new_survived_count = 0
    new_WC_count = 0
    for y in range(0,len(train_df['Ticket'])):
        if train_df['Ticket'][y]== count_plot_index[x]:
            new_ticket_count =new_ticket_count+1
            if (train_df['Age'][y]<16) or (train_df['Sex'][y]==0):
                new_WC_count = new_WC_count+ 1
            if train_df['Survived'][y]== 1:
                new_survived_count = new_survived_count+1
                
    ticket_count.append(new_ticket_count)
    survived_count.append(new_survived_count)
    WC_count.append(new_WC_count)
ag_count_plot = pd.DataFrame({'Ticket_Number': count_plot_index, 
                              'Ticket_Count': ticket_count,
                              'Survived_Count':survived_count,
                              'WC_Count': WC_count})
#To Do add Class

g =sns.barplot(x="Ticket_Number", y='Ticket_Count', data=ag_count_plot, color = "red")
topbar =sns.barplot(x="Ticket_Number", y='WC_Count', data=ag_count_plot, color = 'yellow', )
bottombar =sns.barplot(x="Ticket_Number", y='Survived_Count', data=ag_count_plot, linewidth=2.5, facecolor=(1, 1, 1, 0))

print(ag_count_plot)


# For ticket, my thought process was to see if carried importance to the learning process. Maybe exposing an  unknown trend or did it just reflect other factors in features we are also training on. Based on the above I believe it doesn't just reflect other factors, so we will keep it in the training set.

# In[ ]:



#storing PassengerId for Submission:
Test_PId= test_df['PassengerId']

#Droping PassengerId
train_df=train_df.drop(['PassengerId'], axis=1)
test_df=test_df.drop(['PassengerId'], axis=1)
#Droping Embarked
train_df=train_df.drop(['Embarked'], axis=1)
test_df=test_df.drop(['Embarked'], axis=1)


train_df.info()
print("----------------------------")
test_df.info()


# For  improvement # 2,
# 
# I think I am going to try and convert name in to something more descriptive. From reading other blogs it seem that Name can be quite important, but currently my seems to have a pretty low weight

# In[ ]:


TitleTrain=[]
TitleTest=[]
trainTitle_index =  train_df['Name'].index.tolist()
testTitle_index =  test_df['Name'].index.tolist()

for X in train_df['Name']:
    NameTitle = X.partition(', ')[-1].rpartition('.')[0] 
    TitleTrain.append(NameTitle)
    
for X in test_df['Name']:
    NameTitle = X.partition(', ')[-1].rpartition('.')[0] 
    TitleTest.append(NameTitle)

trainTitle_Replace = pd.Series(TitleTrain, index=trainTitle_index)
train_df['Name']= trainTitle_Replace

testTitle_Replace = pd.Series(TitleTest, index=testTitle_index)
test_df['Name']= testTitle_Replace

#Changing MRS and MISS to one category:
train_df.loc[train_df['Name'] == 'Mrs', 'Name'] = 'Miss'
test_df.loc[test_df['Name'] == 'Mrs', 'Name'] = 'Miss'

NameListIndex = train_df['Name'].value_counts().index.tolist()

NameList = train_df['Name'].value_counts().tolist()
for x in range(0,len(NameListIndex)):
    if NameList[x] <10:
        train_df.loc[train_df['Name'] == NameListIndex[x], 'Name'] = 'Misc'
    else:
        train_df.loc[train_df['Name'] == NameListIndex[x], 'Name'] = NameListIndex[x]

NameTestListIndex = test_df['Name'].value_counts().index.tolist()
NameTestList = test_df['Name'].value_counts().tolist()
for x in range(0,len(NameTestListIndex)):
    if NameTestList[x] <10:
        test_df.loc[test_df['Name'] == NameTestListIndex[x], 'Name'] = 'Misc'
    else:
        test_df.loc[test_df['Name'] == NameTestListIndex[x], 'Name'] = NameTestListIndex[x]

sns.countplot(x="Name", hue="Survived", data=train_df)
print(train_df['Name'].value_counts())


# In[ ]:



le.fit(train_df['Name'])
train_df['Name'] = le.transform(train_df['Name'])
le.fit(test_df['Name'])
test_df['Name'] = le.transform(test_df['Name'])

le.fit(test_df['Ticket'])
test_df['Ticket'] = le.transform(test_df['Ticket'])


# In[ ]:



#Now to split the data into a form that can be run in a model 
X_train = train_df.drop(['Survived'], axis=1)
Y_train = train_df['Survived']
X_Pred = test_df
print(X_train.columns.values)


# Run it through a couple different classifiers

# In[ ]:


from sklearn import  grid_search 
parameters = {'n_estimators':[100, 150, 200]}
random_forest = RandomForestClassifier()
RF_clf = grid_search.GridSearchCV(random_forest, parameters)
RF_clf.fit(X_train, Y_train)

#Cross Validation Output
from sklearn.cross_validation import KFold, cross_val_score
k_fold = KFold(len(Y_train), n_folds=10, shuffle=True, random_state=0)
CV_AVG = cross_val_score(random_forest, X_train, Y_train, cv=k_fold, n_jobs=1)
print (sum(CV_AVG) / float(len(CV_AVG)))

# Submission Output
#Y_Pred = random_forest.predict(X_Pred)
#print(Y_Pred)


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

parameters = {'n_estimators':[100, 150, 200]}
Ada = AdaBoostClassifier()
Ada_clf = grid_search.GridSearchCV(Ada, parameters)
Ada_clf.fit(X_train, Y_train)

#Cross Validation Output
k_fold = KFold(len(Y_train), n_folds=10, shuffle=True, random_state=0)
CV_AVG = cross_val_score(Ada_clf, X_train, Y_train, cv=k_fold, n_jobs=1)
print (sum(CV_AVG) / float(len(CV_AVG)))

# Submission Output
Y_Pred = Ada_clf.predict(X_Pred)
print(Y_Pred)


# Create the submission data

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": Test_PId,
        "Survived": Y_Pred
    })
submission.to_csv('titanic.csv', index=False)

