#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# below I was exploring the data, specifically isolating the men and women data. Ignore the below exploratory data anaylsis

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
#set sns as default style
sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
#file paths
test_data_file_path = '../input/test.csv'
train_data_file_path = '../input/train.csv'
#read test data 
test_data = pd.read_csv(test_data_file_path)
train_data = pd.read_csv(train_data_file_path)
#create two plots of the same data to take a look at age and sex

_ = sns.swarmplot(x='Sex', y='Age', data=test_data)
_ = plt.xlabel('Sex')
_ = plt.ylabel('Age')
plt.show()

sex_age = test_data[['Sex', 'Age']].dropna()
female_data = sex_age[sex_age.Sex == 'female']
male_data = sex_age[sex_age.Sex == 'male']

len_sex_age = len(sex_age)
sex_bins = np.sqrt(len_sex_age)
sex_bins = int(sex_bins)

temp = plt.hist(male_data.Age, bins=sex_bins, alpha= .5, label= 'males')
temp = plt.hist(female_data.Age, bins=sex_bins, alpha= .5, label ='females')
temp = plt.xlabel('Age')
temp = plt.ylabel('Frequency')
temp = plt.legend(loc='upper right')
plt.show()


# In order to create a machine learning (ML) model that can successfully predict whether a passenger on the titanic survives we need to go through the data frame and decide which columns are useful.
# 

# In[ ]:


#Let's do some basic exploration

print(train_data.head())
print(train_data.describe())

#with the code above we can see that there are a lot of 'NaN' values in the 'Cabin' column

print(train_data.count())

#upon further inspection we can see that cabin has a alot of missing values,
#we can eliminate that column as well as the ticket number and name columns, 
#then create a copy and name the new dataframe

mod_data = train_data[[ 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 
                       'Parch', 'Fare', 'Embarked']].copy()
#we print some info just to look at the data
mod_data.head()
mod_data.info()




# because our end result is a ML model that can predict whether someone surived, we don't want to over budren the model with unnecessary information, such as name (class and sex should account for anything a name could have provided), ticket number (should be mostly random or if class is associated witht the number, the Pclass will account for it), passangerId (same reason as name), and cabin( too much missing data, although proxamition to the life boats could account for something but this is something that we can maybe revisit later).
# 
# As for some of the columns that i did include there are three that might seem weird at first, embarked, sibsp, and parch. I kep embarked because it would be interesting to see if where they are from affected whether they survived ( perhaps a certain region had the most suriving number )

# In[ ]:


#from the code above we can see that there are a lot of missing
#values in the age column, I decided to use a mean age to fill in the data, although this might
#not be the best way to go (later we can readdress this)
# i want to look at the mean, and median to be sure they are close enough to each other 
print(mod_data.Age.describe())

#after confirming they're relatively close to one another, I get the mean as a variable 
#to redistribute

mean_age = round(mod_data.Age.mean())

#orginally i printed the mean age to be sure it was correct
#print(mean_age)

mod_data['Age'] = mod_data.Age.fillna(mean_age)

# now let's look at the dataframe to make sure everything is there still

print(mod_data.info())


# In[ ]:


#now i want to investigate the rows which contain 
#NaN values in the embarked rows

mod_data[mod_data['Embarked'].isnull()]

#becuase the NaN values represent two surived females and since I want to use the
#Embarked column as part of the ML model
#I have decicded to omit the rows completely (which is .22% of the total data)

mod_data = mod_data.dropna()

print(mod_data.info())


# now that we have the dataframe figured out, we can move into building a model

# In[ ]:


# we need to assign a prediction target for people who survived 
# using convention we label it y

#train_y = mod_data.Survived

# we need to also define the columns, X, since 'Embarked' and 'Sex' are 
# features that we want to use to we need to change their values into numeric 
# first I want to see the values of both Sex and Embarked

#print(mod_data.Sex.unique())


#print(mod_data.Sex)

#mod_data.Sex = 
mod_data = mod_data.replace(['male', 'female'], ['0', '1'])

#print(mod_data.Sex)

#mod_data_features = mod_data['Pclass']
mod_data.info()

mod_data.Embarked = train_data.Embarked

mod_data.Embarked.unique()
print(mod_data)


# so i kinda messed up and my data and I hope I can fix it the way above
# 
# next, we are gonna replace the remaining object column, 'Embarked', into numberic 

# In[ ]:


#i ran into an error earlier using the replace method, 
#however i think i fixed it
#next we'll use the replace again with more float types

mod_data = mod_data.replace(['S', 'C', 'Q'], ['0', '1', '2'])

print(mod_data.head(10))


# In[ ]:


features = ['Pclass', 'Sex', 'Age', 'SibSp', 
                       'Parch', 'Fare', 'Embarked']

# now that the dataframe as been modified we can pick the 
# features we want to help shape the model

mod_data1 = mod_data.sample(n=450)

mod_data2 = mod_data.sample(n=450)

print(mod_data1.head(), mod_data2.head())
train_X = mod_data1[features]
train_y = mod_data1.Survived
Val_X = mod_data2[features]
Val_y = mod_data2.Survived


# In[ ]:


# now we can start importing the ML libraries

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
#print(train_X.head())
print(forest_model.predict(Val_X.head(20)).round())
print(Val_y.head(20))
#print()


#     based on the output, the code seems to predict well
#         now to clean up the data that we are going to use
#         we need to keep in mind that we cannot use anything less than 418 entries

# In[ ]:


test_data.info()


# the largest issue I can find with the data was that it contains 86 missing values for age.
# before going ahead and entering the average age to be 30, like in the practice set i want to be sure the average of the remainding 332 is close
# 

# In[ ]:


test_mod = test_data.copy()
test_mean_age = round(test_mod.Age.mean())
print(test_mean_age)


# based on the data, the average is the same, for a later model I would like to use a seperate model to predict the age, then use the results into the next model

# In[ ]:


test_mod.Age = mod_data.Age.fillna(test_mean_age)

print(test_mod.info())
print(test_mod.Age.unique())
nan_test_age = test_mod[test_mod.Age.isnull()]
nan_test_age.replace('nan', 30)
test_mod.Age = nan_test_age
print(test_mod.info())
test_mod.Age = test_data.Age
print(test_mod.info())
test_mod.Age = mod_data.Age.fillna(test_mean_age)
print(test_mod.info())
print(nan_test_age)
test_mod.loc[61, 'Age'] = test_mean_age
print(test_mod.info())
print(features)
embarked_test = test_mod.Embarked
print(embarked_test.head())
embarked_test = embarked_test.replace(['S', 'C', 'Q'], ['0', '1', '2'])
print(embarked_test.head())
test_mod.Embarked = embarked_test
print(test_mod.info())


# In[ ]:


test_mod.Embarked = pd.to_numeric(test_mod.Embarked)
print(test_mod.info())
test_mod.Fare.describe()


# gonna create a graph to compare the averages 

# In[ ]:


fare_len = test_mod.Fare.dropna()
len_fare = len(fare_len)
print(len_fare)
fare_bins = np.sqrt(len_fare)
print(fare_bins)
fare_bins = int(fare_bins)
print(fare_bins)

fare_aver = plt.boxplot(fare_len, showfliers=False, showmeans=True)
#fare_aver = plt.xlabel('Fare')
#fare_aver = plt.ylabel('Price')

plt.show()


# gonna enter the median average to fill the nan in the fare column

# In[ ]:


nan_test_fare = test_mod[test_mod.Fare.isnull()]
print(nan_test_fare)
fare_median = round(test_mod.Fare.median())
print(fare_median)
nan_test_fare.replace('nan', fare_median)
print(nan_test_fare)
test_mod.loc[152, 'Fare'] = fare_median
print(test_mod.info())


# changing thesex  data to have it match the train data

# In[ ]:


sex_test = test_mod.Sex
print(sex_test)

sex_test = sex_test.replace(['male', 'female'], ['0', '1'])
print(sex_test)
test_mod.Sex = sex_test
print(test_mod)
print(test_mod.info()) 
test_mod.Sex = pd.to_numeric(test_mod.Sex)
print(test_mod.info())


# move to next part, test the model

# In[ ]:


test_X = test_mod[features]
print(test_X)
survived = forest_model.predict(test_X).round()
print(survived)
survived = pd.DataFrame(survived, columns=['Survived'])
print(survived.info())
survived['PassengerId'] = test_mod['PassengerId']
print(survived.info())
columnsTitles = ['PassengerId', 'Survived']
survived = survived.reindex(columns=columnsTitles)
survived = survived.astype(int)
print(survived.head())
print(survived.info())


# 
# next step is to create a csv with with data

# In[ ]:


survived.to_csv('passengers_survived.csv', index=False)


# UGLY NOTEBOOK, will return to fix
