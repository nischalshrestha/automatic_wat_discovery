#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# ## Cleuton Sampaio
# 
# Hi. This is my first attempt in the "Titanic" competition. I do not have much time to work on this assignment, so I'll try to get the closest as possible to a good estimation in the first attempt. 
# Anything greater than 0,80 is good for me. 
# Later on I'll try another algotithms and refine the model.

# ### 1. Getting the data

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
rawdata = pd.read_csv("../input/train.csv")
rawdata.head()


# Label is column "Survived", and we have 11 attribute columns. Some of them may be irrelevant, like the "PassengerId". We have different types of data:
# - Categorical attributes: Embarked (C = Cherbourg, Q = Queenstown, S = Southampton), Sex (male/female), Cabin;
# - Ordinal attributes: Pclass (Passenger's ticket class - 1: Upper, 2: Middle, 3: Lower);
# - Numerical continuous: Age (should be discrete, but there is a fractional part) and Fare;
# - Numerical discrete: SibSp and Parch (I don't know whether they are relevant yet);
# 

# First, let's see how many values we have:

# In[ ]:


rawdata.count()


# There are missing values... Let's see how many survived and how many died:

# In[ ]:


totalSurvivers = rawdata['Survived'][rawdata.Survived == 1].count()
totalDeaths = 891 - totalSurvivers
percentOfSurvivers = totalSurvivers / 891 * 100
print("Total Survivers:", totalSurvivers)
print("Total Deaths:", totalDeaths)
print("% of Survivers:", percentOfSurvivers)


# Let's check the ** Sex ** column impact on deaths:

# In[ ]:


percentOfWomen = (rawdata['Sex'][rawdata.Sex == 'female'].count() / 891) * 100
print("Percent of Women passengers:", percentOfWomen)
percentOfMen = (rawdata['Sex'][rawdata.Sex == 'male'].count() / 891) * 100
print("Percent of Men passengers:", percentOfMen)


# In[ ]:


genderSurvivers = rawdata.groupby(['Sex', 'Survived'])['Survived'].count().reset_index(name="count")
genderSurvivers


# In[ ]:


genderSurvivers.pivot(index='Sex', columns='Survived', values='count').plot(kind='bar')


# Interesting! The proportion of Female survivers is greater than the proportion of Male survivers, so Gender has impact on survival.

# Now, let's study the **Age** factor:

# In[ ]:


ageSurvivers = rawdata[rawdata.Survived == 1][['Age']].dropna().hist()


# Hmmmm. What does it mean? We have basically two large groups of survivors: Children under 10 and adults between 18 and 40. Seems that ** Age ** plays a role here.

# Now, let's check whether the wealth status (** Pclass **) has influence on surviving...

# In[ ]:


plcassSurvivers = rawdata.groupby(['Pclass', 'Survived'])['Survived'].count().reset_index(name="count")
plcassSurvivers


# In[ ]:


plcassSurvivers.pivot(index='Pclass', columns='Survived', values='count').plot(kind='bar')


# This is sad... As we can clearly see, the probability of a passenger survives is related to his 'class'...

# So, what we have until now? Sex, Age and Pclass, as we suspect, have impact on survival. We could finish here, and I would be confident that we've got a good model. But there are more attributes.
# Note that I will not measure correlation for now. I'm just trying to select relevant attributes. 
# We have 2 interesting attributes: # of sibilings (** SibSp **) and # of parent/children (** parch **).

# We have to understand the ** parch ** attribute. A child normally travels with both parents, but there may be exceptions... Children traveling with a nanny or only with one parent. Let's see if we can find something like that...
# 

# In[ ]:


childrenSpec = rawdata[rawdata.Age < 18][rawdata.Parch < 2][['Age','Parch']]
print("Children in special status:", childrenSpec.count())
childrenSpec.head()


# We have to separate adults and children... Let's study children and Survival related to Parch
# 

# In[ ]:


chSpecSurvivers = (rawdata[['Parch','Survived']][(rawdata.Age < 18) & (rawdata.Survived == 1)]).groupby('Parch').count()
chSpecSurvivers


# Well ** Parch ** is related with children survival, but in a different way... Children traveling alone survived less than children traveling with parents.

# In[ ]:


adultsVsSibSp = (rawdata[['Parch','Survived']][(rawdata.Age > 18) & (rawdata.Survived == 1)]).groupby('Parch').count()
adultsVsSibSp


# We can see clearly that adults with more children had less chance to survive, but this may be related to the passenger's class. Let's see the childred distribution per class:

# In[ ]:


adultsVsParch = (rawdata[['Pclass','Parch','Survived','PassengerId']][(rawdata.Age > 18)]).groupby(['Pclass','Parch','Survived']).count()
adultsVsParch


# Now let's find out the same for siblings (Brothers, Spouses etc)...

# In[ ]:


adultsVsSibSp = (rawdata[['SibSp','Survived', 'PassengerId']][(rawdata.Age > 18)]).groupby(['SibSp','Survived']).count()
adultsVsSibSp


# What conclusion we have? Passengers with less siblings survived more and died more, simply because they are the majority. We cannot use SibSp as a good estimator. Maybe we could combine ** Parch ** and ** SibSp ** to get information about the Passenger's party. Who had more chances to survive? An alone Passenger or one in a large party? Let's see...

# In[ ]:


rawdata['Party']=rawdata['SibSp'] + rawdata['Parch']
rawdata[['Party', 'Survived']].groupby(['Party', 'Survived'])['Survived'].count().reset_index(name="count")


# Interesting... Passengers traveling alone or in small parties, had more chance to survive...

# Ok, let's study ** Fare **, ** Cabin **, and ** Embarked **
# 
# ** Fare ** is a continuous value, which may be difficult to grasp. Let's create "Fare categories" to study its relation to Survival. 

# In[ ]:



fareClasses = pd.qcut(rawdata['Fare'],5)
fareClasses.value_counts()


# Let's see how these classes related to survival:
# 

# In[ ]:


rawdata['FareCategory']=pd.qcut(rawdata['Fare'],6)
rawdata[(rawdata.Survived == 1)].groupby(['FareCategory'])['Survived'].count().to_frame().style.background_gradient(cmap='summer_r')


# As the fare category increases, also increases the number of survivors. 

# In[ ]:


cabinDf = (rawdata[['Cabin','Survived']][(rawdata.Survived == 1)]).groupby('Cabin').count()
cabinDf


# As I suspected ** Cabin ** is very difficult to associate because we do not have the ship's Map, and the information may be missing.

# In[ ]:


embarkedDf = (rawdata[['Embarked','Survived']][(rawdata.Survived == 1)]).groupby('Embarked').count()
embarkedDf


# Finally we have something! Passengers who embarked in Southampton tend to survive more than other passengers, so, ** Embarked ** is a good estimator attribute.

# Columns selected until now: 
# - Sex (Gender);
# - Age;
# - Pclass (Passenger's class);
# - Party (Passenger's group size);
# - FareCategory (How much the Passenger paid for the fare, in categories);
# - Embarked;
#           
# Let's study the correlation between these attributes:

# In[ ]:


rawdata_corr = rawdata.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(rawdata_corr, vmin=-1, vmax=1)
fig.colorbar(cax)
names = rawdata_corr.columns.values.tolist()
print(names)
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# - Positive correlation: A goes up when B goes up. Color tend to yellow;
# - Negative correlation: A goes down when B goes up. Collor tend to blue;
# - Perfect correlation: Near 1. Collor is yellow.
# 
# Attributes with high correlation may be redundant, in this case: SibSp and Parch. 

# ### 2. Cleaning and scaling

# So, we have some attributes to work: ** Sex **, ** Age **, ** Pclass **, ** Party **, ** FareCategory **, and ** Embarked **.
# 

# We need to decide which attributes we'll include in our final dataFrame. We may create derivated attributes and remove original attributes as well. In this process we'll try to "clean" the values, treating missing values (NaN) and scale problems.
# Just to give an example, Age is a numerical continuous value, and some Classification algorithms may have problems with that. We cannot "count" based on Age, but if we create something like an "Age Class", that would be fine:
# - Child: 0 to 11
# - Teens: 12 to 17
# - Youngs: 18 to 25
# - Adults: 25 to 50
# - Elders: 51 to ...
# 

# ### Sex
# We need to check for missing values and convert it to a numerical category:

# In[ ]:


rawdata['Sex'][(rawdata.Sex != 'male') & (rawdata.Sex != 'female')].count()


# In[ ]:


rawdata['Sex'].replace(['male','female'],[0,1],inplace=True)


# ### Age
# Age is a continuous value, and algorithms doesn't like it. So, let's do the same we did for the ** FareCategory ** attribute. But first, let's check for missing values. 

# In[ ]:


rawdata.info()


# In[ ]:


np.count_nonzero(np.isnan(rawdata['Age']))


# Hmmm. We have 177 NaN values! This is bad. We could drop the lines with NaN values or we can substitute the NaN by the Age average. 

# Let's find out the impact of these lines with NaN ages...

# In[ ]:


countSurvivorsWithAgeNaN = rawdata[(rawdata.Survived == 1) & (np.isnan(rawdata.Age))]['PassengerId'].count()
countDeathsWithAgeNaN  = rawdata[(rawdata.Survived == 0) & (np.isnan(rawdata.Age))]['PassengerId'].count()
print("Survivors with Age NaN:", countSurvivorsWithAgeNaN)
print("Deaths    with Age Ok :", countDeathsWithAgeNaN)


# Well... Remove these passengers would have an impact on the training because we will have aprox 20% less data. But replace the NaNs by the Age average would impact the precision, because some passengers that survived may receive an Age that normally died in the same case. 
# I believe that user "ashwin" (https://www.kaggle.com/ash316) gave an ingenious solution to the problem, trying to guess the age range for the treatment pronouns included in the names. 
# I will take a more conservative approach and substitute the missing values by the average.

# In[ ]:


survAgeAvg = rawdata[(rawdata.Survived == 1)]['Age'].mean()
print(survAgeAvg)
deadAgeAvg = rawdata[(rawdata.Survived == 0)]['Age'].mean()
print(deadAgeAvg)


# In[ ]:


rawdata[(np.isnan(rawdata.Age))]


# In[ ]:


rawdata.ix[(np.isnan(rawdata.Age)) & (rawdata.Survived == 1), 'Age']=survAgeAvg
rawdata.ix[(np.isnan(rawdata.Age)) & (rawdata.Survived == 0), 'Age']=deadAgeAvg


# Now, let's convert ** Age ** to an ** AgeCategory **
# 
# -    Child: 0 to 11
# -    Teens: 12 to 17
# -    Youngs: 18 to 25
# -    Adults: 25 to 50
# -    Elders: 51 to ...
# 

# In[ ]:


rawdata['AgeCategory']=0
rawdata.ix[rawdata.Age < 12, 'AgeCategory'] = 0
rawdata.ix[(rawdata.Age >= 12) & (rawdata.Age < 18), 'AgeCategory'] = 1
rawdata.ix[(rawdata.Age >= 18) & (rawdata.Age < 25), 'AgeCategory'] = 2
rawdata.ix[(rawdata.Age >= 25) & (rawdata.Age < 51), 'AgeCategory'] = 3
rawdata.ix[rawdata.Age > 51, 'AgeCategory'] = 4


# Finally, let's remove the ** Age ** attribute:

# In[ ]:



rawdata.drop('Age', axis=1)


# ### FareCategory
# Well, ** FareCategory ** labels are intervals. We must convert them to a number:

# - [0, 7.775] = 0
# - (7.775, 8.662] = 1
# - (8.662, 14.454] = 2
# - (14.454, 26] = 3
# - (26, 52.369] = 4
# - (52.369, 512.329] = 5

# In[ ]:


rawdata.drop('FareCategory', axis=1)
rawdata['FareCategory']=0
rawdata.ix[rawdata.Fare <= 7.775, 'FareCategory'] = 0
rawdata.ix[(rawdata.Fare > 7.775) & (rawdata.Fare <= 8.662), 'FareCategory'] = 1
rawdata.ix[(rawdata.Fare > 8.662) & (rawdata.Fare <= 14.454), 'FareCategory'] = 2
rawdata.ix[(rawdata.Fare > 14.454) & (rawdata.Fare <= 26), 'FareCategory'] = 3
rawdata.ix[(rawdata.Fare > 26) & (rawdata.Fare <= 52.369), 'FareCategory'] = 4
rawdata.ix[(rawdata.Fare > 52.369), 'FareCategory'] = 5


# In[ ]:


rawdata.drop('Fare', axis=1)


# ### Embarked
# This is a categorical textual attribute, we need to study and convert it to numerical values. First, let's check for null values...

# In[ ]:


rawdata.info()


# We have 714 passenger's in the DataFrame. Two of them have ** null ** in the Embarked attribute. I'll change them to 'S', because there are more passengers who embarked in Southampton.

# In[ ]:


rawdata.ix[(rawdata.Embarked.isnull()),'Embarked'] = 'S'
rawdata.info()


# In[ ]:


rawdata['Embarked'].value_counts()


# In[ ]:


rawdata['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)


# ### Drop the unecessary attributes

# In[ ]:


rawdata.info()


# In[ ]:


rawdata.drop(['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)


# In[ ]:


rawdata.info()


# ## 3. Training and estimating
# Now we have to train our model, clean the ** test data ** and evaluate, generating the submission file.

# ### Binary classification with TensorFlow
# I will try a Neural net approach with TensorFlow and Keras, using "relu" as the activation function.

# In[ ]:



import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.optimizers import Adam
features = rawdata.drop('Survived', axis=1).values
labels = rawdata['Survived'].values
model = Sequential()
model.add(Dense(128, input_dim=6, kernel_initializer='normal', activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(features, labels)


# Hmmmm. Not bad for a first try... 0.82%

# ### Generating submit file

# In[ ]:


testdata = pd.read_csv("../input/test.csv")
testdata.head()


# ### Cleaning and preparing test data
# I'll not detail the steps here. Refer to the Train data preparation.

# In[ ]:


testdata['Party']=testdata['SibSp'] + testdata['Parch']


# In[ ]:


testdata['Sex'].replace(['male','female'],[0,1],inplace=True)


# In[ ]:


testdata[(np.isnan(testdata.Age))]


# In[ ]:


ageMean = testdata['Age'].mean()
testdata.ix[(np.isnan(testdata.Age)) , 'Age']=ageMean


# In[ ]:


testdata['AgeCategory']=0
testdata.ix[testdata.Age < 12, 'AgeCategory'] = 0
testdata.ix[(testdata.Age >= 12) & (testdata.Age < 18), 'AgeCategory'] = 1
testdata.ix[(testdata.Age >= 18) & (testdata.Age < 25), 'AgeCategory'] = 2
testdata.ix[(testdata.Age >= 25) & (testdata.Age < 51), 'AgeCategory'] = 3
testdata.ix[testdata.Age > 51, 'AgeCategory'] = 4


# In[ ]:


#testdata.drop('FareCategory', axis=1)
testdata['FareCategory']=0
testdata.ix[testdata.Fare <= 7.775, 'FareCategory'] = 0
testdata.ix[(testdata.Fare > 7.775) & (testdata.Fare <= 8.662), 'FareCategory'] = 1
testdata.ix[(testdata.Fare > 8.662) & (testdata.Fare <= 14.454), 'FareCategory'] = 2
testdata.ix[(testdata.Fare > 14.454) & (testdata.Fare <= 26), 'FareCategory'] = 3
testdata.ix[(testdata.Fare > 26) & (testdata.Fare <= 52.369), 'FareCategory'] = 4
testdata.ix[(testdata.Fare > 52.369), 'FareCategory'] = 5


# In[ ]:


testdata.info()


# In[ ]:


testdata['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)


# We will not drop ** PassengerId ** from the dataframe, but we need to remove it before submiting to the model.

# In[ ]:


testdata.drop(['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)


# In[ ]:


testdata.info()


# In[ ]:


test = testdata[['Pclass', 'Sex', 'Embarked', 'Party', 'AgeCategory', 'FareCategory']].values
probabilities = model.predict(test, verbose=0)
listOfList = np.round(probabilities).astype(int).tolist()
predictions = [x for obj in listOfList for x in obj]

result = pd.DataFrame( { 'PassengerId': testdata['PassengerId'], 'Survived' : predictions } )
result.shape
result.info()
result

result.to_csv( 'titanic_pred_cleuton.csv' , index = False )


# In[ ]:





# In[ ]:




