#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# This is my first attempt at a kernel, as well as setting a personal benchmark for what a machine learning algorithm should beat. Basically, this is a simple analysis using mostly brain-only, but a useful guide to how simple can still be good as it scores over 80%.
# 
# First, we import packages and files.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
plt.style.use('fivethirtyeight')

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#import csv-files and merge the sets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
merged = train.append(test)


# Let's then take an overview the data.

# In[ ]:


merged.info()


# This gives us some information on what we have to work with. The full dataset contains 1309 entries, where 891 are from the training set (as evidenced by the "Survived" column). We have complete information on Name, Parch, PassengerID, Pclass, Sex, SibSp and Ticket, a few missing values on Fare and Embarked.
# 
# Let's now take a look at some of the data itself, and see what we can learn.

# In[ ]:


merged.head()


# Some additional information might also be gained from the "Name" column, as it contains titles. This can possibly tell us something about sex, age and status, all in one parameter. In addition, the "Cabin" column could potentially tell us something about accomodations, and that could potentially be useful.
# 
# Now the first thing we need to establish, is some sort of baseline hypothesis. The first one here is to see if there is a difference in survival between men and women.

# In[ ]:


train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
plt.show()
train[['Sex','Survived']].groupby(['Sex']).mean()


# Indeed there is. Females are WAY more likely to survive then males. Just predicting that females survive and males die would get us a long way. And that is a very important finding. Because now things change a lot. Any and all analysis from this point forward is to try to find exceptions to that rule. In other words, which females die and which males survive?
# 
# This is where the titles might come in handy. Let's extract those and see what we can find.

# In[ ]:


#Extract title and group them
Title=[]
for i in range(len(merged)):
    names = merged.Name.values[i].replace('.',',').split(', ')
    title = names[1]
    Title.append(title)
merged['PTitle'] = Title

pd.crosstab(merged['PTitle'],merged['Sex'])


# They do indeed mostly split along sexes (Dr being the exception). We'll then extract the four major groups, and lump the rest into a rare group. We'll also compare these with the Pclass to see if that gives us more additional information.

# In[ ]:


merged['PTitle'] = merged['PTitle'].replace(['Lady', 'the Countess','Mlle', 'Ms', 'Mme', 'Dona','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Rare')
merged[['Sex','PTitle','Pclass','Survived']].groupby(['Sex','PTitle','Pclass']).mean()


# That should give us a bit more info. On the female side, it looks like almost all passengers in class 1 and 2 made it, while only half in third class did. We will investigate that further. In addition, it also looks like most of the Masters survived, while there might be some shot of survival for the rest of the men if they were in Pclass 1. This will also be further explored.
# 
# We will start this exploration by looking at family size. This is done by adding the "SibSp" and "Parch" columns, as well as the individual him/herself.

# In[ ]:


#Add family size groups
merged["FamilySize"] = merged["SibSp"]+merged["Parch"]+1
merged[['FamilySize','Survived']].groupby(['FamilySize']).mean().plot.bar()
plt.show()


# Looks like a FamilySize between 2 and 4 gives a better than average chance of survival, so let us group these.

# In[ ]:


merged.loc[merged['FamilySize'] == 1, 'Fsize'] = 'Alone'
merged.loc[(merged['FamilySize'] > 1)  &  (merged['FamilySize'] < 5) , 'Fsize'] = 'Small'
merged.loc[merged['FamilySize'] >4, 'Fsize'] = 'Large'
fem_analysis = merged[merged['Sex'] == 'female']
fem_analysis[['PTitle','Pclass','Survived','Fsize']].groupby(['Pclass','PTitle','Fsize']).mean()


# Looking first at the females, it seems as though being in Pclass 3 with a large family is deadly. Quite a few of the other women with Pclass 3 also seem to hover around 50%, so let's see if where they embarked has any impact. So this we will predict that these women die, while all the rest survive.
# 
# Now let's take a look at the men.

# In[ ]:


fem2_analysis = fem_analysis[ (fem_analysis['Pclass'] == 3) &
                              (fem_analysis['Fsize'] != 'Large')]
fem2_analysis[['PTitle','Embarked','Survived']].groupby(['PTitle','Embarked']).mean()


# In[ ]:


fem2_analysis[['PTitle','Embarked','Survived']].groupby(['PTitle','Embarked']).count()


# Looks like being a Miss in Pclass 3 who embarked at Southampton puts us below 50%. There are also enough of them in the dataset that it makes sense to include that as a parameter of females who die. The rest we predict will survive.
# 
# Now let's take a look at the men.

# In[ ]:


men_analysis = merged[merged['Sex'] == 'male']
men_analysis[['PTitle','Pclass','Survived','Fsize']].groupby(['PTitle','Pclass','Fsize']).mean()


# Again, Masters in first and second class survive, as do those in small families in Pclass 3. So these are exceptions to the rule that males die.
# 
# For the rest, it looks like they might have a shot if they are in Pclass 1 and not in a large family. We will have to slize further though, and will combine this with info on whether they have a cabin or not.

# In[ ]:


men2_analysis = men_analysis[(men_analysis['Pclass'] == 1) &
                             (men_analysis['PTitle'] != 'Master') &
                             (men_analysis['Fsize'] != 'Large')]
men2_analysis['HasCabin'] = men2_analysis['Cabin'].notnull()
men2_analysis[['PTitle','HasCabin','Survived']].groupby(['PTitle','HasCabin']).mean()


# And it looks like we have another exception to the rule, in that males with rare titles in Cabins on Pclass 1 without large families are more likely to survive than not.
# 
# Let us look at the categories we have established.

# In[ ]:


merged['Group'] = merged['Sex']

#Females who don't survive
merged.loc[(merged['Sex'] == 'female') & 
           (merged['Fsize'] == 'Large') &
           (merged['Pclass'] == 3), 'Group'] = 'Females, large family, class 3'

merged.loc[(merged['PTitle'] == 'Miss') & 
           (merged['Fsize'] != 'Large') &
           (merged['Pclass'] == 3) &
           (merged['Embarked'] == 'S'), 'Group'] = 'Miss, class 3, embarked Southampton'

#Males who survive
merged.loc[(merged['PTitle'] == 'Master') & 
           ((merged['Pclass'] < 3) | 
            (merged['Fsize'] == 'Small')), 'Group'] = 'Masters in Pclass 1&2 or small families'

merged.loc[(merged['PTitle'] == 'Rare') & 
           (merged['Sex'] == 'male') & 
           (merged['Pclass'] == 1) &
           (merged['Cabin'].notnull()) &
           (merged['Fsize'] != 'Large'), 'Group'] = 'Rare title, Cabin Pclass 1, not large'

merged[['Group','Survived']].groupby(['Group'], sort=False).mean().plot.bar()
plt.show()
merged[['Group','Survived']].groupby(['Group'], sort=False).mean()


# This will then be our basis for predictions. Those groups with survival above 50% are predicted to survive, the rest are not.

# In[ ]:


merged['Predict'] = 1
merged.loc[merged['Group'] == 'male', 'Predict'] = 0
merged.loc[merged['Group'] == 'Females, large family, class 3', 'Predict'] = 0
merged.loc[merged['Group'] == 'Miss, class 3, embarked Southampton', 'Predict'] = 0
merged.loc[merged['Group'] == 'Masters in Pclass 1&2 or small families', 'Predict'] = 1
merged.loc[merged['Group'] == 'Rare title, Cabin Pclass 1, not large', 'Predict'] = 1

#Score accuracy on training set
train_set = merged[merged['Survived'].notnull()]
train_set['Score'] = train_set['Survived'] == train_set['Predict']
print("Accuracy training set:")
print(train_set['Score'].mean())


# A decent score for using simple exceptions and just our brain. So let us run it on the test data and see.

# In[ ]:


#select test set          
test_analysed = merged[merged['Survived'].isnull()]

#See how it predicts various titles will do
test_analysed[['PTitle','Predict']].groupby(['PTitle']).mean().plot.bar()
plt.show()
test_analysed[['Sex','PTitle','Predict']].groupby(['Sex','PTitle']).mean()


# These look pretty decent. We know there will be Mr that survive, but most will not. However, about 60% of those with the title Master or Rare does, as well as most of the females. So let us prepare the submission.

# In[ ]:


#Set solution output
my_solution = pd.DataFrame({'PassengerId': test_analysed['PassengerId'],
                            'Survived':test_analysed['Predict']})
my_solution.to_csv('submission.csv', index = False)


# The submitted solution scores over 80%, which is very nice given it is a simple model without the use of machine learning.
# 
# Would love to get input on how to apply machine learning to this basic setup in order to improve predictions. Primarily by identifying some more Mrs. that will not survive and around 12% of the Mr. that do. Feedback would be much appreciated.
