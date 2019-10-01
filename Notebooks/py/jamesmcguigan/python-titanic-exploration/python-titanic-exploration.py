#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Class Imports
import re
import math
import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Utility Functions
sns_percent = lambda x: sum(x)/len(x)*100


# In[ ]:


train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')


# In[ ]:


# Data mappings
for dataset in [test, train]:
    dataset['Gender']     = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
    dataset['CabinClass'] = dataset['Cabin'].astype(str).map(lambda x: re.sub('^(\w)?.*', '\\1', x) if x != "nan" else None )
    dataset['LogFare']    = dataset['Fare'].astype(float).map(lambda x: math.log(x) if x else None)
    dataset['Title']      = dataset['Name'].astype(str).map(lambda x: re.findall('(\w+)\.', x)[0])
train.head() 

#cabin_classes = dataset['Cabin'].astype(str).map(lambda x: re.sub('^(\w)?.*', '\\1', x) if x != "nan" else None ).unique()    
#test.groupby('Title')['Title'].count()   


# # 1. Guessing at random is our null hypothesis
# 50% success rate based on no information, strangely there where entries on the leaderboard worst than this

# In[ ]:


output_random = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived"   : np.random.randint(0,2, size=len(test)) # random number 0 or 1
})
output_random.to_csv('random.csv', index=False); # score 0.51196 (6993/7071)


# # 2. Assume everybody died

# In[ ]:


train["Survived"].map({0: "dead", 1: "alive"}).value_counts()/len(train)


# More people died rather survived, so our next predictive model is just to assume the everybody died. Assuming the test dataset is statistically similar, we should expect around 61% accuracy.

# In[ ]:


output_everybody_dead = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived"   : 0
})
output_everybody_dead.to_csv('everybody_dead.csv', index=False) # score 0.62679 (6884/7071)


# We score 62% which means there are about 1% more casualties in the test dataset

# # Women
# 
# Our next model is just to focus on the women

# In[ ]:


train["Sex"].value_counts()/len(train)


# In[ ]:


survivors  = train[train['Survived'] == 1]
casualties = train[train['Survived'] == 0]
pd.DataFrame({
    "survivors":  survivors["Sex"].value_counts()/len(train),
    "casualties": casualties["Sex"].value_counts()/len(train),
})


# In[ ]:





# As we can see: 
# 
# - 38% / 62% of passengers were dead / alive
# - 65% / 35% of passengers where male / female
# - 31% / 68% of male / female passengers survived 
# 
# So our next predictive model is just to assume the women survive

# In[ ]:


output_everybody_dead = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived"   : test["Gender"]
})
output_everybody_dead.to_csv('only_women_survive.csv', index=False) # score 0.76555 (5384/7071)


# # Children
# As the phrase goes: women and children first!
# 
# So what is the age distribution of the survivors?

# This scores 0.76555 (5384/)

# How many people of each age group survived as a percentage of total people that age

# In[ ]:


train_with_age = train[ ~np.isnan(train["Age"]) ]
survivalpc_by_age = train_with_age.groupby(["Sex","Age"], as_index = False)["Survived"].mean()
#sns.boxplot("Age", "Survived", survivalpc_by_age)

for gender in ["male", "female"]:
    plt.figure()
    sns.lmplot(data=survivalpc_by_age[survivalpc_by_age["Sex"]==gender], x="Age", y="Survived", order=4)
    plt.title("%s survival by age" % gender)
    plt.xlim(0, 80)
    plt.ylim(0, 1)


# As we can see, there is a very different age/survival distribution between the genders.
# 
# - Male children who had not entered their teenage years (<= 12), had a higher survival rate than adult males
# - Maybe a statistical anomaly, but age of 80+ also got you a ticket onto the lifeboat (there where no 80 year women)
# - Female survival, whilst significantly higher than male survival, was actually worse for children and young adults (>=30)
# 
# Our next model is to assume:
# 
# - All women survived
# - All males 12 or under survived
# - All males 80 or over survived
# - All other males died

# In[ ]:


output_women_and_children_first = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived"   : ((test["Sex"] == "female") | ((test["Age"] <= 12) | (test["Age"] >= 80))).astype(int)
})
output_women_and_children_first.to_csv('women_and_children_first.csv', index=False) # score 0.77033 (4523/7071)


# A 0.5% improvement over only women survive, what about just limiting children to toddlers (<=6) which is roughly where the regression line reaches 50% survival for male children

# In[ ]:


output_women_and_toddlers_first = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived"   : ((test["Sex"] == "female") | (test["Age"] <= 6)).astype(int)
})
output_women_and_toddlers_first.to_csv('women_and_toddlers_first.csv', index=False) # score 0.75598 (4523/7071)
# Your submission scored 0.75598, which is not an improvement of your best score. Keep trying!


# Surprisingly this model (75.5%) does even worse than just women (76.5%) or women and children first (77%)

# # Confusion Matrix
# 
# Given that age and gender are probably the two strongest correlations, the next question is to explore what other correlations exist

# In[ ]:


train


# In[ ]:


train_dummies = pd.get_dummies(train, columns=["Title","CabinClass","Embarked"]).corr()
sns.heatmap(train_dummies.corr(), square=True, annot=False)


# In[ ]:


train_dummies.corr()['Survived']


# In[ ]:




