#!/usr/bin/env python
# coding: utf-8

# Loading the data set: 

# In[ ]:


import pandas as pd 
from pandas import Series, DataFrame


# In[ ]:


titanic=pd.read_csv('../input/train.csv')


# Getting to know the data:

# In[ ]:


titanic.head()


# In[ ]:


titanic.info()


# **So here are a few questions I am trying to answer using this data base: **
# #who are the passengers on Titanic?
# #what deck were the passengers on and how does that relate to their class?
# #where did the passengers come from?
# #who was alone and who was with family?
# #what factor helped someone survive the sinking?
# 

# **who are the passengers on Titanic?**

# In[ ]:


import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


sns.catplot('Sex',data=titanic,kind='count', palette='Pastel1')


# *So there are half as many women on board as there are men. Makes sense I guess. There weren't a lot of adveturous women at the time.  Now let's see what the composition was like in terms of class:*

# In[ ]:


sns.catplot('Pclass',data=titanic,kind='count',hue='Sex', palette='Pastel1')


# *Alright, it looks like there are way more men in the third class comapred to the first and the second class. But the number of men are still higher in every class. Now i'm curious to know how many children were on board. I'm going to define anybody below 16 years old as a child: *

# In[ ]:


titanic['Ptype']=np.where(titanic.Age<16,'child',titanic.Sex)


# *Let's see:*

# In[ ]:


titanic.Ptype.value_counts().plot(kind='pie',autopct='%.2f',title='passengers breakdown', figsize=(5,5), cmap='Set3')


# *Alright looks good. So the composition was about 60% men, 30% women, and around 10% children. Now I woud like to know what the compostion looked like in each class, like, were rich people (class 1) bringing their kids with them or where there more poor people (class 3) with children on board?:*

# In[ ]:


fig=sns.FacetGrid(titanic,col='Pclass',hue='Ptype',palette="Set1")
fig.map(sns.kdeplot,'Age')
fig.add_legend()


# In[ ]:


sns.factorplot('Pclass',data=titanic,hue='Ptype',kind='count', palette='Accent')


# *hmm, intersting. looks like there were a lot of kids in the third class. In general looks like most of the passengers were young (based on where the kde plots are peaking). Let's see: *

# In[ ]:


titanic.Age.plot(kind='hist',bins=80)


# Hmmm, just a few people above 50 years old. 

# **Who was alone and who was with family?**
# 

# *Alright, can't help but wonder how many people were on board with family members? the data set has two columns that determine if a passenger had a parent/child, or a sibling on board. Spouses I guess don't count as family based on this data set:)) 
# Ok, first, let's find people who have a family member on board, either a sibling or a child/parent: *

# In[ ]:


family=titanic.loc[(titanic.SibSp !=0) ^ (titanic.Parch!=0)]
family['family']='with family'
titanic_p=titanic.combine_first(family)


# In[ ]:


n=np.nan
titanic_p.family.replace(n,'without family', inplace=True)


# Now let's see what we got:

# In[ ]:


sns.catplot('Ptype',data=titanic_p, hue='family',kind='count', palette='Accent')


# Alright overall, more people are traveling alone rather than with family members. That being said, looks like there are more dependant (with family) women on borad compared to men and even children. I think maybe 16 is a little bit too old to count as a child considering how many teenagers used to travle on their own at the time for work (Ok, I am obviously under the influence of the movie but still...) . Now I'm curious to see who is the youngest child that's traveling alone:

# In[ ]:


kids_no_family=titanic_p.loc[(titanic_p['Ptype']=='child') & (titanic_p['family']=='without family')]


# In[ ]:


kids_no_family.Age.plot(kind='hist')


# Holy moly! there are 17 children under the age of 2 who are traveling without a family member. now that doesn't neccessarily mean they are traveling alone but I wouldn't guess this. Maybe my theory was right about teenagers who travel alone but there aren't that many of them. Let's see: 

# In[ ]:


teens=kids_no_family.loc[kids_no_family['Age']>8]


# In[ ]:


sns.catplot('Age', data=teens,hue='Pclass', kind='count', palette='rainbow')


# So this means that the majority of 9 to 15 year olds who are traveling without family are poor (3rd class)

# Now I want to have an overall image of what the composition of with/without family in each class looks like: 

# **what deck were the passengers on and how does that relate to their class?**

# In[ ]:


titanic_p.head(5)


# Looks like we have a lot of null values. Let's get rid of them: 

# In[ ]:


deck=DataFrame(titanic_p.loc[titanic_p.Cabin.notnull()])
deck.reset_index(drop=True,inplace=True)


# In[ ]:


deck.head(5)


# Alright, now I want to have the deck (A, B, C, D, E, F, ,G) for each row. To do this, I'll define a function that takes out the first element of the string and apply it to all the values in the Cabin column. 

# In[ ]:


def fl(a):
    return list(str(a)[0])


# In[ ]:


levels=[]
for i in range(0,204):
    levels=levels+fl(deck.Cabin.loc[i])
deck['deck']=levels


# let's see:

# In[ ]:


deck.deck.unique()


# hmm, the "T" looks really odd. let's see what we have for T:

# In[ ]:


deck.loc[deck['deck']=='T']


# ok so it's just one 45 year old guy who was traveling alone in first class. He didn't survive. To keep things neat, I will get rid of this row. 

# In[ ]:


deck.drop(index=78, inplace=True)


# Now I want to see what portion of people were in each deck: 

# In[ ]:


deck.deck.value_counts().plot(kind='pie',autopct='%.1f', figsize=(7,7), cmap='Set2')


# Now I want to see how the decks were occupied in terms of class: 

# In[ ]:


deck=deck.sort_values(['deck'])


# In[ ]:


sns.catplot('deck',data=deck,kind='count',hue='Pclass', palette='rainbow')


# OK. Looks like the decks A, B, and C were exclusively occupied by first class passengers and the deck G was exclusivley occupied by third class passengers. All other decks accommdated a combination. 

# **where did the passengers come from?**

# In[ ]:


titanic_p.Embarked.unique()


# Looks like we have some NaN values. I'm just going to clean this up and assign actual city names to the "Embarked" column: 

# In[ ]:


port=DataFrame(titanic_p.loc[titanic_p.Embarked.notnull()])
port.reset_index(drop=True,inplace=True)


# In[ ]:


cities={'S':'South Hampton', 'C':'Cherbourg', 'Q':'Queenstown'}


# In[ ]:


port.Embarked.replace(cities,inplace=True)


# In[ ]:


sns.catplot('Embarked',data=port, kind='count', palette='winter', sharex=True)
sns.factorplot('Embarked',data=port, kind='count', hue='Pclass', palette='winter')
sns.factorplot('Embarked',data=port, kind='count', hue='Ptype', palette='winter')


# so almost everyone who boarded from Queenstown was in 3rd class (Queenstown wasn't economically doing so well at the time maybe?) but interstingly the same numer of women and men boarded from Queenstown. Let's see what the family situation was like: 

# In[ ]:


sns.catplot('Embarked',data=port, kind='count', hue='family', palette='winter', sharex=True)


# In[ ]:


Qtown=port.loc[port['Embarked']=='Queenstown']


# In[ ]:


Qtown.Cabin.replace(np.nan, 'N/A', inplace=True)


# In[ ]:


Qtown.Cabin.value_counts().plot(kind='pie', figsize=(7,7), cmap='Dark2',autopct='%.1f', title='Where were the people from Queenstown located on the ship?')


# Interesting. Looks like the majority of people who boarded in Queenstown were not assigned any specific cabins. Or at least there aren't any records about them. After a simple Google search, I found the route for Titanic and it looks like the ship started its journy from South Hampton (where it loaded the majority of the passengers), made a stop in Cherbourg and then made a last stop in Queenstown. So the people who boarded from Queenstown were probably poor people who were just happy to be on the ship and travel to US, regardless of the cabin types they were getting. 

# **#what factor helped someone survive the sinking?***

# It's time to see who survived the trip and who didn't. 

# Let's take a quick look at what the survival situation was like as a function of sex, age, deck, class, etc.. . 

# In[ ]:


titanic_p['Survival']=np.where(titanic_p['Survived']==1, 'yes', 'no')


# In[ ]:


f, ax= plt.subplots(2,2,figsize=(15,15))
plt.subplot(2,2,1)
sns.countplot('Survival',data=titanic_p,palette='Set2')
plt.subplot(2,2,2)
sns.countplot('Survival',data=titanic_p, hue='Ptype',palette='Set1')
plt.subplot(2,2,3)
sns.countplot('Survival',data=titanic_p, hue='Pclass', palette='Set2')
plt.subplot(2,2,4)
sns.countplot('Survival',data=titanic_p, hue='family',palette='Set1' )


# Hmm, some quick observations: 
# * a big portion of passengers have died. a quick estimation would give a 40-60 ratio between passengers who survived and the ones who didn't.
# * more women and children have survived the sink than men. 
# * the supposedly rich people (on first class) have a higher survival rate.
# * the people who didn't have family members on board had a higher probabilty of death 

# if we define the chance of survival as the ratio of (count of surived)/(count of total) in each group of people, we can look at the chance of survival as a function of different conditions: 

# In[ ]:


sns.factorplot('Pclass','Survived',data=titanic_p,col='Ptype', hue='family')


# Phewww. Look at this. let's see what we have here: 
# * so if you were a man, unfortunately it looks like you have a low survival chance on Titanic and it doesn't matter if you are with family members or not. Although if you are a rich man who is in first class on Titanic, you are twice as more likely to survive. 
# * If you were a woman on Titanic your chances for survival would be pretty damn good! if you are a rich woman (regardless of whether you are traveling with family or not) you will certianly survive. Your chances of survival in 2nd class are still close to 100% ( pretty great!) but if you were in 3rd class, your chances of survival would plummet by 50%. Bad news for poor women!
# * If you were a child (or a young adult in our case) on Titanic, your chances of survival would really depend on whether you are with family members or not! even if you were a 3rd class passenger, your chances of survival would be twice as much if you were accompanied by a family member. 

# we can go a little bit deeper and take a closer look at what happned on each deck (remember that we discarded a whole bunch of data because we didn't have exact records for every passenger's cabin): 

# In[ ]:


sns.factorplot('deck','Survived',data=deck,hue='Ptype', palette='Set1')


# most of the female casualities died on deck "G". Aside from that, all other decks look pretty safe for women. Deck "C" looks particularly dangerous for kids though.  

# Just to get a good perspective on what happened, let's look at the composition of who died and who survived the crash: 

# In[ ]:


grid = sns.FacetGrid(data=titanic_p, col='Survival', row='Pclass', size=4.2, aspect=1, hue='Ptype', palette='Set2')
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

