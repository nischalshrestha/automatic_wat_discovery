#!/usr/bin/env python
# coding: utf-8

# # Titanic Competition: Exploring Data - Iteration 1
# 
# Welcome! This kernel is part of the *Titatic competition learning series* which can be accessed from <a href="https://www.kaggle.com/sergioortiz/titanic-competition-a-learning-diary">here</a>.  
# 
# Let's start with the data exploration basics...

# In[ ]:


import os
import pandas as pd

input_io_dir="../input"

original_train_data=pd.read_csv(input_io_dir+"/train.csv")
original_test_data=pd.read_csv(input_io_dir+"/test.csv")
print('original_train_data',original_train_data.shape)
print('original_test_data',original_test_data.shape)


# ## General overview
# Let's first have a look at some data with basic pandas DataFrame functions...

# In[ ]:


original_train_data.head()


# In[ ]:


print('Training data --------------')
print(original_train_data.info())
print('Test data ------------------')
print(original_test_data.info())


# In[ ]:


original_train_data.describe()


# In[ ]:


original_test_data.describe()


# ### General overview: conclusions
# * Small training data set - less than 1,000 rows. 
# * Features by type
#   * Categorical
#     * Pclass: ordinal
#     * Sex
#     * Embarked
#   * Numeric
#     * Age: continuous
#     * Fare: continuous
#     * SibSp: discrete
#     * Parch: discrete
#   * Other
#     * Name
#     * Ticket
#     * Cabin
# * Missing data
#   * Age: some missing values - strategy relevant for training
#   * Cabin: many missing values - watch out as training may not be good on such a reduced data set
#   * Embarked: few missing values - strategy unlikely to affect training
# * Potential outliers (Fare - max is far way from mean+-std)

# ## Potential barriers for learning
# Let's analyse different factors that can hinder learning
# ### Comparing training and test data sets
# Learning models can be ineffective when data distribution is very different between training and test sets.<br/>
# Let's explore this subject for a while...

# In[ ]:


def ExploreCategoricalVariable(dataSet,variableName):
    print('Variable:'+variableName)
    print(dataSet[variableName].value_counts()/len(dataSet[variableName]))
    print('')

print('----------------------- Training set')
ExploreCategoricalVariable(original_train_data,'Sex')
ExploreCategoricalVariable(original_train_data,'Pclass')
ExploreCategoricalVariable(original_train_data,'Embarked')
print('----------------------- Test set')
ExploreCategoricalVariable(original_test_data,'Sex')
ExploreCategoricalVariable(original_test_data,'Pclass')
ExploreCategoricalVariable(original_test_data,'Embarked')


# In[ ]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
fig, axarr = plt.subplots(4, 2, figsize=(12, 8))

original_train_data['Age'].hist(ax=axarr[0][0])
original_test_data['Age'].hist(ax=axarr[0][1])
original_train_data['Fare'].hist(ax=axarr[1][0])
original_test_data['Fare'].hist(ax=axarr[1][1])
original_train_data['Parch'].hist(ax=axarr[2][0])
original_test_data['Parch'].hist(ax=axarr[2][1])
original_train_data['SibSp'].hist(ax=axarr[3][0])
original_test_data['SibSp'].hist(ax=axarr[3][1])


# ### Potential barriers for learning: conclusions
# There are no significant differences in both categorical and numeric data - that is, it appears to be evenly distributed between the two data sets.  
# On the other hand, there can be limitations derived from the reduced data set size - e.g. some learning models can improve its accuracy with increased datasets. Learning curves will help to evaluate if this is the case.
# Finally, existing features present very different values and this can hinder learning. Data values must be scaled and normalised.

# ## Trends and correlations
# Let's start identifying correlations with the corr function.
# This will be useful only for ordinal variables as it reflects the extent to which a variable (e.g. Survived) varies when other features values change.  

# In[ ]:


original_train_data.corr()


# In the table above, we will look at the the second column - Survived column.
# As we can see, the following fields are clearly correlated:
# * Fare
# * Pclass
# 
# Unlike I would have expected, it is curious how Age is not directly correlated with survival.
# Both SubSp and Parch are not directly related with the survival.  

# For categorical data (non-ordinal), let's analyse data manually...

# In[ ]:


original_train_data.groupby('Sex')['Survived'].sum().plot.bar(stacked=True)


# In[ ]:


original_train_data.groupby('Embarked')['Survived'].sum().plot.bar(stacked=True)


# In both cases, it seems there is a clear relationship with survival - e.g. female are more likely to survive than male - and also people embarking in Southampton.

# ### Trends and correlations: conclusions
# Initial exploration only directly relates a small subset of features with survival:
# * Sex
# * Embarked
# * Fare
# * Pclass
# 
# This does not mean that the rest of features are not related - may be it's only that the relationship is not obvious.

# ## Building new features
# The following features are good candidates for creating new features:
# * Name
# * Parch
# * SibSp 
# * Cabin 
# * Ticket

# ### Name: exploring Title feature
# The most obvious case is extracting the title feature from the name string.
# Let's  create an initial extraction routine and analyse how the feature is correlated with survival

# In[ ]:


original_train_data.head()


# In[ ]:


import numpy as np

def extractTitleFromNameForExploring(name):
    pos_point=name.find('.')
    if pos_point == -1: return ""
    wordList=name[0:pos_point].split(" ")
    if len(wordList)<=0: return ""
    title=wordList[len(wordList)-1]
    return title

# Get a list with different titles
training_titleList=np.unique(original_train_data['Name'].apply(lambda x: extractTitleFromNameForExploring(x)))
for title in training_titleList:
    training_titleSet=original_train_data[original_train_data['Name'].apply(lambda x: title in x)]
    # Evaluate survival rate for each subset
    survivalRate=float(len(training_titleSet[training_titleSet['Survived']==1]))/float(len(training_titleSet))
    print('Title['+title+'] count:'+str(len(training_titleSet))+' survival rate:'+str(survivalRate))


# In[ ]:


# Let's check test data set values - just to confirm the training will consider all potential values
test_titleList=np.unique(original_test_data['Name'].apply(lambda x: extractTitleFromNameForExploring(x)))
for title in test_titleList:
    test_titleSet=original_test_data[original_test_data['Name'].apply(lambda x: title in x)]
    print('Title['+title+'] count:'+str(len(test_titleSet)))


# Ups...there is a new value (Dona) which was not present in the training data set.

# The following title values are numerous and will be very useful for the learning model:
# * Miss
# * Mr
# * Mrs
# * Master  
# ...but others are not so numerous, as Capt or Jonkheer.
# It is also remarkable:
# * Some titles seem redundant -e.g. Mme = Miss or Mlle=Mrs  
# * Some titles can be grouped in categories
#   * Army-related
#     * Capt
#     * Col
#     * Major
#   * Nobility
#     * Countess
#     * Lady
#     
#  Let's group titles in these categories and analyse again data... 
# 
# 
# 
# 
# 
# 

# In[ ]:


import numpy as np

def multipleReplace(text, wordDic):
    for key in wordDic:
        if text.lower()==key.lower():
            text=wordDic[key]
            break
    return text

def normaliseTitle(title):
    wordDic = {
    'Mlle': 'Miss',
    'Ms': 'Mrs',
    'Mrs':'Mrs',
    'Master':'Master',
    'Mme': 'Mrs',
    'Lady': 'Nobility',
    'Countess': 'Nobility',
    'Capt': 'Army',
    'Col': 'Army',
    'Dona': 'Other',
    'Don': 'Other',
    'Dr': 'Other',
    'Major': 'Army',
    'Rev': 'Other',
    'Sir': 'Other',
    'Jonkheer': 'Other',
    }     
    title=multipleReplace(title,wordDic)
    return title
def extractTitleFromName(name):
    pos_point=name.find('.')
    if pos_point == -1: return ""
    wordList=name[0:pos_point].split(" ")
    if len(wordList)<=0: return ""
    title=wordList[len(wordList)-1]
    normalisedTitle=normaliseTitle(title)
    return normalisedTitle

# Get a list with different titles
titleList=np.unique(original_train_data['Name'].apply(lambda x: extractTitleFromName(x)))
for title in titleList:
    titleSet=original_train_data[original_train_data['Name'].apply(lambda x: title in extractTitleFromName(x))]
    # Evaluate survival rate for each subset
    survivalRate=float(len(titleSet[titleSet['Survived']==1]))/float(len(titleSet))
    print('Title['+title+'] count:'+str(len(titleSet))+' survival rate:'+str(survivalRate))


# Some of the categories such as Army or Nobility have a so few samples that the learning algorithm may not be able to learn effectively.  
# In addition to these categories, we may be able to extract additional features related with Age (e.g. Master is a young boy) or Marital Status. However, we will stop here and postpone this exploration for the next iteration.
# 
# ### Parch/SibSp: exploring IsAlone and FamilySize feature
# Combining these two features we can extract whether a passenger traveled alone and family size. 
# Let's explore these possibilities...

# In[ ]:


original_train_data['IsAlone']=(original_train_data["SibSp"]+original_train_data["Parch"]).apply(lambda x: 0 if x>0 else 1)
original_train_data['FamilySize']=original_train_data["SibSp"]+original_train_data["Parch"]+1

original_train_data.corr()


# In[ ]:


import numpy as np
total=original_train_data.groupby('IsAlone')['PassengerId'].count()
survived=original_train_data[original_train_data['Survived']==1].groupby('IsAlone')['PassengerId'].count()
notSurvived=original_train_data[original_train_data['Survived']==0].groupby('IsAlone')['PassengerId'].count()
df=pd.concat([total, survived,notSurvived], axis=1, sort=True)
df.fillna(0,inplace=True)
df.columns=['Total','Survived','NotSurvived']
df=df.astype('int64')
print(df)
df.loc[:,['Survived','NotSurvived']].plot.bar(stacked=True,figsize=(20,8))


# Ups...travelling Alone appears to be negatively correlated with Survival - almost 42% of lonely passengers didn't survive!

# In[ ]:


print("FamilySize value distribution")
print(original_train_data['FamilySize'].value_counts()/len(original_train_data))


# In[ ]:


import numpy as np
total=original_train_data.groupby('FamilySize')['PassengerId'].count()
survived=original_train_data[original_train_data['Survived']==1].groupby('FamilySize')['PassengerId'].count()
notSurvived=original_train_data[original_train_data['Survived']==0].groupby('FamilySize')['PassengerId'].count()
df=pd.concat([total, survived,notSurvived], axis=1, sort=True)
df.fillna(0,inplace=True)
df.columns=['Total','Survived','NotSurvived']
df=df.astype('int64')
print(df)
df.loc[:,['Survived','NotSurvived']].plot.bar(stacked=True,figsize=(20,8))


# Survival varies with family size  - families with 3 and 4 members are the most likely to survive.

# ### Cabin: exploring passengers with NoCabin defined
# Cabin is one of the most unreliable features as there are many missing values.
# However, let's explore if defining the cabin is related with survival
# 

# In[ ]:


original_train_data['NoCabin']=original_train_data['Cabin'].isnull().apply(lambda x: 1 if x is True else 0)
original_train_data.corr()


# In[ ]:


total=original_train_data.groupby('NoCabin')['PassengerId'].count()
survived=original_train_data[original_train_data['Survived']==1].groupby('NoCabin')['PassengerId'].count()
notSurvived=original_train_data[original_train_data['Survived']==0].groupby('NoCabin')['PassengerId'].count()
df=pd.concat([total, survived,notSurvived], axis=1, sort=True)
df.fillna(0,inplace=True)
df.columns=['Total','Survived','NotSurvived']
df=df.astype('int64')
print(df)
df.loc[:,['Survived','NotSurvived']].plot.bar(stacked=True,figsize=(20,8))


# Strange but true, it seems that those with no cabin assigned are less likely to survive than those with cabin.
# It would be wise to know exactly what 'not having a cabin assigned' really means.
# 
# Let's know have a look at shared cabins - are people sharing cabins more likely to survive?
# 

# In[ ]:


import numpy as np
# Group data to detect sharing of cabins - excluding missing  values
cabinList=original_train_data[original_train_data['Cabin'].notnull()==True].groupby('Cabin')['PassengerId'].count()
cabinList=cabinList.reset_index()
cabinList.columns=['Cabin','Count']
print('Distribution of people per cabin - not considering those with missing cabin')
print(cabinList['Count'].value_counts())

# Add new column to indicate number of people a passenger is sharing with
# -1 means there is no data to compute the feature
def extractCabinSharedWithFeature(name):
    if (str(name)!='nan'):
        row=cabinList.loc[cabinList['Cabin'] == name]
        count=row['Count']-1
        return count
    else:
        return -1

original_train_data['CabinSharedWith']=original_train_data['Cabin'].apply(lambda x: extractCabinSharedWithFeature(x)).astype(int)
# Let's now analyse this new column
total=original_train_data[original_train_data['CabinSharedWith']!=-1]['PassengerId'].count()
survived=original_train_data[(original_train_data['CabinSharedWith']!=-1) & (original_train_data['Survived']==1)].groupby('CabinSharedWith')['PassengerId'].count()
notSurvived=original_train_data[(original_train_data['CabinSharedWith']!=-1) & (original_train_data['Survived']==0)].groupby('CabinSharedWith')['PassengerId'].count()
survivedPercent=survived/total
notSurvivedPercent=notSurvived/total
print('Survivor distribution by feature CabinSharedWith')
print(survivedPercent)
print('NotSurvivor distribution by feature CabinSharedWith')
print(notSurvivedPercent)
df=pd.concat([survived,survivedPercent,notSurvived,notSurvivedPercent], axis=1, sort=True)
df.fillna(0,inplace=True)
df.columns=['Survived','SurvivedPercent','NotSurvived','NotSurvivedPercent']
df.loc[:,['Survived','NotSurvived']].plot.bar(stacked=True,figsize=(20,8))


# Interesting - notice how survival rate is higher among those passengers sharing cabin.
# We might wonder whether this is true or those sharing cabin have something in common.
# Let's search for correlations...

# In[ ]:


original_train_data.corr()


# There is some correlation with socio-economic status - Fare/Pclass.
# In case these are relevant variables for training, I wonder whether this new variables will contribute much or will be somewhat redudant.

# ### Ticket: exploring TicketType
# The ticket feature can be difficult to explore as contains text and code information altogether.  
# Let's explore the field and try to build something useful...

# In[ ]:


original_train_data['Ticket'].head(10)


# It seems what tickets have prefixes - we will extract them and try to find patterns.

# In[ ]:


def getTicketType(name, normalise):
    item=name.split(' ')
    itemLength=len(item)
    if itemLength>1:
        ticketType=""
        for i in range(0,itemLength-1):
            ticketType+=item[i].upper()
    else:
        ticketType="NORMAL"
    if normalise==True:
        ticketType= ticketType.translate(str.maketrans('','','./'))
    return ticketType

# Let's list what we have - first view without normalising
training_itemList=[]
for ticket in original_train_data['Ticket']:
    training_itemList.append(getTicketType(ticket,False))
ticketTypeList=np.unique(training_itemList)
print("Ticket type values: no normalisation")
print(ticketTypeList)


# Some of the tickets types are quite similar.
# Let's normalise them so that they are grouped.

# In[ ]:


training_itemList=[]
for ticket in original_train_data['Ticket']:
    training_itemList.append(getTicketType(ticket,True))
ticketTypeList=np.unique(training_itemList)
print("Ticket type values: normalisation")
print(ticketTypeList)


# Now, we will explore correlation between these values and survival...

# In[ ]:


pd.set_option('display.max_columns', None)
original_train_data['TicketType']=original_train_data['Ticket'].apply(lambda x: getTicketType(x,True))
total=pd.DataFrame(original_train_data.groupby('TicketType')['PassengerId'].count())
total.columns=['Total']
survived=pd.DataFrame(original_train_data[original_train_data['Survived']==1].groupby('TicketType')['PassengerId'].count())
survived.columns=['Survived']
notSurvived=pd.DataFrame(original_train_data[original_train_data['Survived']==0].groupby('TicketType')['PassengerId'].count())
notSurvived.columns=['NotSurvived']


# In[ ]:


# Let's merge all ticket type in the same list
df_all=total
df_all=df_all.merge(survived,left_index=True, right_on="TicketType")
df_all=df_all.merge(notSurvived,left_on='TicketType',left_index=True, right_on="TicketType")
df_all['Ratio']=df_all['Survived']/df_all['Total']
df_all.loc[:,['Ratio']].plot.bar(figsize=(20,8))
df_all


# Same concern as with the previous feature - what if this feature is highly correlated with socio-economic status?
# It may be redundant and ineffective for training...
# Let's explore the data directly....in particular, the ticketType with the highest ratio.

# In[ ]:


original_train_data[original_train_data['TicketType']=='FCC']


# Interesting! It seems that  three of the members belong to the same family...  
# Let's take note for the next rounds of exploration...

# ### Devising new features: conclusions
# These are the conclusions after work on this section:  
# * New features
#   * Title
#   * IsAlone
#   * FamilySize
#   * NoCabin
#   * CabinSharedWith
#   * TicketType
#  * Doubts on whether some of these features will add more variance and enrich the learning model.
