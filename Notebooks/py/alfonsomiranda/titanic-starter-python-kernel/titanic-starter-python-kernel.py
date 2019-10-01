#!/usr/bin/env python
# coding: utf-8

# # Titanic
# 
# This is an introductory Kernel for the Titanic challenge in Kaggle, it is intended to serve as a tutorial and introduction to the applied techniques while I explore them. 
# 
# I have tried my best to explain the actions and rationale on each section and have tried to include the relevant links and references that I have found to be useful while addressing that particular step, some of the approaches I apply have been adapted from other kernels on the challenge (like those from Chris Deotte and Konstantin).
# 
# I will start by setting up the required libraries and utilities that we will be using through this exercise:
# 
# # 1-) Environment setup
# 
# We start by loading the required libraries and setting some conditions for the environment
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
sns.set(style='darkgrid', context='notebook', palette='coolwarm',font_scale=1.5)
get_ipython().magic(u'matplotlib inline')

from IPython.core.debugger import set_trace

# Gathering the libraries to pre-process the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Gathering the required libraries for the models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import tensorflow as tf

# Getting the libraries to optimize and assess performance
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV

# Turning off warnings
import warnings
warnings.filterwarnings('ignore')


# #### After we have the right libraries on board let's get the data set

# In[ ]:


# The provided data sets
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# A join sand box to assess what we got and to play around without disturbing the originals
join_data = pd.concat([train_data,test_data])
join_data.head()


# # 2-)  Exploratory Data Analysis (EDA)
# 
# During this phase we are going to focus on understanding the data and defining our approach, not modifying the data in any way
# 
#     - Data set profiles (test + train)
#     - Quick glance and the feature relevance for survival
#     - Correlations and significant relations
#     - Potential imputation and grouping strategies
# 
# #### Let's start by getting a good sense of the data we are receiving

# In[ ]:


fig = plt.figure(figsize=(20,10))
sns.heatmap(join_data.isnull(),yticklabels=False,cbar=False)


# In[ ]:


print('\n==== Training Data =====')
print(train_data.info())
print('\n==== Testing Data =====')
print(test_data.info())
print('\n==== Join Data =====')
print(join_data.info())


# In[ ]:


print('\n==== Training Data =====')
print(train_data.describe())
print(train_data.describe(include=['O']))
print('\n==== Testing Data =====')
print(test_data.describe())
print(test_data.describe(include=['O']))
print('\n==== Join Data =====')
print(join_data.describe())
print(join_data.describe(include=['O']))


# #### =============================================================================================
# #### At the first glance we can see we are getting a total of 1309 record, 891 on the training set and 418 on the testing set. 12 features, 7 numeric and 5 are strings
# 
# Quick observations:
#     - Most of the features are fairly complete, Age have about 20% of values missing, Cabin is only available for about 1/5 of the records
#     - There are a handful records where the port of embarkation and the fare are missing
#     - only 38% of the listed passengers survived
#     - There are only 3 ports of embarkation
#     - The passenger ID seems to be just a key to the list but not really providing any information
#     
# #### Let's start going by each feature and see their relevance for survival, starting with:
# 
# ## === Age ===
# 

# In[ ]:


# Plotting the survival rate given the age
g = sns.FacetGrid(train_data,hue="Survived",palette='magma',height=6,aspect=2).add_legend()
g = g.map(plt.hist,'Age',bins=40,alpha=0.7)


# As it might be expected, younger passengers survived at a higher rate than older passengers, also, exploring a little bit more the age and sex relationships with the gender

# In[ ]:


# Making the plot much larger than the default for better visualization
fig = plt.figure(figsize=(20,10))
sns.boxplot(x='Pclass',y='Age',hue='Sex',data=train_data,palette='magma')


# We can see that there are significant diferences between the average ages per class and per gender, we can use the mean values to impute the missing value according to what class and gender the record shows.

# In[ ]:


# Getting the average age for each group
join_data.pivot_table(values='Age',index='Pclass',columns='Sex')


# In[ ]:


# For the imputation strategy we are going to use the average ages we capture before per gender and class
# to achieve this objective we define a short function
def AgeImp (Columns):
    # Gattering the fields:
    Age = Columns[0]
    Pclass = Columns[1]
    Sex = Columns [2]
    # First, we only impute if the age value is missing
    if pd.isnull(Age):
        if Pclass == 3 :# Third class
            if Sex == 'male':
                return 26
            else:
                return 22.2
        if Pclass == 2 :# Second class
            if Sex == 'male':
                return 30.9
            else:
                return 27.5
        if Pclass == 1 :# First class
            if Sex == 'male':
                return 41
            else:
                return 37
    else:
        return(Age)


# Now, let's explore class a little bit more
# 
# ## === Class ===

# In[ ]:


# Plotting some of the relevant metrics along with Class to see if there is further differentiation

g = sns.FacetGrid(train_data, col='Survived',row='Sex',hue='Embarked',
                  palette='magma',height=5, aspect=1.5,margin_titles=True)
g = g.map(sns.countplot,'Pclass',order=[1,2,3]).add_legend()


# In[ ]:


# Looking also at the numbers
train_data.pivot_table(values='Survived',index=['Pclass','Embarked'],columns='Sex',margins=True,margins_name='Total')


# In[ ]:


# Looking at the sizes of this groups on our train set to understand the probabilities above
train_data.pivot_table(values='Ticket',index=['Pclass','Embarked'],columns='Sex',margins=True,margins_name='Total',
                       aggfunc=(lambda x: x.count()))


# #### This is quite a significant influence, we can see that 74% of Females (more than 92% if 1st or 2nd class) survived while only around 19% of the men did, also, 63% of first class survived while just 24% of the third class did, there seems to be a close relationship with the port of embarkation with the first class passengers, we will explore this later once we explore the correlations after the categorical variables have been transformed
# 
# With the current knowledge we could predict with 90%+ certainty that 80 passengers (Females from 1st and Second class) will survive, also, looking at the embarkation ports, only 13% of the third class males that embarked in 'S' survived (even less for 'Q') this give us another 119 passengers, we might hard code this rules (that already set a floor of 87% accuracy or higher) and only rely on our predictions for other groups.
# 
# We have looked already at Age, Class along with Sex in both cases, let's move on to:
# 
# ## === Cabin ===
# 
# For the particulars of the Cabin, during the initial exploration we saw it was available for only around 1/5 (295 in total) of the records, let's explore a little bit more. The first character of the 'cabin' seems to contain the 'Deck', this might be relevant for survival, also, might be highly correlated with the 'class', let's explore this a bit
# 

# In[ ]:


# Capturing the first character, 'n' will be shown for the missing values (NaN)
join_data['Deck'] = join_data['Cabin'].apply(lambda x: str(x)[0])
print(join_data['Deck'].value_counts())
join_data.pivot_table(values='Survived',index='Deck',columns='Pclass',margins=True)


# #### The deck identifier (or absence of it) seem to be significant to the survival probability so we should extract and use this feature, however, a few decks have so few records (T, G) that we should reclassify those to 'n' (not available), let's create a quick dictionary from the list above to accomplish this task later

# In[ ]:


# We keep and ordinal relationship where A > B > C since there is an ordinal relationship of the decks
DeckDict = {'n':10,
            'A':0,
            'B':1,
            'C':2,
            'D':3,
            'E':4,
            'F':5,
            'T':6,
            'G':7}


# ## === Embarked ====

# In[ ]:


# Now looking at the port of embarkation
join_data['Embarked'].value_counts()
join_data.pivot_table(values='PassengerId',index='Embarked',columns='Pclass',
                      aggfunc=(lambda x: x.count()),margins=True,margins_name='Total')


# In[ ]:


# looking at survival rates on the smae populations
train_data.pivot_table(values='Survived',index='Embarked',columns='Pclass',
                      margins=True,margins_name='Total')


# It seems that those that embarked on 'S' have a lower survival probabilty, however, when we look at the passenger counts it is clear that most of the 3rd class embarked there, therefore, this might be the real driver
# 
# ##### For simplicity, for the few records missing we can use the most common port 'S'
# 
# Now let's move to the next feature
# 
# ## ==== Fare ====
# 
# Let's start by the basic step of imputing the missing value with the average per class and gender

# In[ ]:


join_data.pivot_table(values='Fare',index='Pclass',columns='Sex')


# In[ ]:


# Creating a simple funtion to impute the Fare, the thresholds are according to the average Fares on the training data
def FareImp (Columns):
    # Gattering the fields:
    Fare = Columns[0]
    Pclass = Columns[1]
    Sex = Columns [2]
    # First, we only impute if the Fare value is missing
    if pd.isnull(Fare):
        if Pclass == 3 :# Third class
            if Sex == 'male':
                return 12.41
            else:
                return 15.32
        if Pclass == 2 :# Second class
            if Sex == 'male':
                return 19.90
            else:
                return 23.23
        if Pclass == 1 :# First class
            if Sex == 'male':
                return 69.88
            else:
                return 109.41
    else:
        return(Fare)


# Now, on the FARE feature there is a catch we need to get out of the way, first, if you explore the data you might have notice that a lot of people share cabins, to illustrate why this point matters let's pull one cabin:

# In[ ]:


join_data[join_data['Cabin']=='F4']


# We can see this is a family of 4, notice that they all paid the same fare, for every one passenger we could calculate their 'FamilySize' by adding the 'Parch' + 'SibSp' + 1 (Partners + Sibling/childs + themselves), let's add this variable and evaluate the fares accordingly

# In[ ]:


join_data['FamilySize'] = join_data['Parch'] + join_data['SibSp'] + 1


# In[ ]:


# Looking also at the numbers
join_data.pivot_table(values='Fare',index='Pclass',columns='FamilySize',margins=True,margins_name='Average')


# We can see clearly that the Fares increase significantly (for the same class) with the family size, considering that most families would share a cabin this makes no sense, unless, we assume that the reflected Fare is the price paid by the cabin for all members sharing it, therefore the actual Fare per passenguer was most likely the reflected value divided by the Family size.
# 
# Before moving to the next section let's take a last look at the remaining features:
# 
# ## ==== Name, Ticket, Parch, SibSp ====
# 
# As you can see on the Fare analysis with the records for the family, those traveling together also share the same 'Ticket' number, beyond that, some tickets add accronyms preceding the ticket number to document the selling agency, none of this information seem to add over dimensions we are already capturing on the 'FamilySize' variable.
# 
# Regarding the Parch and SibSp, we are summarizing those already, also on the Family size variable, therefore we should drop those features.
# 
# Finally, let's take a short view at the name, there are probably a 1000 things that could be done
# 

# In[ ]:


join_data['Name'].head(5)


# From a first look, there is a 'Title' associated to each person, it is esencially whatever precedes the '.', we could capture that and see what is the dependancy to survival

# In[ ]:


'''
The firts step in to getting the 'Title' out of a name is extract it out of the full string, we are going to create a
short function to do just that
'''
def ExtractTitle (Name):
    namewords = Name.split()
    for word in namewords:
        if '.' in word:
            return (word)


# In[ ]:


# let's test our new function, let's first locate a random name:
testname = join_data['Name'].iloc[10]
testtitle = ExtractTitle(testname)
print('Name: {}, Title: {}'.format(testname,testtitle))


# In[ ]:


# Now let's create a field for the title within our data set
join_data['Title'] = join_data['Name'].apply(ExtractTitle)
join_data['Title'].value_counts()


# We can see that there are many titles with really low occurence, also, many others that have the same meaning, let's copy and paste the list above and create a dictionary to consolidate

# In[ ]:


# Copying the list above and simplify
TitleDict = {'Mr.':'Mr',
             'Miss.':'Miss',
             'Mrs.':'Miss',
             'Master.':'Master',
             'Rev.':'Mr',
             'Dr.':'Mr',
             'Col.':'Mr',
             'Major.':'Mr',
             'Ms.':'Miss',
             'Mlle.':'Miss',
             'Sir.':'Mr',
             'Countess.':'Miss',
             'Capt.':'Mr',
             'Don.':'Mr',
             'Lady.':'Miss',
             'Jonkheer.':'Mr',
             'Dona.':'Miss',
             'Mme.':'Miss'}
# Finally, let's apply our translation to the Title field and see how it relates to survival
join_data['Title'] = join_data['Title'].map(TitleDict)


# In[ ]:


# After this steps we can merge the extraction and mapping in to a single function for future use
def ExtractAndMapTitle (Name):
    namewords = Name.split()
    for word in namewords:
        if '.' in word:
            return (TitleDict[word])


# In[ ]:


join_data.pivot_table(values='Survived',index='Title',columns=['Pclass','Sex'],
                      margins=True,aggfunc=(lambda x: x.count()))


# #### We can see that we have a 'female' Mr, that doesn't make sense, after looking for it, it turns out to be a female Dr,  let's correct this by assigning all females as 'Miss'

# In[ ]:


join_data['Title'][join_data['Sex'] == 'female'] = 'Miss'


# In[ ]:


join_data[join_data['Title'] == 'Master']['Age'].plot.hist()


# #### Also, the title master describes a male kid under 15, not sure how much extra information the title is offering over what is already available with Sex and Age, we will keep it and see if they are useful
# 
# Finally, before moving on to implement all the actions we have identified there are some opportunities to tune our performance given the particularities of the data, by these I mean in relation to the conversion of the categorical values and the scaling of the data
# 
# At some point we will need to transform all our categories in to numerical values, we can do these two ways:
#     - Assigning a 'numerical' index to each relevant category (i.e Mr=0, Miss=1 etc)
#     - Creating a dummy variable flag that expands the categorical value in to n-1 features of true/false expressed as 0 or 1 (i.e from embarkation port 'S','C' and 'Q', to feature 'Embarked-Q' (as 0 or 1) and 'Embarked-S' (as 0 and 1), there is no need for a 'Embarked-C' since the scenario where the other two are 0 already conveys that information.
#     
# The right approach will deepend mostly on the question if the 'order' of the categorical labels conveys information or not, for example for the title, if Mr=1 and Miss=2 or Mr=2 and Miss=1 makes no difference, however, for the Deck field, if B=2 and C=3 or B=0 and C=5 is significant since B is a deck above C with one degree of separation (illustrative example, I am no boat expert to know this for a fact) therefore the order and difference between them matter.
# 
# also, when we have multiple possible values, we could simple keep them all and scale the data or generalize them in to buckets or bin (i.e Age can be transformed in to 'child','kid','adult' and 'senior'. Because of these potential variances we are going to build a superset that incorporate them all and then create sub-sets than given the basic a most common practice data set (scale with dummy variables) we modified one at the time to validate what is the impact on the performance of our model.
# 
# ### Now looking at the ticket field
# 

# In[ ]:


# Let's start by isolating those passenguers that are connected through the ticket number
testDF = train_data.copy()
testDF['SharedTicket'] = testDF['Ticket'][testDF['Ticket'].duplicated(keep=False)]
print(testDF.describe(include=['O']))

testDF.pivot_table(values='SharedTicket',index='Survived',columns='Pclass',fill_value=0,
                      aggfunc=(lambda x: x.count()),margins=True,margins_name='Total')


# #### We can see that there are 134 'groups' of people that share the same ticket numbers, half of them make it, half of them don't, we know that most women and children within the same groups shared the same destiny together, so we need to identify those groups.
# 
# #### Also, from other Kernels we know that some relatives travelled together but stayed on others cabins with consecutive ticket numbers, so let's combine this criteria along with the last name to identify family groups and their survival status, there is a nuance here since that might be several different last names and ticket numbers, therefore this process will need to be recursive to ensure all relationships are captured

# In[ ]:


# First we need to capture the last name of each passenguer a create a family group ID with the ticket #
df = join_data.copy()
df['LastName'] = df['Name'].apply(lambda x: x.split(',')[0])
df['TicketP'] = df['Ticket'].apply(lambda x: x.split()[-1][:-2]) # Removing the last two digits of the ticket
df['FamilyGroup'] = df['LastName']+df['TicketP']+'XX' # replacing last two digits for XX

# Now we link together all the tickets and family names that are related
df2 = df.groupby(['Ticket'])
knowtickets = []
knowfamilies = []
TicketList={}

# filling up a dictionary that relates each ticket # to a unique family group linking all related passengers
for tick, tick_df in df2:
    # first we ensure we have not mapped already this ticket number to another group
    if tick not in knowtickets:
        # We capture all the ticket/family list groupings associated with this ticket number
        ticketlist = []
        ticketlist.append(tick)
        familylist = []
        i = 0
        while i < len(ticketlist):
            # Capture all the families associated with this particular ticket list
            families = list(df[df['Ticket']==ticketlist[i]]['FamilyGroup'].unique())
            for fam in families:
                if fam not in familylist:
                    familylist.append(fam)
                if fam not in knowfamilies:     
                    # mark the family as 'know'
                    knowfamilies.append(fam)
                    # now we recursively capture all the tickets associated with this family and add them to the lookup
                    tickets = list(df[df['FamilyGroup']== fam ]['Ticket'].unique())
                    for tick2 in tickets:
                        if tick2 not in ticketlist:
                            ticketlist.append(tick2)
            i = i+1
        # After we have captured all the associated tickets and families that are linked, then add them on the
        # dictionary and mark them as know
        for tick3 in ticketlist:
            # Mapping all the tickets to the first family name
            TicketList[tick3] = familylist[0]
            # adding the tickets to the know tickets
            knowtickets.append(tick)

# now, we create a field within the data set to document the associated group for each passenger
df['GroupID'] = df['Ticket']
df.replace({'GroupID':TicketList},  inplace=True)
# now we capture the survival profile of each group

df3 = df.groupby(['GroupID'])
FamilyList = []
for fam, fam_df in df3:
    # if there are more than 1 on the group
    if len(fam_df) != 1:
        # Now we analyze what happen with that group
        # count of members on the group
        pcount = len(fam_df) 
        # Number of females and kids on the group
        fkcount = len(fam_df[(fam_df['Age']<15)|(fam_df['Sex']=='female')]) 
        # Count of know survivers
        KnowSurvivers = int(fam_df['Survived'].sum())
        # Count of know female and kid survivers
        KnowFemKidsSurv = int(fam_df['Survived'][(fam_df['Age']<15)|(fam_df['Sex']=='female')].sum())
        # Count of know victims
        KnowVictims = fam_df['PassengerId'][fam_df['Survived'] == 0].count() 
        # count of know male adult victims
        KnowVictimsMA = fam_df['PassengerId'][(fam_df['Survived'] == 0)&(fam_df['Age'] >14)&(fam_df['Sex']=='male')].count()
        # Generating an index between -1 and 1 that indicated if the group survived or not and to what degree
        GroupSurvIndex = (KnowSurvivers-KnowVictims)/pcount
        
        # Finally, taking a page from other Kernels, creating a flag mapping the survival of the Female and kids
        if KnowFemKidsSurv > 0:
            GroupSurvFlag = 1
        elif (KnowVictims - KnowVictimsMA) > 0: # if some female or kids are know victims
            GroupSurvFlag = 0
        else:
            GroupSurvFlag = 0.5
        
        # Capturing all the dimensions
        FamilyList.append([fam,pcount,KnowSurvivers,KnowVictims,
                           KnowFemKidsSurv,KnowVictimsMA,GroupSurvFlag,GroupSurvIndex])

Family_Groups = pd.DataFrame(FamilyList,
                             columns=['GroupID','PeopleCount','K_Survivers','K_Victims','K_FemKidS',
                                      'K_MaleV','GroupSurvFlag','GroupSurvIndex'])


# #### Now, that we have an inventory of families and their survival status we use it along a funtion that receives the name and ticket and add two columns, one flag to identify if the person belonged to a family group and the second will be that family group survival status

# In[ ]:


# Creating a function maps passengers to their groups and their know survival
def FamilyGrouping (df):
    df['GroupID'] = df['Ticket']
    df.replace({'GroupID':TicketList}, inplace=True)
    # now we add the fields to document the survival profile of the groups
    df['GroupSurvIndex'] = 0
    df['GroupSize'] = 1
    df['GroupSurvFlag'] = 0.5

    GroupInfo = []
    GroupList = list(Family_Groups['GroupID'])
    for i in range(len(df)):
        familyID = df.iloc[i]['GroupID']
        if familyID in GroupList:
            SurvStatus = float(Family_Groups[Family_Groups['GroupID'] == familyID]['GroupSurvFlag'])
            SurvProb = float(Family_Groups[Family_Groups['GroupID'] == familyID]['GroupSurvIndex'])
            GroupSize = int(Family_Groups[Family_Groups['GroupID'] == familyID]['PeopleCount'])
            GroupInfo.append([SurvStatus,GroupSize,SurvProb])
        else:
            GroupInfo.append([0.5,1,0]) # Passenger traveling alone (group size of 1) with neutral values

    df[['GroupSurvFlag','GroupSize','GroupSurvIndex']] = GroupInfo
    return(df.drop(['GroupID'],axis=1)) 


# ### From the EDA we have determined that the following actions will help
#     
#     - Age:
#         - Impute the missing values considering the Class and Gender
#         - Create a new feature 'AgeTier' that groups passangers on Age ranges
#     - Cabin:
#         - Extract the 'Deck' feature
#         - Create a new 'DeckSum' feature that group low occurence decks along with unknow assigning a numerical index
#         - Transform the 'raw' 'Deck' feature in to a numerical index
#         - Drop the Cabin feature
#     - Fare (SibSp, Parch)
#         - Impute the missing values
#         - Create FamilySize variable
#         - Calculate RealFare ('FarePP') per passenger
#         - Create a new feature 'FareTier' that group fares by bins
#         - Tranform the 'FairTier' assigning a numerical index
#         - Drop the SibSp and Parch features since they have been absorved by the Family size feature
#     - Name:
#         - Extract whatever is before the '.' as the new 'Title' feature
#         - Use a dictionary to consolidate the 'Titles' in to a new 'TitleCat' feature
#         - Tranform from categorical to numerical using 'dummy' (aka on-hot)
#         - Extract Family Groups and identify if their survival is 'know'
#         - Drop the 'Name' feature
#     - Sex
#         - This categorical will need to be mapped as a numerical flag
#     - Embarqued
#         - Impute the missing value with the most common ('S')
#         - Tranform from categorical to numerical
#     - Finally, drop the ticket and PassengerID features
# 
# #### Since we will be taking this steps over several data sets, and on an actual application repeating them over new batches of data, we will consolidate them in to a subfunction, also leveraging those functions we have already created during the EDA.
# 
# ## Age ->
# 
# Implementing the function to impute the age and to group different age groups

# In[ ]:


# For the imputation strategy we are going to use the average ages we capture before per gender and class
# to achieve this objective we define a short function
def AgeImp (Columns):
    # Gattering the fields:
    Age = Columns[0]
    Pclass = Columns[1]
    Sex = Columns [2]
    # First, we only impute if the age value is missing
    if pd.isnull(Age):
        if Pclass == 3 :# Third class
            if Sex == 'male':
                return 26
            else:
                return 22.2
        if Pclass == 2 :# Second class
            if Sex == 'male':
                return 30.9
            else:
                return 27.5
        if Pclass == 1 :# First class
            if Sex == 'male':
                return 41
            else:
                return 37
    else:
        return(Age)


# In[ ]:


# the thresholds and somehow arbitrary, we could potential test the sensitivity of the results to one or more of them
def AgeMapping (Age):
    if Age < 15: # Kid
        return(0)
    elif Age < 65: # Adult
        return(0.5)
    elif Age >= 65: # Senior
        return(1)


# # 4-) Data pre-processing

# In[ ]:


'''
Data preprocessing function will receive a pandas dataframe containing titanic challenge features as an input, 
11 features are expected (dropping the target variable), this function implements the steps identified on the 
Feature engineering

'''
def DataPreprocessing(FeatureDF):

  
    # == AGE ==
    # Imputing the missing values on the Age
    FeatureDF['Age'] = FeatureDF[['Age','Pclass','Sex']].apply(AgeImp,axis=1)
    # Mapping the Age groups
    FeatureDF['AgeTier'] = FeatureDF['Age'].apply(AgeMapping)
    
    # == CABIN ===
    # Extracting the 'Deck' Feature
    FeatureDF['Deck'] = FeatureDF['Cabin'].apply(lambda x: str(x)[0])
    # Using the dictionary we created during the EDA to consolidate uncommon flags and map all to numeric indexes
    FeatureDF.replace({'Deck':DeckDict},inplace=True)
    
    # == FARE ==
    # Imputing the missing values
    FeatureDF['Fare'] = FeatureDF[['Fare','Pclass','Sex']].apply(FareImp,axis=1)
    # Creating the Family Size feature
    FeatureDF['FamilySize'] = FeatureDF['Parch'] + FeatureDF['SibSp'] + 1
    # Calculating the Real Fare per passenger
    FeatureDF['RealFare'] = FeatureDF['Fare']/FeatureDF['FamilySize']
    # Group Fare by tiers (using the thresholds of the quartiles)
    # FeatureDF['FareTiers'] = pd.qcut(FeatureDF['RealFare'], q=4,labels=False)

    # == NAME ==
    # Capturing the Title and mapping it in to a subset of categories
    FeatureDF['Title'] = FeatureDF['Name'].apply(ExtractAndMapTitle)
    # Creating the Family groups survival flags
    FeatureDF = FamilyGrouping(FeatureDF.copy())
    # Correcting for female doctors (just 1 on the training data set)
    FeatureDF['Title'][(FeatureDF['Sex'] == 'female') & (FeatureDF['Title'] == 'Mr')] = 'Miss'
    # Transforming from categorical to numerical
    FeatureDF = pd.get_dummies(FeatureDF,columns=['Title'],drop_first=True)
    
    # == SEX ===
    # Transforming from categorical to numerical
    sex_d = {'male':0,'female':1}
    FeatureDF.replace({'Sex':sex_d}, inplace=True)
    
    # == Embarked ==
    # Impute missing values with 'S'
    # FeatureDF['Embarked'].fillna(FeatureDF['Embarked'].mode()[0], inplace = True)
    # Transform from categorical to numerical
    # FeatureDF = pd.get_dummies(FeatureDF,columns=['Embarked'],drop_first=True)

    # Dropping all the unused unused features
    FeatureDF.drop(['PassengerId','Ticket','Cabin','Name','Parch','SibSp','Embarked'],
                   axis=1,inplace=True)
    return (pd.DataFrame(scaler.fit_transform(FeatureDF),columns=FeatureDF.columns))


# # 5-) Model selection and testing
# 
# We start by using our pre-processing utility to obtain a feature data set to train our models

# In[ ]:


TD_X = DataPreprocessing(train_data.drop(['Survived'],axis=1).copy())
TD_y = train_data['Survived']
K_X = DataPreprocessing(test_data.copy())

# Getting the data splits for training and testing
X_train, X_test, y_train, y_test = train_test_split(TD_X, TD_y, test_size=0.33,random_state=29)


# In[ ]:


fig = plt.figure(figsize=(20,10))
sns.heatmap(TD_X.corr())


# With a few exceptions, most of our features don't seem to be really highly correlated
# 
# ### Now we are ready to star testing models
#     
# ## Trying some alternatives for the models ================================

# In[ ]:


'''
Creating a function to run a grid search over several alternatives
'''
def testmodels(RunDes):
    # Setting up the support parameters for all runs
    num_folds = 5
    seed = 42
    lastRun = len(models) # We capture how many models we have tried already so we don't re-train them again

    #$$$$$$$$$ Defining all the grid seaches for all approaches to be tried $$$$$$$$$$$$$$$

    ################### For Random Forest
    parameters_RD = {"n_estimators":list(range(80,401,20)),"max_depth":[2, 3, 4, 5],
                     "min_samples_split":[2, 3, 10],"min_samples_leaf":[1,3,8]}
    rfc = RandomForestClassifier()
    models.append(['RandomForest',GridSearchCV(estimator=rfc, param_grid=parameters_RD,scoring='roc_auc',
                                               cv=num_folds,n_jobs=-1)])
    ################### For SVC
    parameters_svm = {"C":[2.0, 2.5, 3.0],"max_iter":[250, 500, 1000]}
    svm = SVC()
    models.append(['SupportVectorM',GridSearchCV(estimator=svm, param_grid=parameters_svm,scoring='roc_auc',
                                               cv=num_folds,n_jobs=-1)])              
    ################### For XGBoost
    XGpd = XGBClassifier()
    parameters_xg = {'min_child_weight': [1, 5, 10],'gamma': [0.5, 1, 1.5, 2, 5],'subsample': [0.6, 0.8, 1.0],
                     'colsample_bytree': [0.6, 0.8, 1.0],'max_depth': [3, 4, 5]}
    models.append(['XG Boost',GridSearchCV(estimator=XGpd, param_grid=parameters_xg,scoring='roc_auc',
                                               cv=num_folds,n_jobs=-1)])
    ################### For KNN
    parameters_knn = {"n_neighbors":[6,7,8,9,10,11,12,14,16,18,20,22]}
    knn = KNeighborsClassifier()
    models.append(['KNN',GridSearchCV(estimator=knn, param_grid=parameters_knn,scoring='roc_auc',
                                      cv=num_folds,n_jobs=-1)])

    ########################################################################################
    ################## Running all the models and capturing the results ####################
    ########################################################################################
    i = 0
    for model, grid in models:
        # skip on the models already trained
        if ( i < lastRun):
            i = i+1
        else:
            # first we train our grid
            grid.fit(X_train,y_train)
            # Now we test on the holdout data
            predTrainData = grid.predict(X_test)
            # Capturing the trained model and its results
            predTestData = grid.predict(K_X)       
            ModelResults.append([model,grid.best_score_, precision_score(y_test,predTrainData),
                                 accuracy_score(y_test,predTrainData),RunDes,predTestData,Field2Drop])
            i = i+1


# In[ ]:


# Creating the overall results containers
ModelResults = []
models = []


# In[ ]:


# Running our first attemp with all the features

# Creating a 'tag' for the run, in case some parameters or inputs are optimized to have the reference of performance
# of the model after each change
RunDes = 'Baseline - All Features'
Field2Drop = []
testmodels(RunDes)

################## now we format and print our results for visualization
ResultsDF = pd.DataFrame(ModelResults,columns=['Model','Train Accu','Precision',
                              'Accuracy','Run','Predictions','DroppedFields'])
ResultsDF.sort_values(by=['Accuracy'],ascending=False)


# In[ ]:


########################################################################################
## Using Cross-validation to assess variability of the best model over unseen data  ####
########################################################################################
bestM = ResultsDF['Accuracy'].idxmax()

all_accuracies = cross_val_score(estimator=models[bestM][1], X=X_test, y=y_test, cv=3) 
print(all_accuracies)
print('---------------------')
print('Average accuracy = {}, deviation = {}'.format(all_accuracies.mean(),all_accuracies.std())) 


# In[ ]:


# Create the Recurrent Feature Elimination (RFE) object and compute a cross-validated score.
bestM = ResultsDF['Accuracy'].idxmax()
rfecv = RFECV(estimator=models[0][1].best_estimator_, step=1, cv=4,scoring='accuracy')
rfecv.fit(X_test, y_test)
print("Optimal number of features : %d" % rfecv.n_features_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
print(pd.DataFrame(rfecv.support_,index=TD_X.columns))
print(pd.DataFrame(rfecv.ranking_,index=TD_X.columns))


# In[ ]:


# TDropping the features that the RFE identified as not adding to much value
TD_X = DataPreprocessing(train_data.drop(['Survived'],axis=1).copy())
TD_y = train_data['Survived']
Field2Drop = ['Pclass', 'Age', 'Fare', 'AgeTier', 'Deck', 'FamilySize', 'GroupSize']
TD_X.drop( Field2Drop,axis=1,inplace=True)
# Getting the data splits again for training and testing
X_train, X_test, y_train, y_test = train_test_split(TD_X, TD_y, test_size=0.2,random_state=42)
# PReparing the test data set too
K_X = DataPreprocessing(test_data.copy())
K_X.drop( Field2Drop,axis=1,inplace=True)

# Creating a 'tag' for the run, in case some parameters or inputs are optimized to have the reference of performance
# of the model after each change
RunDes = 'Removing some features'
testmodels(RunDes)

################## now we format and print our results for visualization
ResultsDF = pd.DataFrame(ModelResults,columns=['Model','Train Accu','Precision',
                              'Accuracy','Run','Predictions','DroppedFields'])
ResultsDF.sort_values(by=['Accuracy'],ascending=False)


# ### Trying a neural network

# In[ ]:


'''
Creating a funtion that builds, trains and test a neural network according to a set of parameters
'''
def NeuralNetwork (X_train, y_train, X_test, parameters):
    # We start by capturing the pass parameters
    batch = parameters[0]
    epochs = parameters[1]
    hiddenU = parameters[2]
    
    # Capturing the feautres in Tensors
    features = []
    for column in X_train.columns:
        features.append(tf.feature_column.numeric_column(column))

    # Defining the input function and training the classifier
    input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=batch,num_epochs=epochs,shuffle=True)
    classifier = tf.estimator.DNNClassifier(hidden_units= hiddenU,n_classes=2,feature_columns=features)
    classifier.train(input_fn=input_func,steps=200)

    # testing
    prediction_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
    predictions = list(classifier.predict(input_fn=prediction_func))

    # Capturing the results
    ts_final = []
    for pred in predictions:
        ts_final.append(pred['class_ids'][0])
    return(ts_final)


# In[ ]:


# Trying cleaning up some features that don't seem to add too much value
TD_X = DataPreprocessing(train_data.drop(['Survived'],axis=1).copy())
TD_y = train_data['Survived']
Field2Drop = ['Pclass', 'Age', 'Fare', 'AgeTier', 'Deck', 'FamilySize', 'GroupSize']
TD_X.drop( Field2Drop,axis=1,inplace=True)

# Preparing the test data set too
K_X = DataPreprocessing(test_data.copy())
K_X.drop( Field2Drop,axis=1,inplace=True)
# Getting the data splits again for training and testing
X_train, X_test, y_train, y_test = train_test_split(TD_X, TD_y, test_size=0.33,random_state=42)


# In[ ]:


# One time prediction with the best setting for the output to Kaggle
TestD = DataPreprocessing(test_data.copy())
NNparam = [40,6,[20,35,40]]
NNTPred = NeuralNetwork(X_train,y_train,X_test,NNparam)
NNRPred = NeuralNetwork(TD_X,TD_y,K_X,NNparam)
NNTaccu = accuracy_score(y_test,NNTPred)
NNTprec = precision_score(y_test,NNTPred)
print('##'*30)
print('Neural Network, precision: {}, Accuracy: {}'.format(NNTprec,NNTaccu))


# The introduction of a Neural Network seem to have added a little more precision we can try to summit both results and see what happens

# In[ ]:


# writing the output file
result_df = pd.DataFrame(list(NNRPred),index=test_data['PassengerId'],columns=['Survived'])
result_df.to_csv('TitanicPredictions-NN.csv')


# # 6-) Output generation

# In[ ]:


########################################################################################
###################### Generating the final results for Kaggle #########################
########################################################################################
# Using the best model we got
bestM = ResultsDF['Accuracy'].idxmax()
Field2Drop = ModelResults[bestM][6]
# Generating the prediction with the best possible model
Test_X = DataPreprocessing(test_data.copy())
Test_X.drop( Field2Drop,axis=1,inplace=True)
FinalPred = models[bestM][1].predict(Test_X)

# writing the output file
result_df = pd.DataFrame(list(FinalPred),index=test_data['PassengerId'],columns=['Survived'])
result_df.to_csv('TitanicPredictions-Final.csv')


# ### This notebook contain just the tip of the iceberg, I personally attempted many other approaches and iterations through different paths that didn't provide better results, also, as mentioned at nauseum on every other kernels and forums, the test data for this particular challenge don't seem to be that consistent with the training set, therefore, as good as the approach could appear on cross-validation it didn't performed better than 0.81 on multiple attemps.
# 
# If you found this notebook helpful or you just liked it, some upvotes would be very much appreciated - That will keep me motivated :)

# In[ ]:




