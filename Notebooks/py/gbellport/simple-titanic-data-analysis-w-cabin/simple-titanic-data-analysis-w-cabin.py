#!/usr/bin/env python
# coding: utf-8

# # _Simple Titanic Data Analysis & Processing (with Cabin analysis)_
# Begginer analysis on Titanic dataset, trying to tackle the Cabin analysis as well through research and imputation. I am currently a begginer too and this is my first Kernel. Please be sure to leave any comments or opinions on whether my analysis is correct or if something could be polished. Also, feel free to use the analysis if you think it properly done (with some credit if you happen to publish a Kernel). Also let me know if you get any good results using my analysis, I am always open to discussion!
# 

# # <font color='maroon'>SECTION 1: Data</font>
# 
# **_References:_**
# 
# [1] General Titanic info: https://www.encyclopedia-titanica.org
# 
# [2] Titanic cabin info: https://www.encyclopedia-titanica.org/cabins.html
# 
# [3] Regular expression function: https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

# # General Libraries

# In[ ]:


# Filter out warnings
import warnings
warnings.filterwarnings('ignore')

# For dataframe displaying purposes
from IPython.display import display

# Data analysis and processing
import pandas as pd
import numpy as np
import re

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# # Import Data

# Lets start by importing the data and making copies of it. The processing will be done on the copies and we will have the originals for comparisson:

# In[ ]:


# Original, unprocessed data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


# Save for submission and training respectively
passenger_id = test['PassengerId']
target = train['Survived']


# In[ ]:


# Drop PassengerId on both copies
for df in [train,test]:
    df.drop('PassengerId',axis=1,inplace=True)


# Finally,  concatenate the train and test data into one big dataframe for uniformity:

# In[ ]:


all_data = pd.concat([train.drop('Survived',axis=1),test]).reset_index(drop=True)


# In[ ]:


# Save indexes for splitting all_data later on
ntrain = train.shape[0]
ntest = test.shape[0]


# #  Exploratory Analysis & Feature Engineering

# Lets start by creating a feature for family size to thin out the amount of columns. It more of a choice to include or exclude the passenger in this feature. It is also advisable drop **SibSip** and/or **Parch** to avoid the features being too correlated:

# In[ ]:


# Create FamSize feature and drop 
all_data['FamSize'] = 1 + all_data['SibSp'] + all_data['Parch']
all_data.drop(['SibSp','Parch'],axis=1,inplace=True)


# ## <font color='blue'>Missing Values</font>
# First, lets explore what features contain missing data (NaN or Null) so we can then work on each feature individually:

# In[ ]:


total_miss = all_data.isnull().sum()
percent_miss = (total_miss/all_data.isnull().count()*100)

# Creating dataframe from dictionary
missing_data = pd.DataFrame({'Total missing':total_miss,'% missing':percent_miss})

missing_data.sort_values(by='Total missing',ascending=False).head()


# ## <font color='blue'>Embarked</font>

# In[ ]:


all_data[all_data['Embarked'].isnull()]


# In[ ]:


sns.boxplot(x='Pclass',y='Fare',
            data=all_data[(all_data['Pclass']==1)&(all_data['Fare']<=200)],
            hue='Embarked')


# Theres very little to work here with, and the reference guide [1] gives scarce information on these two passengers. We could either drop these two entries or fill them with an estimate. Given the boxplot above, the safest choice would be use *C* for **Embarked**:

# In[ ]:


_ = all_data.set_value(61,'Embarked',value='C')
_ = all_data.set_value(829,'Embarked',value='C')


# ## <font color='blue'>Fare</font>
# Now lets explore fare, but lets also look at those passengers whose Fare is 0 and try to understand why this might be the case:

# ### Missing Fares 

# In[ ]:


display(all_data[all_data['Fare'].isnull()])
display(all_data[all_data['Fare']==0])


# It seems as if these entries were people who either worked in the Titanic or their ticket was paid for. Finding information on the passenger with the missing from [1] shows that:
# 
# > (...) Philadelphia's westbound voyage was cancelled, with Storey and several other shipmates; Andrew Shannon [Lionel Leonard], August Johnson, William Henry Törnquist, Alfred Carver and William Cahoone Johnson) forced to travel aboard Titanic as passengers.
# 
# The passenger whose **Fare** is not recorded happens to work with some of the passengers whose ticket was paid for:

# In[ ]:


_ = all_data.set_value(1043,'Fare',value=0)


# ### Discretizing Fare
# Since this is categorical problem the algorithms that will be used further on will benefit from the data being categorical. For this reason, we will convert **Fare** , which currently continuous, to discrete data by using `pd.qcut()`. **Fare** will benefit from the quantile discretization since its range is large. 
# 
# We will use 5 groups to categorize the data, starting from 1: 

# In[ ]:


splits = 5


# In[ ]:


# Intervals for discretizing fare values
for i in range(splits):
    print(f'Group {i+1}:',pd.qcut(all_data['Fare'],splits).sort_values().unique()[i])


# In[ ]:


def discretize_fare(val):
    
    fare_group = pd.qcut(all_data['Fare'],splits).sort_values().unique()
    
    for i in range(splits):
        
        if val in fare_group[i]:
            return i+1
        elif np.isnan(val):
            return val


# In[ ]:


all_data['Fare'] = all_data['Fare'].apply(discretize_fare)


# **Note:** For some reason, this process is not considering the 4 values with the highest fare. The imputation here will be done manually and with hopes that it will be fixed later. Since they have paid the most for a ticket they belong to class 5:

# In[ ]:


all_data['Fare'] = all_data['Fare'].fillna(5).astype(int)


# ## <font color='blue'>Age</font>

# ### Discretizing Age
# The same discretization process will be done for **Age**. In this case however, it is better to use `pd.cut()`, since the ages are distributed somewhat normally and it will benefit from being evenly spaced. Also, in this case we will fill empty values with 0 indicating they don't yet belong to any age group:

# In[ ]:


# Intervals for discretizing each age value
for i in range(splits):
    print(f'Group {i+1}:',pd.cut(all_data['Age'].dropna(), splits).unique()[i])


# In[ ]:


def discretize_age(val):
    
    age_group = pd.cut(all_data['Age'],splits).sort_values().unique()
    
    for i in range(splits):
        
        if val in age_group[i]:
            return i+1
        elif np.isnan(val):
            return 0


# In[ ]:


all_data['Age'] = all_data['Age'].apply(discretize_age).astype(int)


# ### Titles
# Since the titles of each passenger also provide information useful to determine the age of a passenger, lets make a function that outputs the title of each passenger (I got this function from [3], I had no idea how to use regular expressions on python. They have a great kernel that helped me a lot when I first started doing stacking so definitely check it out!)

# In[ ]:


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


# The function above uses the `re` python library for regular expressions. It basically searches, within the name of each passenger, a combination of letters that start with an upper case and followed by lower case letters and finally a dot (.), which is the format of a title. We will create a feature out of these titles:

# In[ ]:


all_data['Title'] = all_data['Name'].apply(get_title)


# In[ ]:


# Looking at unique titles
all_data['Title'].unique()


# Harnessing the power of Google, we will try to understand some of these lesser known titles:
# 
# - **Master:** boys and young men 
# - **Don/Donna/Lady/Sir/Countess/Jonkheer:** royal or high profile title
# - **Rev:** reverend, priest
# - **Mme/Ms:** woman of unknown marrital status (usually unmarried)
# - **Major/Col/Capt:** military title
# - **Mlle:** married woman
# 
# With this we can make the following replacements:

# In[ ]:


all_data['Title'] = all_data['Title'].replace(['Ms','Mlle'],'Miss')
all_data['Title'] = all_data['Title'].replace('Mme','Mrs')
all_data['Title'] = all_data['Title'].replace(['Don','Dona','Lady','Sir',
                                                 'Countess','Jonkheer'],'Royal')
all_data['Title'] = all_data['Title'].replace(['Rev','Major','Col','Capt','Dr'],'Other')


# ### Missing Ages
# Lets now see how **Title** relates to **Age**:

# In[ ]:


plt.figure(figsize=(10,4))
sns.stripplot(x='Title',y='Age',data=all_data[all_data['Age']!=0],
              hue='Pclass',dodge=True)
plt.legend(loc=1)


# With the information provided by this plot we can now fill the missing age group values. We have to options here, to either fill with the mean for each (very) specific category, or filling randomly. Lets try filling with the mean:

# In[ ]:


def impute_age(row):
    
    # Features from row
    pclass = row['Pclass']
    title = row['Title']
    age = row['Age']
    
    if age == 0:
        return int(round(all_data.loc[(all_data['Age']!=0)&
                                      (all_data['Pclass']==pclass)&
                                      (all_data['Title']==title)]['Age'].mean(),1))
    else:
        return age


# In[ ]:


all_data['Age'] = all_data.apply(impute_age,axis=1)


# ## <font color='blue'>Cabin/Deck</font>
# Finally the big boss. Lets first change **Cabin**  for **Deck**, and fill the empty values with a placeholder 'N'. This will be easier to analyze. Note that also each deck could be split into parts but that would make the task much less simple.

# In[ ]:


_ = all_data.rename({'Cabin':'Deck'},axis=1,inplace=True)


# In[ ]:


all_data['Deck'] = all_data['Deck'].fillna('N')


# Now, grab the first letter of each cabin code, which is the deck level where the cabin is located:

# In[ ]:


def cabin_to_deck(row):
    return row['Deck'][0]


# In[ ]:


all_data['Deck'] = all_data.apply(cabin_to_deck,axis=1)


# ### Missing Deck based on Ticket
# Looking at the Ticket column we can see that there are some passengers that share the same Ticket, which means that they bought the tickets together and were likely sharing a cabin, or at least were in the same **Deck**. The code below does the following:
# - Counts how many times a ticket shows up in the data sets
# - Finds out whether at least one of the passengers with that **Ticket** has an unidentified **Deck**
# - Appends **Ticket** to list given some constraints

# In[ ]:


ticket_list = []
for ticket_id in list(all_data['Ticket'].unique()):
    
    count = all_data[all_data['Ticket']==ticket_id].count()[0]
    decks = all_data[all_data['Ticket']==ticket_id]['Deck']
    empty_decks = (decks=='N').sum()
    
    if (count > 1) and (empty_decks > 0) and (empty_decks < len(decks)):
        ticket_list.append(ticket_id)

print(ticket_list)


# So now that we have these ticket IDs, we can explore the dataset and see if we can fill any **Deck** values individually, that is with the help of [2] (cabin reference)

# In[ ]:


# Show dataframes with the previous specifications
for ticket in ticket_list:
    display(all_data[all_data['Ticket']==ticket])


# **IMPORTANT NOTE:** All decks imputed/filled below were not pulled out of thin air. I went through the cabin reference [2] and found all this information which I then used to fill the missing values. Some assumptions were also made, such as families sharing the same **Deck**

# In[ ]:


# ticket ID, information

# 2668, 2 siblings (sharing with mother)
_ = all_data.set_value(533,'Deck',value=all_data.loc[128]['Deck'])
_ = all_data.set_value(1308,'Deck',value=all_data.loc[128]['Deck'])

# PC 17755, maid to Mrs. Cardeza
_ = all_data.set_value(258,'Deck',all_data.loc[679]['Deck'])

# PC 17760, manservant to Mrs White 
_ = all_data.set_value(373,'Deck',value='C')

# 19877, maid to Mrs Cavendish
_ = all_data.set_value(290,'Deck',value=all_data.loc[741]['Deck'])

# 113781, maid and nurse to the Allisons
_ = all_data.set_value(708,'Deck',value=all_data.loc[297]['Deck'])
_ = all_data.set_value(1032,'Deck',value=all_data.loc[297]['Deck'])

# 17421, maid to Mrs Thayer
_ = all_data.set_value(306,'Deck',value='C')

# PC 17608, governess (teacher) to Master Ryerson
_ = all_data.set_value(1266,'Deck',value=all_data.loc[1033]['Deck'])

# 36928, parents (sharing with daughters)
_ = all_data.set_value(856,'Deck',value=all_data.loc[318]['Deck'])
_ = all_data.set_value(1108,'Deck',value=all_data.loc[318]['Deck'])

# PC 17757, maid and manservant to the Astors
_ = all_data.set_value(380,'Deck',value='C')
_ = all_data.set_value(557,'Deck',value='C')

# PC 17761, maid to Mrs Douglas, occupied room with another maid
_ = all_data.set_value(537,'Deck',value='C')

# 24160, maid to Mrs. Robert, testimony that she was on deck E
_ = all_data.set_value(1215,'Deck',value='E')

# S.O./P.P. 3, very little information, will assume on deck E with Mrs. Mack
_ = all_data.set_value(841,'Deck',value=all_data.loc[772]['Deck'])


# ### Missing Deck based on Pclass
# Lets check were out passengers were situated based on Pclass

# In[ ]:


fig,ax = plt.subplots(1,2,figsize = (10,4))
plt.tight_layout(w_pad=2)
ax = ax.ravel()

sns.countplot(x='Pclass',data=all_data[all_data['Deck']!='N'],hue='Deck',ax=ax[0])
ax[0].legend(loc=1)
ax[0].set_title('Pclass count for known Deck')
sns.countplot(x='Pclass',data=all_data[all_data['Deck']=='N'],hue='Deck',ax=ax[1])
ax[1].set_title('Pclass count for unkown Deck')


# Accoring to our reference [2], 3rd class passengers were on the lower levels of the Titanic, mainly E,F, and G decks. Let's start by making lists of the possible decks each passenger might have belonged to based on their Pclass
# 
# **Note:** The reference also states that the T deck was unique, and only one person was housed there.

# In[ ]:


decks_by_class = [[],[],[]]
for i in range(3):
    decks_by_class[i] = list(all_data[all_data['Pclass']==i+1]['Deck'].unique())
    print(f'Pclass = {i+1} decks:',decks_by_class[i])


# In[ ]:


# Removing null ('N') entries and single 'T' cabin
for i in range(3):
    if 'N' in decks_by_class[i]:
        decks_by_class[i].remove('N')
    if 'T' in decks_by_class[i]:
        decks_by_class[i].remove('T')


# Lets also assign weights so when we select randomly from each list the selections are properly distributed:
# 
# **Note:** Since we removed *T* from the data (which belonged to **Pclass** = 1), we need to account for it only for that class, if not the probability will not add to 1!

# In[ ]:


weights_by_class = [[],[],[]]

for i,deck_list in enumerate(decks_by_class):
    for deck in deck_list:
        if i == 0:
            class_total = all_data[(all_data['Deck']!='N')&(all_data['Pclass']==i+1)].count()[0]-1
        else:
            class_total = all_data[(all_data['Deck']!='N')&(all_data['Pclass']==i+1)].count()[0]
        deck_total = all_data[(all_data['Deck']==deck)&(all_data['Pclass']==i+1)].count()[0]
        weights_by_class[i].append(deck_total/class_total)
    print(f'Pclass = {i+1} weights:',np.round(weights_by_class[i],3))


# As mentioned before calculationf the weights, we will impute the remaining **Deck** values semi-randomly as it is hard to determine in which deck each passenger was based on statistics. We will also make sure that families are housed in the same deck by analyzing their **Ticket**. So if we impute a **Deck** value randomly to some passenger we have to make sure their family is placed in the same deck:

# In[ ]:


# Store tickets that were already looped with cabin position
ticket_dict = {}


# In[ ]:


def impute_deck(row):
    
    ticket = row['Ticket']
    deck = row['Deck']
    pclass = row['Pclass']
    
    if (deck == 'N') and (ticket not in ticket_dict):
        
        if pclass == 1:
            deck = list(np.random.choice(decks_by_class[0],size=1,
                                         p=weights_by_class[0]))[0]
        elif pclass ==2:
            deck = list(np.random.choice(decks_by_class[1],size=1,
                                         p=weights_by_class[1]))[0]
        elif pclass ==3:
            deck = list(np.random.choice(decks_by_class[2],size=1,
                                         p=weights_by_class[2]))[0]
        
        ticket_dict[ticket] = deck
        
    elif (deck == 'N') and (ticket in ticket_dict):
        deck = ticket_dict[ticket]
    
    return deck


# In[ ]:


all_data['Deck'] = all_data.apply(impute_deck,axis=1)


# ## <font color='blue'>Filtering Features</font>
# Now that most of the features have been used to fill missing values on other features, it is time to get rid of those columns that don't provide any useful information anymore:

# In[ ]:


all_data.head(1)


# In[ ]:


all_data = all_data.drop(['Name','Ticket','Title'],axis=1)


# ## <font color='blue'>Label Encoding</font>
# Finally, we need all features to be numerical. For this reason we will assign a numerical integer value to each unique value for the string columns by using pythons `.map()`

# In[ ]:


all_data['Deck'] = all_data['Deck'].map({'F':0,'C':1,'E':2,
                                             'G':3,'D':4,'A':5,
                                             'B':6,'T':7}).astype(int)


# In[ ]:


all_data['Embarked'] = all_data['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)


# In[ ]:


all_data['Sex'] = all_data['Sex'].map( {'female':0,'male':1}).astype(int)


# ## <font color='blue'>_Alone_ Feature</font>
# Many Kernels suggest using a feature to determine whether a passenger is alone or not. After many hours of experimenting, I finally caved in and created this feature, which indeed has provided me with the the best score I have achieved, so lets add it to our data:

# In[ ]:


all_data['Alone'] = 0
all_data.loc[all_data['FamSize']==1,'Alone'] = 1


# ## <font color='blue'>Checking</font>
# Lets make sure everything still has the same shapes and no data has been lost in the process of well, processing:

# In[ ]:


all_data.head()


# In[ ]:


all_data.shape[0] == ntrain + ntest


# # <font color='maroon'>SECTION 2: Machine Learning</font>

# In[ ]:


# Cross-validation
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

# Estimators
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.svm import SVC


# ## <font color='blue'>CV & Metrics</font>
# We need a few functions ways to judge our fit, these are the ones I used while trying to find the best result:

# In[ ]:


# These are for using with CV while testing parameters
def rmse_cv(model,train): 
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train)
    return np.sqrt(-cross_val_score(model,train,target,scoring='neg_mean_squared_error',cv=kf))

def logloss_cv(model,train):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train)
    return -cross_val_score(model,train,target,scoring='neg_log_loss',cv=kf)

def accuracy_cv(model,train):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)
    return cross_val_score(model,train,target,scoring='accuracy',cv=kf)


# In[ ]:


# These are for using with predictions and target
def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

def accuracy(y_true,y_pred):
    return accuracy_score(y_true,y_pred)


# ## <font color='blue'>Parameters</font>
# After using `GridSearchCV`, these are the parameters that (so far) have worked the best for the data. Also, I did extensive research using other estimators such as XGB and LGB classifiers, but to no luck (previously my best score was achieved using only LGB)

# In[ ]:


rf = RandomForestClassifier(n_estimators=700,max_depth=4,
                            min_samples_leaf=1,n_jobs=-1,
                            warm_start=True,
                            random_state=42)

et = ExtraTreesClassifier(n_estimators=550,max_depth=4,
                          min_samples_leaf=1,n_jobs=-1,
                          random_state=42)

ada = AdaBoostClassifier(n_estimators=550,learning_rate=0.001,
                         random_state=42)

svc = SVC(C=2,probability=True,random_state=42)


# ## <font color='blue'>Creating Stacking Class</font>
# This is the class/helper I use for stacking. It is inspired (and some of the code borrowed) from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard, which is a Kernel from another competition. I learned a LOT about stacking from that Kernel so do check it out if you're a beginner:

# In[ ]:


# Class is inheriting methods from these classes
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
# Metrics for measuring our fit
from sklearn.metrics import mean_squared_error, accuracy_score


# In[ ]:


class StackerLvl1(BaseEstimator, ClassifierMixin, TransformerMixin):
    
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    
    # Get OOF predictions
    def oof_pred(self, X, y):
        
        self.base_models_ = [list() for x in self.base_models]
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            
            for train_index, test_index in kfold.split(X, y):
                
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X.loc[train_index], y.loc[train_index])
                y_pred = instance.predict(X.loc[test_index])
                out_of_fold_predictions[test_index, i] = y_pred
            
        return out_of_fold_predictions

    # Fit meta model using OOF predictions
    def fit(self, X, y):
        
        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(self.oof_pred(X,y), y)
        return self
    
    # Predict off of meta features using meta model
    def predict(self, test):
        self.meta_features_ = np.column_stack([
            np.column_stack([model.predict(test) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(self.meta_features_)


# ## <font color='blue'>Predictions</font>
# Lets first split the data into a train and test set, using the indices we saved in the previous section (`ntrain`). Remember we also saved the target feature and the passenger ID for submission

# In[ ]:


train = all_data[:ntrain]
test = all_data[ntrain:]


# In[ ]:


# Create our stack object and fit it
stack_model  = StackerLvl1(base_models=(rf,et,svc),meta_model = ada)
stack_model.fit(train,target)

# Get metrics from cv (note that we are fitting to train data and comparing to target!)
print('Accuracy:',accuracy(stack_model.predict(train),target)) 
print('RMSE:',rmse(stack_model.predict(train),target)) 


# When we are satisfied with the scores, we predict off of the test set and get our final predictions:

# In[ ]:


stack_model_pred = stack_model.predict(test)


# ## <font color='blue'>Submission</font>

# In[ ]:


sub = pd.DataFrame({'PassengerId':passenger_id, 
                    'Survived':stack_model_pred})
sub.to_csv('submission.csv',index=False)

