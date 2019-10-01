#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 26/8/18 - Revisied for better readability


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')

# Plotting parameters
plt.rcParams['figure.figsize'] = (16,9)
sns.set_palette('gist_earth')


# ## Brief Introduction
# First, the 2 datasets are imported. The training one contains survival value while the testing one does not.  <br>
# The aim here is to predict the survival of passengers in the test dataset using the features given.

# In[ ]:


# Read datasets from csv
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# Merge the 2 dataframes for EDA and feature engineeraing
full = pd.concat([df_train, df_test], axis = 0, sort=True)

# Set PassengerId as Index
full.set_index('PassengerId', drop = False, inplace=True)
train = full[:891]

# Display Data
display(full.head(3))
print(f"Dataset contains {full.shape[0]} records, with {full.shape[1]} variables.")


# ## Missing Values
# Missing Values are found on Age, Cabin and Fare. <br>
# As too many values are missing in Cabin, this features may not be useful for predicting survival;
# Age can be an important factor and could be inferred from other features, e.g., Title, Parch and the families the passengers was belong to.

# In[ ]:


# Identify Missing Values
nan = full.isnull().sum()
idx_nan = nan.mask(nan==0).dropna().index

sns.heatmap(full[idx_nan].transpose().isnull(), cmap = 'binary', cbar = False)
nan[idx_nan].drop('Survived').sort_values()


# ## Data Cleaning
# - simplifying the values of categorical variables
# - the first few letters of Ticket and Cabin may tell us the location of the passengers

# ### Simplifying the Ticket
# Tickets provide information on where the passengers are located on ship, which may be vital for survival.
# Here, I group the tickets by their first few letters.

# In[ ]:


np.sort(full['Ticket'].unique())
# note: can further explore tcket - cabin - Pclass relationship here


# In[ ]:


def parse_ticket(str1):
    """
    Function to parse the Letter part of the Ticket code
    """
    m = re.search(r'(.*)(\s\d|\s\d{4,7}$)',str1)
    s = re.search(r'[A-Z]+',str1)
    if m: # removing non alphanumeric characters and binding the numbers and letters before the space
        str2 = m.group(1)
        n =re.search(r'([A-Z]+)[^A-Z0-9]*([A-Z]+)*[^A-Z0-9]*([A-Z0-9]*)[^A-Z]*([A-Z]*)*',str2)
        new_str = ''
        if n:    
            if n.group(1):
                new_str+=n.group(1)
                if n.group(2) or n.group(3):
                    if n.group(2):
                        new_str+=n.group(2)
                    if n.group(3):
                        new_str+=n.group(3)
                        if n.group(4):
                            new_str+=n.group(4)
                            if n.group(5):
                                new_str+=m.group(5)
    elif s:
        new_str = s.group(0) # Ticket with letters only
    else:
        new_str = 'XXX' #Ticket with only numercial values
    return new_str

full['Ticket_short'] = full.Ticket.map(parse_ticket)


# ### Cabin

# In[ ]:


def parse_Cabin (cabin):
    if type(cabin) == str:
        m = re.search(r'([A-Z])+', cabin)
        return m.group(1)
    else:
        return 'X'
        
full['Cabin_short'] = full['Cabin'].map(parse_Cabin)


# ###  Fare
# Fare value was found to be distorted as the Fare feature in original dataset calculates the total amount paid for one single ticket, i.e., no. of person * base rate of ticket. To get a more accurate fare paid by individual value, the fare is divided by the no. of person holding that ticket.

# In[ ]:


# Fare Adjustment
fare_original = full['Fare'].copy()

dict_ticket_size = dict(full.groupby('Ticket').Fare.count())
ticket_size = full['Ticket'].map(dict_ticket_size)
full['Fare'] = full.Fare/ticket_size


# Plot Fare Adjustment
fig, (ax0, ax1) = plt.subplots(2)
ax0.hist(fare_original.dropna(), bins=80);
ax0.set_xlabel('Fare(Original)')

ax1.hist(full['Fare'].dropna(), bins=80);
ax1.set_xlabel('Fare (Corrected)');


# After adjustment, the range has reduced from 0 - 510 to 0 - 130 and the 3 Pclass are more clearly shown by the 3 peaks of Adjusted Fare.

# ### Missing Fare
# With fare adjusted, we should be able to fill in the missing Fare value by the Pclass the passenger is in.

# In[ ]:


# Calculate mean fare cost for each PClass
dict_fare_by_Pclass = dict(full.groupby('Pclass').Fare.mean())
# fill value according to PClass
missing_fare = full.loc[full.Fare.isnull(),'Pclass'].map(dict_fare_by_Pclass)
full.loc[full.Fare.isnull(),'Fare'] = missing_fare


# ## Exploring the Data - Distributions

# In[ ]:


# Descriptive Statistics
display(full.describe())
print(f"survived: {full.Survived.mean()*100:.2f}%")


# In[ ]:


# EDA - Distributions
var_to_plot = ['Pclass','Sex','SibSp','Parch','Embarked','Survived']

# Plot Categorical Var
fig, axs = plt.subplots(4,3, figsize=(15,12))
for i,key in enumerate(var_to_plot):
    sns.countplot(key, data=full, ax=axs[i//3,i%3])
     
# Plot Age
plt.subplot2grid((4,3),(2,0),rowspan=1,colspan=3);
sns.distplot(full.Age.dropna(), bins=range(0,80,2), kde=False)
plt.xlabel('Age');

# Plot Fare
plt.subplot2grid((4,3),(3,0),rowspan=1,colspan=3);
sns.distplot(full.Fare.dropna(), bins=100, kde=False)
plt.xlabel('Fare');
plt.tight_layout()


# ## EDA - Relationships between features and survival

# In[ ]:


# Plot all categorical features with Survival rate
var_to_plot = ['Pclass','Sex','SibSp','Parch','Embarked','Cabin_short']

f, axs = plt.subplots(3,5, sharey=True)
coord = [(0,0),(0,2),(1,0),(1,2),(2,0),(2,2)]
for i,key in enumerate(var_to_plot): # except feature Survived
    plt.subplot2grid((3,5),(coord[i]),rowspan=1,colspan=2);
    sns.barplot(data = full, x= key, y='Survived', color='darkgreen');
    plt.axhline(y=0.3838, color='k', linestyle='--')

# Plot Correlation
corr = pd.DataFrame(full.corr()['Survived'][:-1])
plt.subplot2grid((3,5),(0,4),rowspan=3,colspan=1);
sns.heatmap(corr, cmap = "BrBG", annot = True, annot_kws = {'fontsize': 12 });
plt.tight_layout()


# Findings:
# - Sex seems to have a strong predictive power, which makes sense due to the "Women and Children First" instructions for deciding who can get on the lifeboats. <br>
# - Pclass and Fare also showed a moderate correalation with Survival. These higher class passengers lives and have most of their activities near the deck, thus, closer to the lifeboats. <br>
# - It is surprising to find no significant correlation between Age and Survived. Their relationship may not be linear.
# - Cabin seem to have some relationships with survival, although we have lots of Nan values in this feature. Perhaps it's possible to guess those Nan values after looking into its relationships with Ticket no., Embark and PClass.
# - Embark C seem have significantly higher survival rate compared to Embark S, which also have a relatively low variance, There may be a relationship of where they board the Titanic and where they stay on boat.

# In[ ]:


# Create DataFrame Features to record potential predictors for later model training
features = pd.DataFrame()
features['Pclass'] = full['Pclass']
features['Fare'] = full['Fare']
features['Sex'] = full['Sex']


# ### Ticket_short with Survival
# There were still too many types of tickets even after parsing. In the plot below, only those with >10 count are plotted.

# In[ ]:


d = dict(full['Ticket_short'].value_counts())
ticket_count = full['Ticket_short'].map(d)
# Show % survived by Ticket
display(full.groupby('Ticket_short').Survived.aggregate(['mean','count']).dropna().sort_values('count').transpose())
# Plot % survived by Ticket, droping those tickets with <10 count
sns.barplot(data = full[ticket_count > 10], x = 'Ticket_short', y = 'Survived')
plt.axhline(y=0.3838, color='k', linestyle='--');


# Tickets with the most Predictive power: A5, PC. 

# In[ ]:


features['A5'] = (full['Ticket_short'] == 'A5').astype(int)
features['PC'] = (full['Ticket_short'] == 'PC').astype(int)


# ### Further exploring the relationship between Pclass, Sex, Age and Survival
# I suspect that some sort of interaction may exist between PClass, Sex and Age on predicitng survival. It is plotted below.

# In[ ]:


# Plot number of survived passengers by PClass, Sex and Age
facet = sns.FacetGrid(full, row = 'Pclass',col='Sex', hue = 'Survived', aspect=2, palette = 'Set1')
facet.map(plt.hist, 'Age', histtype='step', bins = np.arange(0,80,4))

facet.add_legend();


# Findings: 
# - Agree with the "Women first" pattern. <br>
# - Child survival advantage seems to apply for those < 12 years old for male. <br>
# - Much higher survival rate for people in 1st and 2nd class. Children and Women in these 2 classes have a much higher survival rate (some age range even with  100%), compared to those in the 3rd class (which has around 50% chance). Still, some women did not survive in the 1st and 2nd class, perhaps they were in a really bad location on the boat, even they live near the deck level.

# ### Age and Sex
# Instead of count of passangers survived, ploting the rate of survival with Age may give us a clearer look on the effect of age on survival rate.
# 

# In[ ]:


# Create Age Quartiles
Age_quartile = pd.qcut(full.Age,10)

# Plot age quartiles by sex with survival rate
sns.barplot(data = full, x= Age_quartile, y='Survived', hue = 'Sex');
plt.axhline(y=0.3838, color='k', linestyle='--')
plt.xticks(rotation = 30)
plt.title('Across All Classes');


# From the FacetGrid and the bar plot above, it seems that age does not matter on rate of survival for female. For male, Survival advantage for males seem to appy for those with Age < 12.

# ## Feature Engineering 1
# After a brief look of relationships between the existing features, it's time to engineer some new features from the existing features to improve the predictive power of the model.
# 
# First of all, we can try to extract the title of passengers from the name. Apart from seeing their predicting power, this can give us information about the pasenger's age,
# 

# ### Title - Filling Missing Age
# Titles of the Passengers may give us valuable information to infer the missing age, e.g., Master is a title for children. Additonally, royalties and officials may have a higher priority to get on the lifeboats.

# In[ ]:


# Parse Titles from Names
def parse_title(str):
    m = re.search(', (\w+ *\w*)\.',str)
    return m.group(1)
    
title = full.Name.map(parse_title)
title.unique()


# In[ ]:


# Simplify title groups
dict_Title = {"Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }

title = title.map(dict_Title)

# Plot the distribution of Age by Title
plt.figure(figsize = (14,6))
sns.violinplot(x = title, y = full['Age']);


# Passengers with title 'Master' are likely to be children, we can infer those missing age as the mean age of Master
# Passengers with title 'Miss' seem to comprise both children and adult, the followings is an attempt to infer their age from other given features <br>
# However, age of female here is relatively unimportant, since all female regardless of age have high priority to board the lifeboats.

# In[ ]:


# Calculate mean age of each title group
df_title = pd.DataFrame(title).join(full[['Age','Survived']])
dict_age = df_title.groupby('Name').Age.mean()

# Fill in Age according to passenger's title
idx = full.Age.isnull()
full.loc[idx,'Age'] = df_title.loc[idx, 'Name'].map(dict_age)


# ### Title - Survival

# In[ ]:


# Plot title with Survived
sns.barplot(data = df_title, x= 'Name', y='Survived');
plt.axhline(y=0.3838, color='k', linestyle='--');


# In[ ]:


# Record useful features in features dataframe
features['Title'] = df_title['Name']
features['Child'] = (full['Age'] <= 14).astype(int)


# ## Feature Engineering 2
# Another  interesting relationships to look at is between Survival, Parch and SibSp. It is not difficult to imagine those within the same family/ same group will stay together when in danger, thus, having any of them survived would mean the other members of the group will likely to have a better chance to survive, and vice versa.

# ### Surname
# First, parse the Surnames of the passengers. Those from the same family should share the surname. <br>
# Surnames are grouped together and their occurance caluculated respectively.

# In[ ]:


# function to parse surname of the passengers
def parse_surname(name):
    return name.split(',')[0]
# Calculate Family Size
family = pd.DataFrame(full[['Parch','SibSp','Ticket']])
family['Family_size'] = 1 + family.Parch + family.SibSp

# Parse Surname from Name
family['Surname'] = full.Name.map(parse_surname)

# Surname Code and Surname Size
dict_scount = dict(family.groupby('Surname').Family_size.count())
dict_scode = dict(zip(dict_scount.keys(), range(len(dict_scount))))

family['Surname_code'] = family['Surname'].map(dict_scode)
family['Surname_count'] = family['Surname'].map(dict_scount)

# Examples with common surname
display(full[family.Surname == 'Smith'])


# However, some common surnames may be shared by people from different families. <br> 
# The following function is an attempt to address this issue: <br>
# To judge if passengers are likely to be in the same family, the function check their ticket code.  <br>
# The function decides if people with the same surname are from the same family by checking the level of  similarity of their tickets. Those with the exact same tickets or tickets that have values close to each other are grouped together.

# In[ ]:


def tick2fam_gen(df):
    """
    Function to judge if passengers are likely to be in the same family.
    Input: DataFrame with Passenger surname and ticket
    Return: Code generated to specify different families
    """
    # initialize ticket dict
    dict_tick2fam = {'000000': 0}
    fam_counter = 0
        
    for i in df.index:    
        keys = list(dict_tick2fam.keys())
        chk_key = df.loc[i, 'Ticket']
        for key in keys:
            if len(chk_key) == len(key): #if their tickets have high similarity
                if (chk_key[-4:].isdigit()) & (key[-4:].isdigit()): 
                    if (chk_key[:-2] == key[:-2]) & (np.abs(int(chk_key[-2:]) - int(key[-2:])) <= 10):
                        dict_tick2fam[chk_key] = dict_tick2fam[key]
                        break
                    
            if key == keys[-1]: # no match, assign a new code to the passenger
                fam_counter += 1
                dict_tick2fam[chk_key] = str(fam_counter)  
                
    return dict_tick2fam


# In[ ]:


# Single out Surnames with size > true family size (may have more than 1 family involved)
surname2chk = family[family['Family_size'] < family['Surname_count']].Surname.unique() 
# chk_surname2 = family_infer[family['FamilySize'] > family['SurnameSize']].Surname.unique() # unidentified fam

# Regrouping Families according to Family Size and Ticket.
family['Surname_adj'] = family['Surname'] #new column for corrected family_group

for s in surname2chk:
    family_regroup = family[family['Surname'] == s] #get family with specific surname
    fam_code_dict = tick2fam_gen(family_regroup) #pass in df to get family codes within the same surname

    for idx in family_regroup.index: #assign family code 1by1
        curr_ticket = full.loc[idx].Ticket
        fam_code = fam_code_dict[curr_ticket]

        if family_regroup.loc[idx, 'Family_size'] == 1: #for passengers traveling alone
            #relatives that shares surname and ticket, which Parch and SibSp failed to record
            if family_regroup.Ticket.value_counts()[curr_ticket] > 1: 
                family.loc[idx, 'Surname_adj'] =  s + '-hidfam' + fam_code
            #single traveler
            else: 
                family.loc[idx, 'Surname_adj'] =  s + '-single' + fam_code
        #different families
        else: 
            family.loc[idx, 'Surname_adj'] =  s + '-fam' + fam_code

display(family[family.Surname == 'Smith'])


# After Adjusting the surnames of families, group these true families together again. The no. of families here should increase.

# In[ ]:


# Assign codes to families
dict_fcount = dict(family.groupby('Surname_adj').Family_size.count())
dict_fcode = dict(zip(dict_fcount.keys(), range(len(dict_fcount))))

family['Family_code'] = family['Surname_adj'].map(dict_fcode)
family['Family_count'] = family['Surname_adj'].map(dict_fcount)

print(f"No. of Family Before Regrouping: {len(family.Surname_code.unique())}")
print(f"No. of Family After Regrouping: {len(family.Family_code.unique())}")


# ### Identify Roomates by Ticket
# People who share the same ticket can be families as well as friends traveling together. They are expected to stay together during the incidents. 

# In[ ]:


# Identify Groups (Those holding the same ticket code, could be friends/family)
group = pd.DataFrame(family[['Surname_code','Surname_count','Family_code','Family_count']])

dict_tcount = dict(full.groupby('Ticket').PassengerId.count())
dict_tcode = dict(zip(dict_tcount.keys(),range(len(dict_tcount))))

group['Ticket_code'] = full.Ticket.map(dict_tcode)
group['Ticket_count'] = full.Ticket.map(dict_tcount)

print(f"No. of Tickets Identified: {len(group['Ticket_code'].unique())}")
display(full[(full.Ticket == 'A/4 48871') |(full.Ticket == 'A/4 48873')])


# ### Combining Friends and Families as Groups
# Finally, the families and friend groups are combined together.  <br>
# People who share either the same room or same family are grouped together.

# In[ ]:


def ChainCombineGroups(df, colA, colB):
    '''
    This function takes in 2 columns of labels and chain all items which share
    the same labels within each of the 2 columns
    input:
    df - DataFrame
    colA - Key for Col
    colB - Key for Col  
    output:
    array of numeric grouping labels
    '''
    # make a copy of DFs for iteration
    data = df.copy()
    search_df = data.copy()
    
    group_count = 0

    while not search_df.empty:

        # Initiate pool and Select Reference item
        pool = search_df.iloc[:1]
        idx = pool.index

        # Remove 1st item from searching df
        search_df.drop(index = idx, inplace = True)

        # Initialize Search
        flag_init = 1
        update = pd.DataFrame()

        # While loop to exhausively search for commonalities, pool is updated until no more common features are found
        while (flag_init or not update.empty):

            flag_init = 0

            # target labels to look for
            pool_A_uniq = np.unique(pool[colA])
            pool_B_uniq = np.unique(pool[colB])

            for col in [colA,colB]:
                idx = []

                # get all indexs of items with the same label
                for num in np.unique(pool[col]):
                    idx.extend(search_df[search_df[col] == num].index)

                # update pool
                update = search_df.loc[idx]
                pool = pd.concat([pool, update], axis = 0)

                # remove item from searching df
                search_df = search_df.drop(index = idx)

            # assign group num
            data.loc[pool.index, 'Group_'] = group_count

        group_count += 1
        
    return np.array(data['Group_'].astype(int))


# In[ ]:


# Assign Final group no.
group['Group_code'] = ChainCombineGroups(group, 'Family_code', 'Ticket_code')

# Calculate group sizes
dict_gcount = dict(group.groupby('Group_code').Family_code.count())
group['Group_count'] = group.Group_code.map(dict_gcount)
         
print(f"Family: {len(family['Family_code'].unique())}")
print(f"Group: {len(group['Ticket_code'].unique())}")
print(f"Combined: {len(group['Group_code'].unique())}\n")
print('An example of grouping the both friends and family under a same group:')
display(pd.concat([full['Ticket'],family[['Surname','Family_code']],group[['Ticket_code','Group_code']]], axis = 1)[group['Group_code'] == 458])


# ### Limitations:
# The above function did fail to join some families back together, especially those who had different ticket numbers and had different surnames. <br> 
# For example, female siblings who were married and took different surnames; <br>
# and families who bought tickets with codes that has low similarity, which is likely to be found for those in the 1st Class. 

# ### Survival of the Group
# Finally, the thing that we wanted to know in the first place is if the members in their Family/Friends group has survived or not. Having a surviving friend/family member should have good predictive power of whether a passenger survived or not.

# In[ ]:


# Prepare the df by adding the Survived features
group_final = pd.concat([family[['Surname_code','Surname_count','Family_code','Family_count']],
                       group[['Ticket_code','Ticket_count','Group_code','Group_count']],
                        full['Survived']], axis = 1)


# In[ ]:


for param in [('Surname_code','Surname_count'),
              ('Family_code','Family_count'),
              ('Ticket_code','Ticket_count'),
              ('Group_code','Group_count')]: # keep group at last
    
    # No. of member survived in each group
    n_member_survived_by_gp = group_final.groupby(param[0]).Survived.sum()
    
    # No. of member survived in a particular group, discounting the passenger concerned
    n_mem_survived = group_final[param[0]].map(n_member_survived_by_gp)
    n_mem_survived_adj = n_mem_survived - group_final.Survived.apply(lambda x: 1 if x == 1 else 0)

    # Same for the dead
    n_member_dead_by_gp = group_final.groupby(param[0]).Survived.count() - group_final.groupby(param[0]).Survived.sum()
    n_mem_dead  = group_final[param[0]].map(n_member_dead_by_gp)
    n_mem_dead_adj = n_mem_dead - group_final.Survived.apply(lambda x: 1 if x == 0 else 0)

    # How many people from that group that we do not have data on.
    unknown_factor = (group_final[param[1]] - n_mem_survived_adj - n_mem_dead_adj)/group_final[param[1]]
    confidence = 1 - unknown_factor

    # Ratio of members survived in that group, ranging from -1 to 1, adjusted by the confidence weight
    key = 'Confidence_member_survived'+'_'+param[0]
    ratio = (1/group_final[param[1]]) * (n_mem_survived_adj - n_mem_dead_adj)
    group_final[key] = confidence * ratio

# Display Correlation
plt.barh(group_final.corr().Survived[-4:].index, group_final.corr().Survived[-4:])
plt.xlabel('Correlation with Survived');

features['Cf_mem_survived'] = group_final['Confidence_member_survived_Group_code']


# In[ ]:


features['Parch'] = full['Parch']
features['SibSp'] = full['SibSp']
features['Group_size'] = group['Group_count']

features.head()


# ## Data Transformation
# Used StanardScalar for continuous variables and One-hot encoding for Categorical ones.

# In[ ]:


from sklearn.preprocessing import StandardScaler

# Standardize the continuous variables
scalar = StandardScaler()
features_z_transformed = features.copy()
continuous = ['Fare'] 
features_z_transformed[continuous] = scalar.fit_transform(features_z_transformed[continuous])

# Transform Sex labels into binary code
features_z_transformed.Sex = features_z_transformed.Sex.apply(lambda x: 1 if x == 'male' else 0)

# One-hot Encoding
features_final = pd.get_dummies(features_z_transformed)

encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Seperate Train Data and Test Data
features_final_train = features_final[:891]
features_final_test = features_final[891:]


# ## Model Training and Selection
# 

# In[ ]:


# Spliting Training Sets into Train and Cross-validation sets
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

X_train, X_test, y_train, y_test = train_test_split(features_final_train, 
                                                    train.Survived, 
                                                    test_size = 0.2, 
                                                    random_state = 0)


# In[ ]:


# Create Model Training Pipeline
from sklearn.metrics import accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    
    # Get the predictions on the test set(X_test),
    predictions_test = learner.predict(X_test)
    
    # then get predictions on the training samples(X_train)
    predictions_train = learner.predict(X_train)
            
    # Compute accuracy on the training samples
    results['acc_train'] = accuracy_score(y_train, predictions_train)
        
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
       
    # Success
    print("{} trained on {} samples. Acc: {:.4f}".format(learner.__class__.__name__, sample_size, results['acc_test']))
        
    # Return the results
    return results


# In[ ]:


# Import the three supervised learning models from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier

# Initialize the three models
clf_A = GradientBoostingClassifier(random_state = 0)
clf_B = LogisticRegression(random_state= 0)
clf_C = RandomForestClassifier(random_state= 0)

# Calculate the number of samples for 10%, 50%, and 100% of the training data
samples_100 = len(y_train)
samples_10 = int(len(y_train)/2)
samples_1 = int(len(y_train)/10)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] =         train_predict(clf, samples, X_train, y_train, X_test, y_test)


# In[ ]:


# Reshaping the Results for plotting
df = pd.DataFrame()

for i in results.items():
    temp = pd.DataFrame(i[1]).rename(columns={0:'1% of train', 1:'10% of train', 2:'100% of train'})
    temp['model'] = i[0]
    df = pd.concat([df, temp], axis = 0)
df_plot = df.reset_index().melt(id_vars=['index','model'])

# Ploting the results
fig, axs = plt.subplots(1,2,figsize = (16,5))
for i,key in enumerate(df_plot['index'].unique()[:2]):
    ax = axs[i%2]
    sns.barplot(data = df_plot[df_plot['index'] == key], x = 'model', y = 'value',
                hue = 'variable', ax = ax)
    ax.set_ylim([0.6,1])
    ax.set_title(key)
    ax.legend(loc="lower right")



# ## Model Selection and model tuning
# RandomForestClassifier seemed to have the best out of the box accuracy score and with room for improvement as seen in acc_train.
# Model tuning is performed using GridSearchCV to improve generalizability of the model.

# In[ ]:


from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
warnings.filterwarnings('ignore')

clf = RandomForestClassifier(random_state = 0, oob_score = True)

parameters = {'criterion' :['gini'],
             'n_estimators' : [350], #400
             'max_depth':[5], #5
             'min_samples_leaf': [4], #4
              'max_leaf_nodes': [10], #10]
              'min_impurity_decrease': [0], #0
              'max_features' : [1] #1
             }

scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf, parameters, scoring = scorer, cv = 10)

grid_fit = grid_obj.fit(X_train,y_train)

best_clf = grid_fit.best_estimator_

predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("Oob score on testing data: {:.4f}".format(clf.oob_score_))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final oob score on the testing data: {:.4f}".format(best_clf.oob_score_))
print("\nBest Parameters\n------")
best_clf


# In[ ]:


# Plot Feature Importnace
idx = np.argsort(best_clf.feature_importances_)
plt.figure(figsize = (12,8))
plt.barh(range(len(best_clf.feature_importances_)),best_clf.feature_importances_[idx])
plt.yticks(range(len(best_clf.feature_importances_)),features_final_train.columns[idx]);
plt.title('Feature Importance');


# In[ ]:


# Output for Kaggle competition
final_predict = best_clf.predict(features_final_test)

prediction = pd.DataFrame(full[891:].PassengerId)
prediction['Survived'] = final_predict.astype('int')

prediction.to_csv('predict.csv',index = False)

