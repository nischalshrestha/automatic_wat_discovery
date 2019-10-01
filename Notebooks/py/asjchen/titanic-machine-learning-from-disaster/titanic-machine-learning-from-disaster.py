#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# ### Andy Chen
# ### July 15, 2017

# ## Introduction

# On April 15, 1912, the RMS Titanic struck an iceberg in the North Atlantic Ocean. The impact ruptured several hull compartments, causing the passenger ship to sink. Without enough lifeboats on board or satisfactory emergency procedures, 1502 out of the 2224 passengers and crew perished, causing disbelief in the international community. After the incident, the first International Convention for Safety of Life at Sea (SOLAS) treaty mandated that ships carry enough lifeboats for everyone aboard, hold lifeboat and evacuation drills, and maintain a 24-hour radio watch.
# 
# In this notebook, we analyze which passengers survived the Titanic sinking and what characteristics they have in common. Such features would tell us about how the ship's evacuation occurred. Were women and children given higher priority for the lifeboats? Were certain living quarters less accessible to lifeboats? To test our theories, we examine roughly 900 passengers' data, including whether they survived or not. Then, we set aside approximately 400 of the remaining passengers and attempt to predict which of them survived the sinking.
# 
# The Kaggle page is here: https://www.kaggle.com/c/titanic

# ## Feature Analysis

# First, we open both the training and test data and examine our raw information.

# In[1]:


import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

known_data = pd.read_csv('../input/train.csv')
unknown_data = pd.read_csv('../input/test.csv')

known_data.head()


# We are given that:
# 
# * **Survived** is an integer in {0, 1}
# * **Pclass** (ticket class) is an integer in {1, 2, 3}
# * **Name** is a string, possibly consisting of multiple names
# * **Sex** is either male or female
# * **Age** is a nonnegative float, generally either in [0, 1] or of the form xx.0 or xx.5
# * **SibSp** (number of siblings on board) is a nonnegative integer
# * **Parch** (number of parents or children on board) is a nonnegative integer
# * **Ticket** (ticket number) is a string
# * **Fare** is a nonnegative float
# * **Cabin** (cabin number) is a string
# * **Embarked** (port of embarkation) is in {C, Q, S}, representing {Cherbourg, Queenstown, Southampton}

# We need to also determine which columns have missing data, which we may need to infer based on the other features.

# In[2]:


def num_null_features(data):
    num_null = pd.isnull(data).sum().to_dict()
    return {col: num_null[col] for col in num_null if num_null[col] > 0}

print('Length of Known Data: %s' % len(known_data))
print('Length of Unknown Data: %s' % len(unknown_data))
print('Number Null for Features in Known Data: %s' % num_null_features(known_data))
print('Number Null for Features in Unknown Data: %s' % num_null_features(unknown_data))


# This information tells us that it could be nice to be able to infer `Fare`, but doing so would help us only a tiny bit. When analyzing the `Embarked` feature, we could ignore the two rows in `known_data` that have null values. It is much more important that we find relations between `Age` and `Cabin` with the other features.
# 
# We now examine how each feature, on its own, correlates with survival. For many of the following analyses, we use a bar graph to visualize the frequency of certain feature values.

# In[3]:


# Bar Graph Functionality
def bar_graphs(feature, labels, xtick_locs, bar_width=0.35, map_function=None, data=known_data):
    assert len(labels) == len(xtick_locs)
    
    survived_data = data[data.Survived == 1]
    survived_feature = survived_data[feature]
    if map_function is not None:
        survived_feature = survived_feature.map(map_function, na_action='ignore')
    survived_frequencies = [len(survived_feature[survived_feature == lbl]) for lbl in labels]
    
    perished_data = data[data.Survived == 0]
    perished_feature = perished_data[feature]
    if map_function is not None:
        perished_feature = perished_feature.map(map_function, na_action='ignore')
    perished_frequencies = [len(perished_feature[perished_feature == lbl]) for lbl in labels]
    
    survived_bars = plt.bar(xtick_locs, survived_frequencies, bar_width, bottom=perished_frequencies)
    perished_bars = plt.bar(xtick_locs, perished_frequencies, bar_width)
    
    return survived_bars, perished_bars


# ### Ticket Class (Pclass)

# We first examine a basic histogram of the ticket classes among the passengers in the known data.

# In[4]:


# Ticket Class Frequency
survived_bars, perished_bars = bar_graphs('Pclass', range(1, 4), range(3))
plt.ylabel('Frequency')
plt.xlabel('Ticket Class')
plt.title('Ticket Classes Frequency')
plt.xticks(range(3), ['1', '2', '3'])
plt.yticks(np.arange(0, 600, 100))
plt.legend((survived_bars[0], perished_bars[0]), ('Survived', 'Perished'))
plt.show()


# We can see that the greatest proportion of passengers are in the 3rd ticket class, which roughly reflects the lowest social class on the ship. In addition, the class with highest survival rate is class 1, followed by class 2.

# ### Name

# While the name isn't a quantitative value on its own, we can still extract some useful quantitative information from it, such as social class or family relationship. Let's parse the `Name` values into separate parts.

# In[5]:


# Name Parsing

def split_names(data):
    # Each name is of the form <last_name>, <title>. <first_names> (<maiden name_if_applicable>)
    last_names = data.Name.map(lambda s: s.split(',')[0].strip())
    last_names = last_names.rename('LastName')
    first_name_parts = data.Name.map(lambda s: s.split(',')[1].strip())
    titles = first_name_parts.map(lambda s: s.split('.')[0].strip())
    titles = titles.rename('Title')
    maiden_full_names = first_name_parts.map(lambda s: s.split('(')[1].split(')')[0] if s.find('(') != -1 else None)
    maiden_names = maiden_full_names.map(lambda s: s.split()[-1], na_action='ignore')
    maiden_names = maiden_names.rename('MaidenName')
    name_data = pd.DataFrame([titles, last_names, maiden_names]).T
    return name_data
    
split_names(known_data).head()


# Let us first examine the titles, which could be indicative of social class and possibly of survival. 

# In[6]:


# Title Frequency
known_name_data = pd.concat([known_data, split_names(known_data)], axis=1)

# list of distinct titles
titles = list(known_name_data.Title.unique())
print(titles)
survived_bars, perished_bars = bar_graphs('Title', titles, range(len(titles)), data=known_name_data)
plt.ylabel('Frequency')
plt.xlabel('Title')
plt.title('Title Frequency')
plt.xticks(rotation=45)
plt.xticks(rotation=45, ha='right')
plt.xticks(range(len(titles)), titles)
plt.yticks(np.arange(0, 500, 100))
plt.legend((survived_bars[0], perished_bars[0]), ('Survived', 'Perished'))
plt.show()


# We can see that the majority of Mr's perished while the majority of Mrs's, Miss's, and Master's survived. Examining the less frequent titles yields.

# In[7]:


# Rare Title Frequency
known_name_data = pd.concat([known_data, split_names(known_data)], axis=1)

# list of distinct titles
rare_titles = list(known_name_data.Title.unique())[4: ]
survived_bars, perished_bars = bar_graphs('Title', rare_titles, range(len(rare_titles)), data=known_name_data)
plt.ylabel('Frequency')
plt.xlabel('Title')
plt.title('Title Frequency')
plt.xticks(rotation=45, ha='right')
plt.xticks(range(len(rare_titles)), rare_titles)
plt.yticks(np.arange(0, 10, 2))
plt.legend((survived_bars[0], perished_bars[0]), ('Survived', 'Perished'))
plt.show()


# ### Sex

# Now, we check whether one sex's survival rate is significantly different from the other sex's rate.

# In[8]:


# Sex Frequency
sexes = ['female', 'male']
survived_bars, perished_bars = bar_graphs('Sex', sexes, range(2))
plt.ylabel('Frequency')
plt.xlabel('Ticket Class')
plt.title('Sex Frequency')
plt.xticks(range(2), sexes)
plt.yticks(np.arange(0, 700, 100))
plt.legend((survived_bars[0], perished_bars[0]), ('Survived', 'Perished'))
plt.show()


# We can see that most female passengers actually survived while most male passengers perished.

# ### Age

# We can also analyze the ages of the passengers, when available. We may expect that children, with lower age, have higher survival rates.

# In[9]:


# Age Histogram

bucket_width = 5
max_age = int(math.ceil(known_data.Age.max()))
ages = range(0, max_age + 1, bucket_width)
bucketer = lambda x: (math.floor(x / bucket_width) * bucket_width)
survived_bars, perished_bars = bar_graphs('Age', ages, ages, bar_width=4.5, map_function=bucketer)

plt.ylabel('Frequency')
plt.xlabel('Age')
plt.title('Age Histogram')
plt.xticks([age - 2.5 for age in ages], ages)
plt.yticks(np.arange(0, 150, 25))
plt.legend((survived_bars[0], perished_bars[0]), ('Survived', 'Perished'))
plt.show()


# Indeed, the survival rate is much higher for children below age 5, though the survival rate stays somewhat constant across ages above 5.

# Since many passengers in both `known_data` and `unknown_data` have the Age parameter missing, we can use a random forest algorithm to fill in those missing age values based on other features. (More details in the prediction section.)

# ### Number of Siblings

# We now investigate how the number of siblings for each passenger correlates with his/her survival rate.

# In[10]:


# Sibling Number Frequency
max_siblings = known_data.SibSp.max()
survived_bars, perished_bars = bar_graphs('SibSp', range(max_siblings + 1), range(max_siblings + 1))
plt.ylabel('Frequency')
plt.xlabel('Number of Siblings')
plt.title('Sibling Number Frequency')
plt.xticks(range(max_siblings + 1), range(max_siblings + 1))
plt.yticks(np.arange(0, 700, 100))
plt.legend((survived_bars[0], perished_bars[0]), ('Survived', 'Perished'))
plt.show()


# On its own, the number of siblings doesn't seem to be too correlated with survival, though high numbers seems to yield lower survival rates.

# ### Number of Parents and Children

# Next, we examine the number of parent and children.

# In[11]:


# Parent/Children Number Frequency
max_parch = known_data.Parch.max()
survived_bars, perished_bars = bar_graphs('Parch', range(max_parch + 1), range(max_parch + 1))
plt.ylabel('Frequency')
plt.xlabel('Number of Parents/Children')
plt.title('Parent/Children Number Frequency')
plt.xticks(range(max_parch + 1), range(max_parch + 1))
plt.yticks(np.arange(0, 800, 100))
plt.legend((survived_bars[0], perished_bars[0]), ('Survived', 'Perished'))
plt.show()


# Again, there doesn't seem to be a notable correlation between the parent/children number and survival, other than those with nonzero numbers having slightly higher survival rates.
# 
# It is worth checking how many passengers have both siblings and parents/children on board.

# In[12]:


# Passengers with both siblings and parents/children on board
has_siblings_data = known_data[known_data.SibSp > 0]
has_both = has_siblings_data[has_siblings_data.Parch > 0]
print('Number of Passengers with Both Siblings and Parents/Children: %s' % len(has_both))
print('Number of Total Passengers: %s' % len(known_data))


# We can also consider family size, which is `SibSp` + `Parch` + 1.

# In[13]:


# Family Size Frequency
known_family_data = known_data.copy()
known_family_data['FamilySize'] = known_family_data.SibSp + known_family_data.Parch + 1
max_family = known_family_data.FamilySize.max()
survived_bars, perished_bars = bar_graphs('FamilySize', range(max_family + 1), range(max_family + 1), data=known_family_data)
plt.ylabel('Frequency')
plt.xlabel('Number of Family Members')
plt.title('Family Number Frequency')
plt.xticks(range(max_family + 1), range(max_family + 1))
plt.yticks(np.arange(0, 800, 100))
plt.legend((survived_bars[0], perished_bars[0]), ('Survived', 'Perished'))
plt.show()


# Again, we see that family sizes between 2 and 4 seem to yield lower survival rates, but family size doesn't offer much other direct correlation to survival.

# ### Ticket Number

# The ticket number appears to be somewhat unstructured; most tickets have a numerical suffix (though some also end in "LINE") and possible a prefix (ex. A/5 or PC). We can check whether having this prefix is correlated with survival.

# In[14]:


# Ticket Number Prefix Frequency
prefix_data = known_data.copy()
prefix_data['HasTicketPrefix'] = prefix_data.Ticket.map(lambda ticket: (ticket.strip().find(' ') != -1))
survived_bars, perished_bars = bar_graphs('HasTicketPrefix', [True, False], range(2), data=prefix_data)
plt.ylabel('Frequency')
plt.xlabel('Ticket Number Type')
plt.title('Prefix Presence Frequency')
plt.xticks(range(2), ['Has Prefix', 'Has No Prefix'])
plt.yticks(np.arange(0, 800, 100))
plt.legend((survived_bars[0], perished_bars[0]), ('Survived', 'Perished'))
plt.show()


# We can see that on its own, the presence of a prefix does not seem to be correlated with survival.

# ### Ticket Fare

# Next, we examine the ticket fare for each passenger.

# In[15]:


# Ticket Fare Histogram
bucket_width = 5
max_fare = int(math.ceil(known_data.Fare.max()))
fares = range(0, max_fare + 1, bucket_width)
bucketer = lambda x: (math.floor(x / bucket_width) * bucket_width)
survived_bars, perished_bars = bar_graphs('Fare', fares, fares, bar_width=4.5, map_function=bucketer)

plt.ylabel('Frequency')
plt.xlabel('Ticket Fare')
plt.title('Ticket Fare Histogram')
plt.xticks([tick for tick in range(0, max_fare, 50)], range(0, max_fare, 50))
plt.yticks(np.arange(0, 350, 25))
plt.legend((survived_bars[0], perished_bars[0]), ('Survived', 'Perished'))
plt.show()


# It seems that the high majority of passengers that paid more than 75 pounds actually survived, which is consistent with our observations of ticket class. Zooming into the right part of the histogram yields the following. 

# In[16]:


# Zoomed Ticket Fare Histogram
bucket_width = 5
threshold = 75
max_fare = int(math.ceil(known_data.Fare.max()))
fares = range(0, max_fare + 1, bucket_width)[threshold // bucket_width: ]
bucketer = lambda x: (math.floor(x / bucket_width) * bucket_width)
survived_bars, perished_bars = bar_graphs('Fare', fares, fares, bar_width=4.5, map_function=bucketer)

plt.ylabel('Frequency')
plt.xlabel('Ticket Fare')
plt.title('Ticket Fare Histogram')
plt.xticks([tick for tick in range(threshold, max_fare, 50)], range(threshold, max_fare, 50))
plt.yticks(np.arange(0, 25, 5))
plt.legend((survived_bars[0], perished_bars[0]), ('Survived', 'Perished'))
plt.show()


# ### Cabin Number

# In our data, cabin number is commonly a blank field. While the actual cabin number may not be too informative, the fact that someone has a cabin number might be. We can take a look at the passengers who actually have cabin rooms; since such rooms may be more common for those with higher ticket class, we plot a bar graph with respect to ticket class.

# In[17]:


# Cabin Number vs. Ticket Class
has_cabin_data = known_data[known_data.Cabin.notnull()]
survived_bars, perished_bars = bar_graphs('Pclass', range(1, 4), range(3), data=has_cabin_data)
plt.ylabel('Frequency')
plt.xlabel('Ticket Class')
plt.title('Ticket Classes Frequency')
plt.xticks(range(3), ['1', '2', '3'])
plt.yticks(np.arange(0, 200, 50))
plt.legend((survived_bars[0], perished_bars[0]), ('Survived', 'Perished'))
plt.show()


# Recall that passengers in ticket class 1 have higher survival rates. In this situation, we reaffirm that observation, and we also note that most passengers with cabins were in class 1. In all cases, the survival rate was high, so having a cabin is correlated with survival.
# 
# Since cabins with different decks (the letters in front of the numbers) were likely in different locations on the ship, we can examine the passengers' decks and determine whether they were correlated with survival.

# In[18]:


# Cabin Histogram
cabin_set = known_data.Cabin.unique()
decks = sorted(list(set(re.sub('[\s0-9]+', '', ''.join(cabin_set[1:])))))
survived_list = []
deck_list = []
for idx in range(len(known_data)):
    if known_data.iloc[idx].Cabin is np.nan:
        continue
    for deck in decks:
        if deck in known_data.iloc[idx].Cabin:
            survived_list.append(known_data.iloc[idx].Survived)
            deck_list.append(deck)
cabin_data = pd.DataFrame({'Survived': survived_list, 'CabinDeck': deck_list})
survived_bars, perished_bars = bar_graphs('CabinDeck', decks, range(len(decks)), data=cabin_data)
plt.ylabel('Frequency')
plt.xlabel('Cabin Deck')
plt.title('Cabin Deck Frequency')
plt.xticks(range(len(decks)), decks)
plt.yticks(np.arange(0, 80, 20))
plt.legend((survived_bars[0], perished_bars[0]), ('Survived', 'Perished'))
plt.show()


# The more common cabin letters (B, C, D, E, F) have the highest survival rates, which are not too different from the overall survival rate among all cabin passengers.

# ### Port of Embarkation

# Finally, we examine where the passengers embarked on their journeys. The possible ports are Cherbourg (Northwestern France), Queenstown (South Ireland, now called Cobh), and Southampton (South England). 

# In[19]:


# Embarkation Port Frequency
ports = ['C', 'Q', 'S']
survived_bars, perished_bars = bar_graphs('Embarked', ports, range(3))
plt.ylabel('Frequency')
plt.xlabel('Embarkation Port')
plt.title('Embarkation Port Frequency')
plt.xticks(range(3), ports)
plt.yticks(np.arange(0, 700, 100))
plt.legend((survived_bars[0], perished_bars[0]), ('Survived', 'Perished'))
plt.show()


# We observe a general lack of correlation between port and survival. However, we can also examine the relationship among survival, ticket class, and sex within the population originating from each port.

# In[20]:


# Relationships Within Passengers From the Same Port
ports = ['C', 'Q', 'S']
pclass_sex = ['1female', '2female', '3female', '1male', '2male', '3male']
triple_data = known_data.copy()
triple_data['Triple'] = triple_data.Pclass.map(str) + triple_data.Sex + triple_data.Embarked
for port in ports:
    triples = ['%s%s' % (ps, port) for ps in pclass_sex]
    survived_bars, perished_bars = bar_graphs('Triple', triples, range(len(triples)), data=triple_data)
    plt.ylabel('Frequency')
    plt.xlabel('Ticket Class and Sex')
    plt.title('Ticket Class and Sex within Port %s' % port)
    plt.xticks(rotation=45, ha='right')
    plt.xticks(range(len(triples)), ['Class %s, %s' % (t[:1], t[1:-1]) for t in triples])
    if port != 'S':
        plt.yticks(np.arange(0, 70, 10))
    else:
        plt.yticks(np.arange(0, 400, 50))
    plt.legend((survived_bars[0], perished_bars[0]), ('Survived', 'Perished'))
    plt.show() 


# The trends within each population seem consistent with previous observations of sex; most females survived in every category except for those from Port S (Southampton) in class 3. We also note that nearly all passengers in Queenstown were in class 3.

# ## Prediction Methods

# First, since we don't have access to the test survival data, we perform k-fold cross validation on the training data to test our training methods' accuracy. We set k = 10.

# In[21]:


# model is a function with parameters (training data, test data) that returns a list of entries from {0, 1}
def k_fold_cross_validation(model, k = 10):
    boundaries = [i * len(known_data) // k for i in range(k + 1)]
    total_accuracy = 0.0
    for i in range(k):
        non_eval_data = pd.concat([known_data[: boundaries[i]], known_data[boundaries[i + 1]: ]])
        eval_data = known_data[boundaries[i]: boundaries[i + 1]]
        predictions = model(non_eval_data, eval_data)
        num_correct = len([idx for idx in range(len(predictions)) if predictions[idx] == eval_data.Survived.iloc[idx]])
        total_accuracy += float(num_correct) / len(known_data)
    return total_accuracy


# We also construct a function to output the predictions for the unknown dataset.

# In[22]:


# model is a function with parameters (training data, test data) that returns a list of entries from {0, 1}
def predict_unknown(model, csv_name):
    raw_predictions = model(known_data, unknown_data)
    survived = pd.Series(data=raw_predictions, name='Survived')
    predictions = pd.concat([unknown_data.PassengerId, survived], axis=1)
    predictions.to_csv(csv_name, index=False)


# As a baseline, we look at our accuracy if we asserted that every passenger in our test perished.

# In[23]:


# Everyone Perishes
def everyone_perishes(train_data, test_data):
    return [0] * len(test_data)

print('Everyone Perishes: %s' % k_fold_cross_validation(everyone_perishes))


# In the remainder of our training methods, we construct a feature vector based on our previous analysis. Recall that we need to fill in the missing age, fare, and embarked port values using the other features. Since age is a numerical feature (that is commonly missing), we can use random forest training. Since fare is a numerical feature that is heavily correlated with ticket class (and only missing for one data point), we use the median of the fares for passengers in the same ticket class. Since port is a categorial feature that is correlated with ticket class, we use the mode of the the ticket class to fill the missing ports.

# In[24]:


def full_embarked(data):
    embarked_col = data.Embarked.copy()
    for idx in range(len(data)):
        if data.iloc[idx].Embarked == np.nan:
            pclass = data.iloc[idx].Pclass
            sim_data = data[data.Pclass == pclass]
            embarked_col.iloc[idx] = sim_data.Embarked.mode().iloc[0]
    return embarked_col

def full_fare(data):
    fare_col = data.Fare.copy()
    for idx in range(len(data)):
        if data.Fare.isnull().iloc[idx]:
            pclass = data.iloc[idx].Pclass
            sim_data = data[data.Pclass == pclass]
            fare_col.iloc[idx] = sim_data.Fare.median()
    return fare_col

def fill_missing_ages(features):
    # assuming age is the last feature
    known_age_data = features.loc[features.Age.notnull()]
    known_age_X = known_age_data.drop('Age', axis=1).values
    known_age_y = known_age_data.Age.values
    age_model = RandomForestRegressor()
    age_model.fit(known_age_X, known_age_y)
    unknown_age_data = features.loc[features.Age.isnull()]
    unknown_ages = age_model.predict(unknown_age_data.drop('Age', axis=1).values)
    features.loc[features.Age.isnull(), 'Age'] = unknown_ages
    
def construct_features(data):
    # Ticket Class 
    first_class_feature = data.Pclass.map(lambda c: c == 1).rename('FirstClass').astype(float)
    second_class_feature = data.Pclass.map(lambda c: c == 2).rename('SecondClass').astype(float)
    third_class_feature = data.Pclass.map(lambda c: c == 3).rename('ThirdClass').astype(float)
    
    # Title with Majority Survival Rate
    majority_survived_titles = ['Mrs', 'Miss', 'Master', 'Mme', 'Ms', 'Lady', 'Sir', 'Mlle', 'the Countess']
    majority_title_classify = lambda title: title in majority_survived_titles 
    majority_title_feature = split_names(data).Title.map(majority_title_classify).rename('TitleMajority').astype(float)
    
    # Title with Approximately Half Survival Rate
    half_survived_titles = ['Dr', 'Major', 'Col']
    half_title_classify = lambda title: title in half_survived_titles
    half_title_feature = split_names(data).Title.map(half_title_classify).rename('TitleHalf').astype(float)
    
    # Sex (Female)
    is_female = lambda sex: sex == 'female'
    sex_feature = data.Sex.map(is_female).rename('Female').astype(float)
    
    # Number of Siblings
    sibling_feature = data.SibSp.astype(float)
    
    # Number of Parents and Children
    parch_feature = data.Parch.astype(float)
    
    # Family Size > 1
    family_sizes = data.SibSp + data.Parch + 1
    has_family_feature = family_sizes.map(lambda s: s > 1).rename('HasFamily').astype(float)
    
    # Family Size < 4
    small_family_feature = family_sizes.map(lambda s: s < 4).rename('SmallFamily').astype(float)
    
    # Ticket Fare
    fares = full_fare(data)
    fare_feature = fares.rename('Fare')
    
    # Ticket Fare >= 75 pounds
    expensive_feature = fares.map(lambda f: f >= 75).rename('Expensive').astype(float)
    
    # Has Cabin?
    has_cabin_feature = data.Cabin.notnull().rename('HasCabin').astype(float)
    
    # Favorable Deck
    is_favorable_deck = lambda d: d is not None and d in ['B', 'C', 'D', 'E', 'F']
    deck_feature = data.Cabin.map(is_favorable_deck).rename('FavorableDeck').astype(float)
    
    # Embarked
    embarked_data = full_embarked(data)
    southampton_feature = embarked_data.map(lambda p: p == 'S').rename('Southampton').astype(float)
    queenstown_feature = embarked_data.map(lambda p: p == 'Q').rename('Queenstown').astype(float)
    cherbourg_feature = embarked_data.map(lambda p: p == 'C').rename('Cherbourg').astype(float)
    
    # Southampton Class 3 Female exception
    is_southampton_exception = lambda row: row.Embarked == 'S' and row.Pclass == 3 and row.Sex == 'female'
    s_exception_feature = data.apply(is_southampton_exception, axis=1).astype(float).rename('SouthException')
    
    # Raw Age
    ages = data.Age
    
    raw_features = pd.concat([first_class_feature, second_class_feature, third_class_feature,                              majority_title_feature, half_title_feature, sex_feature,                              sibling_feature, parch_feature, has_family_feature, small_family_feature,                              fare_feature, expensive_feature, has_cabin_feature, deck_feature,                              southampton_feature, queenstown_feature, cherbourg_feature, s_exception_feature,                              ages], axis=1)
    
    fill_missing_ages(raw_features)
    child_feature = raw_features.Age.map(lambda a: a <= 10).rename('Child').astype(float)
    features = pd.concat([raw_features, child_feature], axis=1)
    
    return features
    
construct_features(known_data).head()


# We first try logistic regression, which is commonly used in binary classification problems such as this one. (This model performs L2 regularization with constant C = 1 by default.) Our features seem to be fairly linearly separable, especially the ones corresponding to booleans.

# In[25]:


def logistic_regression(training_data, test_data, reg=1.0):
    train_X = construct_features(training_data).values
    train_y = training_data.Survived.values
    model = LogisticRegression(C=(1.0 / reg))
    model.fit(train_X, train_y)
    test_X = construct_features(test_data)
    return list(model.predict(test_X))

reg_consts = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
best_reg = 0.0
best_acc = 0.0
for reg in reg_consts:
    curr_acc = k_fold_cross_validation(lambda train, test: logistic_regression(train, test, reg=reg))
    print('Logistic Regression with regularization constant %f: %f' % (reg, curr_acc))
    if best_acc < curr_acc:
        best_acc = curr_acc
        best_reg = reg
predict_unknown(lambda train, test: logistic_regression(train, test, reg=best_reg), 'logistic_regression.csv')


# Another classification algorithm is support vector machine (SVM) classification. This method attempts to find a hyperplane that takes advantage of the "gap" between data points of those who survived and data points of those who didn't. Note that we use a linear kernel, as the features are mostly linearly separable.

# In[ ]:


def svm_classification(training_data, test_data, penalty=1.0):
    train_X = construct_features(training_data).values
    train_y = training_data.Survived.values
    model = SVC(kernel='linear', C=penalty)
    model.fit(train_X, train_y)
    test_X = construct_features(test_data)
    return list(model.predict(test_X))

penalty_consts = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
best_penalty = 0.0
best_acc = 0.0
for penalty in penalty_consts:
    curr_acc = k_fold_cross_validation(lambda train, test: svm_classification(train, test, penalty=penalty))
    print('SVM Classification with penalty %f: %f' % (penalty, curr_acc))
    if best_acc < curr_acc:
        best_acc = curr_acc
        best_penalty = penalty

predict_unknown(lambda train, test: svm_classification(train, test, penalty=best_penalty), 'svm_classification.csv')


# Nearly all of our features are weak classifiers on their own (or represemt decision stumps), so we could attempt using boosting models that are ensembles of these weak classifiers. 

# In[27]:


def gradient_boosting(training_data, test_data):
    train_X = construct_features(training_data).values
    train_y = training_data.Survived.values
    model = GradientBoostingClassifier(n_estimators=50)
    model.fit(train_X, train_y)
    test_X = construct_features(test_data)
    return list(model.predict(test_X))

print('Gradient Boosting: %f' % k_fold_cross_validation(gradient_boosting))
predict_unknown(gradient_boosting, 'gradient_boosting.csv')


# In[28]:


def adaboost(training_data, test_data):
    train_X = construct_features(training_data).values
    train_y = training_data.Survived.values
    model = AdaBoostClassifier(n_estimators=50)
    model.fit(train_X, train_y)
    test_X = construct_features(test_data)
    return list(model.predict(test_X))

print('Adaboost: %f' % k_fold_cross_validation(adaboost))
predict_unknown(adaboost, 'adaboost.csv')


# ## Conclusion

# Based on the above analysis, we observed that:
# * Passengers with higher ticket class and fare (and likely socioeconomic status) were significantly more likely to survive than those with lower class and fare, all other factors equal.
# * Females were significantly more likely to survive than males were, all other factors equal.
# * Most passengers under age 10 survived, but age did not otherwise correlate well with survival rate.
# * Ticket fare and presence of a cabin were both correlated with ticket class, so those who paid more fare and those who lived in cabins had higher likelihood of surviving.

# Using this information, we constructed our feature vectors to use in our toolbox of classification algorithms. Using k-fold cross validation accuracies, we saw that logistic regression performed the best on the known training data (highest accuracy of 0.836139), followed by SVM classification (highest accuracy of 0.835017). 

# When run on the unknown data set, SVM performed the best of all the proposed methods, achieving a test accuracy of 0.79426. It did seem that either all of our algorithms overfit the training data to some extent or the distributions of the unknown data are slightly different from those of the known data.

# We certainly have several other features we could consider. For instance, we could attempt to find each passenger's other family members on board and infer some data about the family and individual. We would determine whether the last name (and possible maiden last name) is a good indicator of family relationships. If so, then we would determine whether those family members' ages or sexes influence the original passenger's survival. We could also remove certain outliers in ticket fare before performing our analysis. 
# 
# (I'm open to any suggestions to improve the quality of this analysis, as this is my first stab at a Kaggle data science challenge.)

# ## References

# http://www.imo.org/en/About/Conventions/ListOfConventions/Pages/International-Convention-for-the-Safety-of-Life-at-Sea-(SOLAS),-1974.aspx
# 
# http://www.history.com/this-day-in-history/titanic-sinks

# In[ ]:




