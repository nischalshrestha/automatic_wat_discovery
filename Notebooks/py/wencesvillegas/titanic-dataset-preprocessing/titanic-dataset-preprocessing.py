#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset Preprocessing
# 

# First of all let's import basic libraries to load, edit and visualize the dataset

# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# Now lets load the csv file for the training and test set.

# In[44]:


training_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
training_set.head()


# Here we can see that we have many categorical features and some numeric ones too. Before turning this dataset into vectors of numbers that our classification algorithms can use, we should deal with missing values.
# Let's check how many missing values has our dataset per feature.

# In[45]:


training_set.isnull().sum()


# In[46]:


test_set.isnull().sum()


# We could delete the training samples that have NaN values but in this case we dont have a huge dataset.
# First we are going to transform the Cabin feature into a Deck feature, each cabin starts with a letter that denotes the deck and we dont really need more information than that.

# In[47]:


# make a list of all the posible Decks, the last element is used when no cabin code is present
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
#define a function that replaces the cabin code with the deck character
def search_substring(big_string, substring_list):
    for substring in substring_list:
        if substring in big_string:
            return substring
    return substring_list[-1]


# We have a similar problem with the Name feature, we have too much information that is hard to encode and nt useful. So we can take only the title of the name for each person, lets define a function for that.

# In[48]:


# replace passenger's name with his/her title (Mr, Mrs, Miss, Master)
def get_title(string):
    import re
    regex = re.compile(r'Mr|Don|Major|Capt|Jonkheer|Rev|Col|Dr|Mrs|Countess|Dona|Mme|Ms|Miss|Mlle|Master', re.IGNORECASE)
    results = regex.search(string)
    if results != None:
        return(results.group().lower())
    else:
        return(str(np.nan))


# In[49]:


# dictionary to map to generate the new feature vector
title_dictionary = {
    "capt":"Officer", 
    "col":"Officer", 
    "major":"Officer", 
    "dr":"Officer",
    "jonkheer":"Royalty",
    "rev":"Officer",
    "countess":"Royalty",
    "dona":"Royalty",
    "lady":"Royalty",
    "don":"Royalty",
    "mr":"Mr",
    "mme":"Mrs",
    "ms":"Mrs",
    "mrs":"Mrs",
    "miss":"Miss",
    "mlle":"Miss",
    "master":"Master",
    "nan":"Mr"
}


# Now that we have the functions we need, lets apply them and create the features Title and Deck

# In[50]:


training_set['Deck'] = training_set['Cabin'].map(lambda x: search_substring(str(x), cabin_list))
test_set['Deck'] = test_set['Cabin'].map(lambda x: search_substring(str(x), cabin_list))
# delete the Cabin feature
training_set.drop('Cabin', 1, inplace=True)
test_set.drop('Cabin', 1, inplace=True)

training_set['Title'] = training_set['Name'].apply(get_title)
test_set['Title'] = test_set['Name'].apply(get_title)
training_set['Title'] = training_set['Title'].map(title_dictionary)
test_set['Title'] = test_set['Title'].map(title_dictionary)
# delete the Name feature
training_set.drop('Name', 1, inplace=True)
test_set.drop('Name', 1, inplace=True)


# Let's take a look at the results we got

# In[51]:


training_set.tail()


# Now we will drop the Ticket feature that does not really give much insight.

# In[52]:


#dropping ticket column
training_set.drop('Ticket', 1, inplace=True)
test_set.drop('Ticket', 1, inplace=True)


# We have to do something about the NaN values in the Age column. We can replace them with the mean of the age, but that would mean that some kid (Master or Miss) would appear to be older than they are. So we will take the mean of the age from each Title, and then replace each NaN value with the mean of the age of the corresponding persons title.

# In[53]:


means_title = training_set.groupby('Title')['Age'].mean()


# In[54]:


title_list = ['Mr','Miss','Mrs','Master', 'Royalty', 'Officer']
def age_nan_replace(means, dframe, title_list):
    for title in title_list:
        temp = dframe['Title'] == title #extract indices of samples with same title
        dframe.loc[temp, 'Age'] = dframe.loc[temp, 'Age'].fillna(means[title]) # replace nan values for mean
        

age_nan_replace(means_title, training_set, title_list)
age_nan_replace(means_title, test_set, title_list)


# Now lets fill those two NaN cases in the Embarked column.

# In[55]:


training_set.groupby('Embarked').size().plot(kind='bar')


# We will assgn them the letter S which is the most common case.

# In[56]:


training_set['Embarked'].fillna('S', inplace=True)
test_set['Embarked'].fillna('S', inplace=True)
#fill the fare column in the test set
test_set['Fare'].fillna(test_set['Fare'].mean(), inplace=True)


# In[57]:


training_set.head()


# Let's visualize some aspects of the data so that we can understand what are the most important factors that determined survival.

# In[58]:


index = training_set['Survived'].unique() # get the number of bars
grouped_data = training_set.groupby(['Survived', 'Sex']) 
temp = grouped_data.size().unstack() 
women_stats = (temp.iat[0,0], temp.iat[1,0])
men_stats = (temp.iat[0,1], temp.iat[1,1])
p1 = plt.bar(index, women_stats)
p2 = plt.bar(index, men_stats, bottom=women_stats)
plt.xticks(index, ('No', 'Yes'))
plt.ylabel('Number of People')
plt.xlabel('Survival')
plt.title('Survival of passengers')
plt.legend((p1[0], p2[0]), ('Women', 'Men'))
plt.tight_layout()


# In[59]:


training_set.pivot_table('Survived',index='Sex',columns='Pclass').plot(kind='bar')


# In[60]:


training_set.pivot_table('Survived', index='Title', columns='Pclass').plot(kind='bar')


# In[61]:


age_intervals = pd.qcut(training_set['Age'], 3)
training_set.pivot_table('Survived', ['Sex', age_intervals], 'Pclass').plot(kind='bar')


# In[62]:


parch_intervals = pd.cut(training_set['Parch'], [0,1,2,3])
sibsp_intervals = pd.cut(training_set['SibSp'], [0,1,2,3])
training_set.pivot_table('Survived', parch_intervals, 'Sex').plot(kind='bar')


# In[63]:


training_set.pivot_table('Survived', sibsp_intervals, 'Sex').plot(kind='bar')


# What we can take from this analysis is that Passenger Class was relevant to survive, and that the features SibSp and Parch behave similarly. We can make a new feature called Family Size that is the sum of those 2 feature columns. 

# In[64]:


training_set['Family Size'] = training_set['Parch'] + training_set['SibSp']
test_set['Family Size'] = test_set['Parch'] + test_set['SibSp']
training_set.drop('Parch', axis=1, inplace=True)
training_set.drop('SibSp', axis=1, inplace=True)
test_set.drop('Parch', axis=1, inplace=True)
test_set.drop('SibSp', axis=1, inplace=True)
training_set.head()


# Let's standardize the numerical features Age and Fare

# In[67]:


from sklearn.preprocessing import StandardScaler

numericals_list = ['Age','Fare']
for column in numericals_list:
    sc = StandardScaler(with_mean=True, with_std=True)
#    print(training_set[column].size)
#    print(test_set[column].size)
    sc.fit(training_set[column].values.reshape(-1,1))
    training_set[column] = sc.transform(training_set[column].values.reshape(-1,1))
    test_set[column] = sc.transform(test_set[column].values.reshape(-1,1))


# Now let's encode categorical classes with sklearn's LabelEncoder

# In[68]:


from sklearn.preprocessing import LabelEncoder
categorical_classes_list = ['Sex','Embarked','Deck', 'Title'] #Pclass is already encoded
#encode features that are cateorical classes
encoding_list = []
for column in categorical_classes_list:
    le = LabelEncoder()
    le.fit(training_set[column])
    encoding_list.append(training_set[column].unique())
    encoding_list.append(list(le.transform(training_set[column].unique())))
    training_set[column] = le.transform(training_set[column])
    test_set[column] = le.transform(test_set[column])


# In[69]:


# lets see the results
training_set.head()


# Now we are going to onehot encode categorical features such as Embarked, Title and Pclass

# In[70]:


training_set = pd.get_dummies(training_set, columns=['Embarked','Pclass','Title', 'Deck'])
test_set = pd.get_dummies(test_set, columns=['Embarked','Pclass','Title', 'Deck'])


# The test set lacks a sample where deck 7 is selected so we will have to align the dataframes to fill that column. 

# In[73]:


training_set, test_set = training_set.align(test_set, axis=1)
test_set.drop('Survived', axis=1, inplace=True)
test_set.fillna(0, axis=1, inplace=True)


# Now that we have our dataset clean and ready, we need to transform it into a numpy matrix that a learning algorithm can use.
# We will get a Y vector containing the labels for training, a matrix X that has all the features for training, and X_test that has all the samples for training from the test set.

# In[74]:


#test_set.fillna(0, inplace=True)
y = training_set['Survived'].values
X = training_set.drop(['Survived','PassengerId'], axis=1).values
X_test = test_set.drop('PassengerId', axis=1).values


# In[ ]:




