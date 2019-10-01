#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The goal of this notebook is to explore how much information can be gleaned from the often overlooked cabin column.

# ## Import Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ## Load Data

# In[ ]:


titanic_train = pd.read_csv("../input/train.csv", index_col='PassengerId')
titanic_test = pd.read_csv("../input/test.csv", index_col = "PassengerId")


# In[ ]:


train_results = titanic_train["Survived"].copy()
titanic_train.drop("Survived", axis=1, inplace=True, errors="ignore")
titanic = pd.concat([titanic_train, titanic_test])
traindex = titanic_train.index
testdex = titanic_test.index


# ## Early Notes
# In many tutorial notebooks for this competition the cabin column is discarded. There are some very valid reasons for this behavior. Firstly there is an ovewhelming number of missing values:

# In[ ]:


titanic.shape[0]


# In[ ]:


titanic[titanic["Cabin"].isnull()==True].shape[0]


# As you can see, 1014 of the 1309 passengers do not have any cabin information. 
# 
# The second reason that this information is discard is that it contains a large number of unique values.

# In[ ]:


titanic["Cabin"].value_counts()


# So even within the values we have the data is messy. So I think it is a perfectly reasonable decision to throw this column out due to these criticisms. However, this notebook is not to justify throwing the data out but to see if we can glean anything from keeping it. 

# # Digging into the Cabin Data

# From the previous cell it appears as if most of the cabins consist of a single letter at the beginning followed by a 2 or three digit number. It seems logical that the letter would represent the deck or section of boat where the cabin was located followed by the room number. It would seem that if you knew the section of the boat where someone was staying it would give you a lot of insight into their chances of survival. With that in mind let's work on cleaning up that column and seeing what we can get out of it.

# Start by isolating the rooms which have data.

# In[ ]:


cabin_only = titanic[["Cabin"]].copy()
cabin_only["Cabin_Data"] = cabin_only["Cabin"].isnull().apply(lambda x: not x)


# We'll then take just the first character and assign it to a new column named "Deck" and take the any numerical sequence right after this letter and assign it to "room."

# In[ ]:


cabin_only["Deck"] = cabin_only["Cabin"].str.slice(0,1)
cabin_only["Room"] = cabin_only["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
cabin_only[cabin_only["Cabin_Data"]]


# In[ ]:


cabin_only[cabin_only["Deck"]=="F"]


# Looking through the result there are some things which aren't ideal. 
# * Any entries which have a letter space letter form are only returning the first letter. This looks like it often happens when therer is an "F" in that first character slot. however, this only occurs in four total cells so we will ignore it for now.
# * Some entries look like they contain multiple cabins or rooms. For these we are only getting the data for the first one that occurs. However by inspecting the data it appears that in these cases they all share the same deck and the room numbers are all fairly close. So while this may affect our analysis it should be minimal.

# ## One Hot Encoding

# We'll now deal with the missing values and then convert the deck feature a series of one hot encoded columns.

# First we'll drop the Cabin and Cabin_Data columns.

# In[ ]:


cabin_only.drop(["Cabin", "Cabin_Data"], axis=1, inplace=True, errors="ignore")


# Now we'll deal with the missing values. For the deck column we will  replace the null values with an unused letter to represent lack of data. For the room number we will simply use the mean.

# In[ ]:


cabin_only["Deck"] = cabin_only["Deck"].fillna("N")
cabin_only["Room"] = cabin_only["Room"].fillna(cabin_only["Room"].mean())


# In[ ]:


cabin_only.info()


# We will now use one hot encoding on the deck column. 

# In[ ]:


def one_hot_column(df, label, drop_col=False):
    '''
    This function will one hot encode the chosen column.
    Args:
        df: Pandas dataframe
        label: Label of the column to encode
        drop_col: boolean to decide if the chosen column should be dropped
    Returns:
        pandas dataframe with the given encoding
    '''
    one_hot = pd.get_dummies(df[label], prefix=label)
    if drop_col:
        df = df.drop(label, axis=1)
    df = df.join(one_hot)
    return df


def one_hot(df, labels, drop_col=False):
    '''
    This function will one hot encode a list of columns.
    Args:
        df: Pandas dataframe
        labels: list of the columns to encode
        drop_col: boolean to decide if the chosen column should be dropped
    Returns:
        pandas dataframe with the given encoding
    '''
    for label in labels:
        df = one_hot_column(df, label, drop_col)
    return df


# In[ ]:


cabin_only = one_hot(cabin_only, ["Deck"],drop_col=True)


# In[ ]:


cabin_only.head()


# So there we have it 10 columns of data extracted from the origional cabin column.  Now let's see if any of these are of use in predicted survival.

# # Exploring Relationships between Cabin Data and Survivorship

# Let's see if the is any correlation between these columns and surviving.

# In[ ]:


for column in cabin_only.columns.values[1:]:
    titanic[column] = cabin_only[column]


# In[ ]:


titanic.drop(["Ticket","Cabin"], axis=1, inplace=True)


# In[ ]:


corr = titanic.corr()


# In[ ]:


corr["Pclass"].sort_values(ascending=False)


# Ok so this is quite interesting. It does seem that a lack of cabin data is highly correlated with lower class passengers and that decks B and C are fairly correlated with higher class passengers.

# In[ ]:


corr["Fare"].sort_values(ascending=False)


# Here again it appears as if no cabin data is correlated with lower Fare value and decks B and C are correlated with higher Fare values.
# 
# Now let's split our sets back apart.

# In[ ]:


# Train
train_df = cabin_only.loc[traindex, :]
train_df['Survived'] = train_results

# Test
test_df = cabin_only.loc[testdex, :]


# In[ ]:


test_df.head()


# # Testing predictions using only cabin values.

# Let's set up our cabin_only dataset for use by a random forest classifier.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import scipy.stats as st


# In[ ]:


rfc = RandomForestClassifier()


# In[ ]:


X = train_df.drop("Survived", axis=1).copy()
y = train_df["Survived"]


# In[ ]:


param_grid ={'max_depth': st.randint(6, 11),
             'n_estimators':st.randint(300, 500),
             'max_features':np.arange(0.5,.81, 0.05),
            'max_leaf_nodes':st.randint(6, 10)}

grid = RandomizedSearchCV(rfc,
                    param_grid, cv=10,
                    scoring='accuracy',
                    verbose=1,n_iter=20)

grid.fit(X, y)


# In[ ]:


grid.best_estimator_


# In[ ]:


grid.best_score_


# Ok so now let's generate our predictions based on the best estimator model.

# In[ ]:


predictions = grid.best_estimator_.predict(test_df)


# In[ ]:


results_df = pd.DataFrame()
results_df["PassngerId"] = test_df.index
results_df["Predictions"] = predictions


# In[ ]:


results_df.head()


# In[ ]:


results_df.to_csv("Predictions", index=False)


# # Conclusion
# There we have it. Only using a single model we were able to predict whether or not a passenger survived with close to 70% accuracy using a single model and only the data which would normally have been thrown away. Just goes to show that sometimes a lack of data can sometimes be data.
# 
# ## Edit: 
# After submitting the results from this model we get a score of about 68%.
