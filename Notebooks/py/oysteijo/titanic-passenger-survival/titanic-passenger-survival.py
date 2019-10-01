#!/usr/bin/env python
# coding: utf-8

# # Titanic passenger survival analysis
# Here is my first kernel written on kaggle.com. **This is work in progress.** Please upvote and comment
# of you like it. Here is a little outline of the notebook.
# 
# 1. Munging the data
# 2. Feature engineering
# 3. Feature preparation
# 4. Evaluating classifiers
# 5. Submitting
# 
# *Note:* I'm writing this from my parents residence and the internet connection here is really unstable.
# I loose my connection in the funniest moments.
# I'm also trying to make a nice vacation for my two daughters. That has a higher priority than kaggling.
# So this notebook will probably progress slowly towards something useful.

# ## Munging the data
# I will start by munging the data a bit. Munging data or data wrangling is the process
# of handling the raw data such that it works for the later analysis. The munging is
# often an important part of the data scientist's work.

# ### Loading the data.
# I will load the data into two pandas dataframes. The training dataset will be loaded into titanic,
# the test dataset will be loaded into a dataframe called test. I will also merge the two into a
# dataframe called full, as the statistics
# based on the both sets, will be used to make some estimations for missing data. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


titanic = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
full = pd.concat([titanic, test])


# In[ ]:


titanic.info()
print("-"*40)
test.info()


# ### Filling out some missing values
# #### Filling out the missing embarkment
# I looks like we miss a embarkment port for two passengers in titanic dataframe. Let's not make
# a big thing about this, and assume that the the passengers embarked in Southampton. That is natural
# enough as most passengers embarked in Southampton. I also don't believe that the embarkment port
# is a real big indicator to predict survival. **Southampton it is!**

# In[ ]:


# just to show the numbers
full.Embarked.value_counts()


# In[ ]:


# or maybe even more convincing: A plot!
full.Embarked.value_counts().plot(kind='bar')


# In[ ]:


# See? It's pretty safe two assume they came for Southampton
titanic.Embarked.fillna(value='S', inplace=True)
# Let's update the full dataframe as well:
full.Embarked.fillna(value='S', inplace=True)


# ### Filling out the missing fare
# There is only one passenger where the fare is missing:

# In[ ]:


test[np.isnan(test["Fare"])]


# I will fill this in with the median fare of the passengers for that
# embarkment port (S) on that passenger class (Third)

# In[ ]:


# Mr. Storey from Southampton in third class.
test.loc[test["PassengerId"] == 1044, "Fare"] = full[(full["Embarked"]=='S') & (full["Pclass"]==3)].Fare.median()
test.loc[test["PassengerId"]==1044,:]


# 8.05 GBP from Mr. Storey! Sounds good enough to me!
# ### Filling out the missing ages
# I must admit that I have read Megan's notebook. She uses a MICE regression to fill in the missing ages.
# I really think that is a great idea, but I will settle for a simpler method in this notebook, as I don't
# believe the age is really the strongest indicator. I also want to keep it simple as this is my very
# first kernel on kaggle.
# 
# (I'm at vacation, visiting my parents and the internet connection is a bit unstable here.... I also get a lot of
# interruptions of different kinds, so I may use a few days for this notebook.)
# 
# #### Age filling strategy
# There was a "Women and children first" policy for filling up the lifeboats, so the important thing
# to consider is if a passenger with missing age was considered a child or adult. So to fill in the
# missing age data we will use the title of the person, and fill in the median age for the given title.
# 
# It is important to understand that someone with title **Master** is typically a young boy who will be
# considered a child. **Master** was used address politely a boy who was too young to be called **Mister**.
# Also, a **Miss** is typically unmarried and younger than a **Mrs**, so hopefully the strategy will help
# us get good values for the missing ages.
# 
# So the first step will be to make an additional column with the *Title*, in all three dataframes. Megan
# has already showed us that the title of a passenger is actually an important predictor. So we will add the
# column and keep it there for later.

# In[ ]:


# So let's add a title column to each DataFrame
# we also make a list of all our frames, such that we can easily loop over them.
frames = [titanic, test, full]
for df in frames:
    df["Title"] = df.Name.str.replace('(.*, )|(\\..*)', '')


# In[ ]:


full.head()


# Looks like that worked fine! Let's see how many unique titles there are, and how many
# unique titles we need to fill in the age.

# In[ ]:


# That went good. There are 18 unique titles.
full["Title"].value_counts()


# In[ ]:


# Let's check which titles that are missing age data
full[np.isnan(full["Age"])].Title.unique()


# OK. There are only 6 types of titles that is missing age data. The thing that comes to attraction is that 
# there is both a **Miss** and a **Ms** title. Without much considerations, I'm joining the **Ms** titled
# with **Miss** titled. Then it's only 5 titles to fill with age.

# In[ ]:


# Hmmm what's the difference of Miss and Ms?
full[full["Title"]=="Ms"]


# In[ ]:


# Let's just set the two Ms to Miss. Can't be that bad.
for df in frames:
    df.loc[df["Title"]=="Ms", "Title"] = "Miss"


# Yes! Then we are ready for the real filling of all missing ages.

# In[ ]:


# So here is the main juice. Assign the missing age to the median age with the given title.
for t in full[np.isnan(full["Age"])].Title.unique():
    for df in frames:
        df.loc[(df["Title"]==t) & np.isnan(df["Age"]), "Age" ] = full[full["Title"]==t].Age.median()


# ### Filling out the missing cabin
# There are so many missing cabins in the dataset that we will not even try to fill in the missing elements.
# We will however try to extract some information from this later in this notebook.

# ## Feature engineering
# We do indeed already have some features, but I think we should try to extract some more. We have the
# passenger class, the sex, the age, #siblings_and_spouses, and #childern_and_parents, the embarkment port
# and the fare. We've also extracted a *Title* from the name column. However, I think we can gain some more.
# Let's consider the cabin first.
# ### Engineering a cabin feature
# Important to note: The way to survive is to get onboard a lifeboat. At the RMS Titanic there were
# 20 lifeboats, however only 18 were used. There were less capacity of the lifeboats than the number of
# passengers and crew onboard. Numbers from wikipedia says there was a capacity of 1178 passengers in the
# lifeboats. There were 2224 on board (crew included). Since there was not enough capacity of the lifeboats
# to evacuate everyone, the "Women and children first" policy were applied. 
# 
# #### A side note on the low capacity of the lifeboats
# We may think today that it was really strange to have less capacity of lifeboats than number of
# passengers and crew. However, the number of lifeboats was well within the maritime laws and regulations
# of the time. The next question that then comes up is: Why was the maritime regulation not requiring
# any liner to have lifeboat capacity for every passenger and crew member? The answer to this is that
# there never were any concern that all passengers had to be evacuated over a relatively short period of
# time. A ship in distress would probably stay afloat for many hours. First of all there was a lot of
# maritime traffic of those days, and a liner in distress would always be able to call for assistance
# from a nearby vessel and evacuate the personnel to that vessel. The vessels were equipped with
# wireless telegraphs. That was the common philosophy of the time, and that was the reason why
# this was not considered a problem. After all, Titanc was the ship that could not sink. It was called
# the ship that could not sink due to the double bottom and the watertight bulkheads. However the
# bulkheads was not sealed with a ceiling, so when each bulkhead was filled with water, the water
# simply flooded over to the next bulkhead. She was hence not so unsinkable after all.
# 
# #### A feature for the cabin side
# This is indeed a bit interesting. The "Women and children first" policy was indeed enforced differently on
# the port and starboard side of the ship, so the side of the cabin could be a useful predictor in our
# analysis. Cabin numbers ending with an odd number indicates that the cabin was on starboard side, while
# cabins ending on an even number were located on the port side. Let's create a feature called
# **CabinSide** that takes the values **unknown**, **starboard** or **port**.

# In[ ]:


# Let's look at the cabin a bit. This might be important. The "Women and Children" policy was enforced differently on
# starboard and port side. Odd numbered cabins are starboard side, and even numbers are port side.
for df in [titanic, test]:
    df["CabinSide"] = "Unknown"
    df.loc[pd.notnull(df["Cabin"]) & df["Cabin"].str[-1].isin(["1", "3", "5", "7", "9"]),"CabinSide"] = "Starboard"
    df.loc[pd.notnull(df["Cabin"]) & df["Cabin"].str[-1].isin(["0", "2", "4", "6", "8"]),"CabinSide"] = "Port"


# We need some cleanup. The Ryersons had four cabins, three on starboard and one on port. It is natural
# to set them all on starboard, as they probably gathered. They traveled with ticket **PC 17608**, so
# we use that to index the rows of Ryersons & co.

# In[ ]:


for df in [titanic, test]:
    df.loc[df["Ticket"]=="PC 17608", "CabinSide"] = "Starboard"


# It is also natural to assume that Bowen, Miss. Grace Scott was in cabin B68. According to sources
# she was the maid for the Ryersons and the deck plan drawing shows this cabin as a maids/servant cabin.

# In[ ]:


test.loc[test["Name"].str.contains("Bowen,"),"Cabin"] = "B68"


# In[ ]:


titanic.CabinSide.value_counts()


# #### A feature for the cabin deck
# Lower deck cabins where flooded with water before the higher level deck. Is it natural to think that
# passengers on low decks gathered to the lifeboats earlier than the passengers at the higher decks?
# At least I will try out a feature based on the deck. The deck is labeld as the first letter in the
# cabin number. A-G, where A is the highest and G is the lowest.

# In[ ]:


# Maybe the Deck is important? who knows?
for df in [titanic, test]:
    df["Deck"] = "Unknown"
    df.loc[pd.notnull(df["Cabin"]), "Deck"] = df["Cabin"].str[0]


# We need some cleanup as some cabins are numbered "F Gxx". I am not sure what this means, but
# I guess it means "Fore" deck "G" cabin "xx". I have asked this on the forum, but I have got no replies
# yet.

# In[ ]:


titanic.loc[titanic.Cabin.str.len() == 5,:]


# In[ ]:


for df in [titanic, test]:
    df.loc[pd.notnull(df["Cabin"]) & (df.Cabin.str.len() == 5), "Deck"] = df["Cabin"].str[2]


# In[ ]:


test.loc[test.Cabin.str.len() == 5,:]


# Yes! That looks better.

# In[ ]:


# Test if there is some strange decks as well. Deck T ??
titanic.Deck.value_counts()


# Yes, there is a deck T? What is that?

# In[ ]:


titanic.loc[titanic["Deck"] == 'T',:]


# I have no idea what cabin T is supposed to mean, so I set this to unknown. (I've also checked test set, but
# there is no passenger with cabin T in the test part.)
# 

# In[ ]:


titanic.loc[titanic["Deck"] == 'T',"Deck"] = "Unknown"


# ### A family size feature.
# I really don't believe that this gains better classification than the *SibSp* and *ParCh* features
# separated, since they are linear correlated. I see that a lot of other scripts has this featere, so
# I'm adding this for the fun of it.

# In[ ]:


# Let's define another feature. FamilySize = Parch + SibSp + 1
for df in frames:
    df["FamilySize"] = df.Parch + df.SibSp + 1


# (Let me continue. I've had a beautiful day with my family at Oscarsborg yesterday, and Friday we also
# fantastic day fishing. We caught three atlantic mackerel and had them for dinner.) 
# ### Ticket group size feature
# The ticket looks like they were sold in groups. Can the size of the ticket group be used as an indicator?
# That will catch other relations than the family relations on ParCh and SibSp, like valet, servant, maid etc.
# This may or may not improve the predictions. Such a feature will of course be strongly related to
# family size, and may therefore not improve the predictions as much as I hope. Also, there are some tickets
# that are issues with sequential numbers even for groups traveling together.
# 
# There is possibly a simple way to do this in Pandas, but as a long time Python hacker, I will do this the
# Python way. I will create a python dictionary with the ticket number as the key, and the number of
# tickets with that key.

# In[ ]:


# Ticket group size
# first we make a dictionary
ticket_dict = {}
for t in full.Ticket.unique():
    ticket_dict[t] = 0
for t in full.Ticket:
    ticket_dict[t] += 1

# Then we apply it to the dataframes
for df in frames:
    df["TicketGroupSize"] = df["Ticket"].apply( lambda x: ticket_dict[x])


# ### Ticket group survivors feature
# Here is my last predictor for today. For each ticket group, we will count the other survivors in that
# group. If there is a *everybody or nobody* connection in the traveling group, this may be a good predictor.
# Also, a non-linear classifier may also be able to find relations like *everybody except the adult male*
# connections in the data. That is why I hope this feature can gain something.
# 
# I will reuse and overwrite the same dictionary and use the same Python method to add this feature. Again,
# those who know Pandas well can probably do this simpler. I'm a Pandas beginner myself. 
# 

# In[ ]:


for t in full.Ticket.unique():
    ticket_dict[t] = 0
for row in full.iterrows():
    t = row[1]["Ticket"]
    if row[1]["Survived"] > 0.1:
        ticket_dict[t] += 1
        
# Then we apply this to the dataframes
for df in [titanic, test]:
    df["TicketGroupSurvivors"] = df["Ticket"].apply( lambda x: ticket_dict[x])


# I really hope that this feature can gain something, however it may be biasing a prediction towards
# death to singleton traveling passengers. A singleton passenger (in this case a passenger with a unique
# ticket number) will have either 0 ot 1 in this feature. The passenger will always have 0 as 
# TicketGroupSurvivors in the test set, and a classifier will train to predicting this as a non-survivor,
# This feature can therefore may be really bad, rather than smart. I don't know, I'll try it out.
# 
# **Update:** After trying a few classifiers, I realize that this really happens. The remedy could be to subtract one form TicketGroupSurvivors where a singleton survived.
# 
# 

# In[ ]:


# Here is the problem:
titanic.loc[titanic["TicketGroupSize"] == 1, ["Survived", "TicketGroupSurvivors"]]


# In[ ]:


# Here is the remedy:
titanic.loc[titanic["TicketGroupSize"] == 1, ["Survived", "TicketGroupSurvivors"]] = 0


# ### Other features
# At the top of my head I can think of a few other features to add. First of all, if the TicketGroupSize and
# the TicketGroupSurvivors features are fruitful, we might add a two similar features based on the surname of the passengers.
# 
# Another feature to consider from the cabin column is to see if the cabin is closer to the bow or the
# stern of the ship. From the schematic plan on [wikipedia](https://en.wikipedia.org/wiki/Lifeboats_of_the_RMS_Titanic),
# we see that for each side there is four lifeboats by the bow and five by the stern. Also the ship sank
# slowly for the first hour with the bow first, so where the cabin was (by the bow or stern) may be a useful
# predictor. I'll see if I can add this later.

# ## Feature preparation
# To be able to use the features, we have to convert them into numeric values. Some of our features are
# already numerical. It is often a good idea to normalize those features. Some of the features
# are categorical. For categorical features, we simply make indicator (dummy) features.
# 
# For noramlization, I will create a helper function. (I guess scikit learn already has one, but I'm not
# familiar with that one yet.) We will normalize by the function $$\frac{X - \mu}{\sigma}$$ The
# implementation goes like this:

# In[ ]:


# Normalizer
def normalize(feat):
    mean = full[feat].mean()
    stdv = full[feat].std()
    for df in [titanic,test]:
        df[feat + "_norm"] = (df[feat] - mean) / stdv


# Note that we use the full set (combined titanic and test) to estimate the mean and standard deviation.
# ### Features to normalize
# Some features should be normalized, others not. Let's discuss.
# #### Age
# Age is already a numeric value. This makes more sense to normalize, at the values are much higher
# than other numeric features.
# **(Note to myself: Check out sklearn preprocessing
# #### SibSp, ParCh and Family size.
# Just plain normalize all of these. An alternative to be to group then into categories, but let's wait
# with that.
# #### Fare
# This is interesting. Is the fare of the ticket based on the ticket group size? Maybe the fare
# should be divided by the size of the ticket group size and then normalized? Maybe even this should
# be adjusted and normalized to the fare based on the given class. Let's investigate that later,
# and just do a plain normalization for now.
# #### Passenger Class (Pclass)
# This is indeed a numeric value, but we would rather consider this categorical feature than numerical.
# Let's *not* normalize this feature.
# #### TicketGroupSize
# Just normalize this.

# In[ ]:


# Age, SibSp, ParCh, FamilySize, Fare and TicketGroupSize. Those are the ones.
[normalize(x) for x in ["Age", "SibSp", "Parch", "FamilySize", "Fare", "TicketGroupSize"]]


# In[ ]:


titanic.head()


# Looks like it works!
# 
# ### Categorical features. 
# #### Sex
# Let's begin with **Sex** since it's the first thing that comes to my mind (no pun intended). Sex is a
# categorical value, but it can only take two different values: **female** or **male**. Note that is
# should not be necessary to have two features, with one **Sex_male** and another **Sex_female**, as
# that will just make to directly linear correlated values. This actually applies to all categorical
# features. I will therefore drop the most populated category. For sex, the baseline category will
# be male, and hence sex_male will be dropped.
# #### Passenger class (Pclass)
# I will we can treat this as categorical data. I think I'll handle this as categorical data. I will drop
# Pclass_3 as the baseline category.
# #### Embarkment port
# Let's just make a categorical inputs for each port. I'll drop Embarked_S as baseline.
# #### CabinSide
# Plain categorical. Starboard and Port, I will drop CabinSide_unknown as the baseline case.
# #### CabinDeck
# I will also do categories on the deck even though a plain numerical value could be considered.
# Deck_Unknown will be dropped as baseline.
# #### Title
# Magan have already showed us that title is an important feature. However we need to reduce number of
# titles. Some of the titles are rare and should not have their own category. I suggest these title
# categories: **Mr**, **Mrs**, **Miss**, **Master**, **Rev**, **Officer**, **Royal**.

# In[ ]:


full["Title"].value_counts()


# In[ ]:


# These can be discussed, of course.
titledict = {"Dr"   : "Mr",
             "Col"  : "Officer",
             "Mlle" : "Miss",
             "Major": "Officer",
             "Lady" : "Royal",
             "Dona" : "Royal",
             "Don"  : "Royal",
             "Mme"  : "Mrs",
             "the Countess": "Royal",
             "Jonkheer": "Royal",
             "Capt" : "Officer",
             "Sir"  : "Mr"
             }
#There is probably a pandas way to do this, however I do it the python way today.
for df in frames:
    for key,val in titledict.items():
        df.loc[df["Title"]==key, "Title"] = val


# In[ ]:


full["Title"].value_counts()


# Then we have only seven titles. Let's create the indicator variables.

# In[ ]:


category_list = ["Pclass", "Sex", "Embarked", "Title", "CabinSide", "Deck"]
titanic = pd.get_dummies(titanic, columns=category_list)
test = pd.get_dummies(test, columns=category_list)


# #### TicketGroupSurvivors
# Maybe this can be divided by the size of the group?

# In[ ]:


for df in [titanic, test]:
    df["TGS_norm"] = df["TicketGroupSurvivors"] / df["TicketGroupSize"]


# In[ ]:


titanic.columns


# In[ ]:


test.columns


# Enough features for today! Instead of dropping the unwanted columns that will not be used in the
# classifier, I will rather make a new dataframe for training and testing.

# ## Evaluating classifiers
# Let's try a few different classifier and use cross validation to find the one we like the most. I suggest
# we try Naive Bayes, KNN, Logistic Regression and Random Forest. It would be übercool if we also could try
# a simple neural network to classify, but let's try the other first.
# 
# ### The dead simple classifier
# Before we do anything at all, we have to just make a dead simple classifier. I call this dead simple,
# because it predicts everyone dead. It can't be much simpler than that. Such preductor will of course
# make the wrong prediction for every passenger that actually survived. It is nice to have such a reference
# submission to know if we are doing good. 

# In[ ]:


# What should we expect if a predictor predicting all dead.
1 - titanic.Survived.mean()


# In[ ]:


# Let's see the real.
ds_submission = pd.DataFrame(test["PassengerId"])
ds_submission["Survived"] = 0  # All dead


# In[ ]:


# This is actually the simplest predictor I can imagine so for the fun of it, let's submit this
# and see how it scores
ds_submission.to_csv("all_dead.csv", index=False)


# Submission says: *Your submission scored **0.62679**, which is not an improvement of your best score.* Keep trying!
# It's actuallly better than the **0.616** estimated.
# 
# We can actually use this result to say how many survivors there should be in the test set. That can be
# useful information in the fine tuning of the model.

# In[ ]:


round(len(test) * 0.62679)


# So there should be about 262 dead passengers in the test set. We do not know which passengers are
# creating the 0.62679 score, but at least we know the ballpark.

# ### Gaussian Naive Bayes
# (To be done later)
# ### K Nearest Neighbors classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


k_range = range(1, 31)
param_grid = dict(n_neighbors=list(k_range),weights = ["uniform", "distance"])


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')


# In[ ]:


titanic.columns


# In[ ]:


features = ['Age_norm', 'SibSp_norm', 'Parch_norm',
       'FamilySize_norm', 'Fare_norm', 'TicketGroupSize_norm', 'Pclass_1',
       'Pclass_2', 'Sex_female', 'Embarked_C',
       'Embarked_Q', 'Title_Master', 'Title_Miss', 
       'Title_Mrs', 'Title_Officer', 'Title_Rev', 'Title_Royal',
       'CabinSide_Port', 'CabinSide_Starboard', 'Deck_A',
       'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G',
       'TGS_norm']


# In[ ]:


len(features)


# In[ ]:


grid.fit(titanic[features], titanic.Survived)


# In[ ]:


grid.best_score_


# In[ ]:


grid.best_params_


# No! 0.935 is way better than I expected. This is *too good*! I don't believe this. I believe that this is
# a matter of the TGS_norm feature being highly correlated to survived in the train set. Let's try to submit
# this and see and check this theory.

# In[ ]:


# rerun fit() with the best parameters with the entire set (no CV)
knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
knn.fit(titanic[features], titanic.Survived)


# In[ ]:


knn_predictions_with_TGS = knn.predict(test[features])


# In[ ]:


knn_predictions_with_TGS.sum()


# Only 102. Far from the 262 we estimated. Makes me believe the TGS theory is right. But let's submit
# anyway.

# In[ ]:


knn_submission1 = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": knn_predictions_with_TGS
        })
knn_submission1.to_csv("knn_predictions_with_TGS.csv", index=False)


# On the leaderboard this scores: 0.72727. Not really impressive and far far from the 0.934 we found
# in cross validation. If the TGS_norm feature is bad for the singleton passengers, let's try to
# redo the above steps w/o that feature. 

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5)
# Fit once more without TGS_norm
grid.fit(titanic[features[:-1]], titanic.Survived)
print(grid.best_score_ , grid.best_params_)


# This number looks more like something I could believe. Let's try to submit this.

# In[ ]:


# rerun fit() with the best parameters with the entire set (no CV)
knn = KNeighborsClassifier(n_neighbors=8, weights='distance')
knn.fit(titanic[features[:-1]], titanic.Survived)


# In[ ]:


knn_predictions_wo_TGS = knn.predict(test[features[:-1]])
knn_predictions_wo_TGS.sum()


# Hmmmm... only 103.... that's still a bit off. Let's submit anyway.

# In[ ]:


knn_submission2 = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": knn_predictions_wo_TGS
        })
knn_submission2.to_csv("knn_predictions_wo_TGS.csv", index=False)


# Well ... 0.77990 (?). Not very impressive, but still an improvement. Can we improve from here?
# 
# I wonder if the two submissions can be combined? We could either do a OR-operation of the two submissions, or we could check the ticket group size to see if we should use the one or the other. Let's try both these. First the OR-operation:

# In[ ]:


knn_submission3 = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": np.logical_or(knn_predictions_with_TGS, knn_predictions_wo_TGS).astype('int') 
})
knn_submission3.to_csv("knn_predictions_or_TGS.csv", index=False)


# *Your submission scored 0.76077*. Worse than the previous. Let's try the other strategy:

# In[ ]:


df_tmp = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "TicketGroupSize": test["TicketGroupSize"],
    "s_with_tgs": knn_predictions_with_TGS,
    "s_wo_tgs": knn_predictions_wo_TGS
})
df_tmp.loc[df_tmp.TicketGroupSize == 1, "Survived"] = df_tmp["s_wo_tgs"]
df_tmp.loc[df_tmp.TicketGroupSize != 1, "Survived"] = df_tmp["s_with_tgs"]
df_tmp.drop(["TicketGroupSize", "s_with_tgs", "s_wo_tgs"], axis=1, inplace=True)
df_tmp.to_csv("knn_predictions_specified_TGS.csv", index=False, float_format='%.f')


# Even worse! Yuck... I'll try something else then. I see a lot of submissions have success with RandomForest. Let me try that.

# ### Random Forest classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


n_range = range(10, 100, 10)
param_grid = dict(n_estimators=list(n_range),criterion = ["gini", "entropy"])
rfc = RandomForestClassifier(n_estimators=20)
grid = GridSearchCV(rfc, param_grid, cv=10, scoring='accuracy')
grid.fit(titanic[features], titanic.Survived)
print(grid.best_score_ , grid.best_params_)


# That took a really long time.....

# In[ ]:


rfc = RandomForestClassifier(n_estimators=60, criterion='entropy')
rfc.fit(titanic[features], titanic.Survived)


# In[ ]:


pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": rfc.predict(test[features])
        }).to_csv("rfc_predictions.csv", index=False)


# In[ ]:




