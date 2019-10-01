#!/usr/bin/env python
# coding: utf-8

# Introduction
# ================
# Welcome to my analysis on who died during the sinking of the Titanic. In this notebook I will be exploring some basic trends to see what are the best predictors of who survived and who perished and discuss how certain methods aid in my final result.
# 
# A submission worthy script can be found on my [GitHub][1].
# 
# 
#   [1]: https://github.com/kmiller96/TitanicData

# 1. Import In The Data
# ==========================
# Before we can do anything with the analysis we need to import in the data in order to visualise it and identify possible trends. To do this I decided to import in the csv using pandas' read_csv function and import the data into a dataframe.

# In[ ]:


# IMPORT STATEMENTS.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualisation
import matplotlib.pyplot as plt # data visualisation

import random # Used to sample survival.

# DEFINE GLOBALS.
NUM_OF_ROLLS = 3

df = pd.read_csv("../input/train.csv", index_col=0)
df.head()


# As you should be able to see, there are a few different parameters we can consider in our models:
# 
#  - **Pclass:** The ticket class which is either 1, 2 or 3. Although I can't support this yet, I would assume that this be a major deciding factor in if someone survived.
#  - **Name:** The name of the passenger for their id.
#  - **Sex** Either male or female. Again I would assume this is a heavily weighted variable in determining if a person survived or died.
#  - **Age:** This is a floating-point number. If the age is estimated then it is given in the form xx.5 and infants ages are given as fractions of one whole year.
#  - **SibSp:** Indicates the number of siblings or spouses on-board.
#  - **Parch:** Indicates the number of parents or children on-board.
#  - **Ticket:** Gives the ticket number. I think this wouldn't be very useful in the analysis.
#  - **Fare:** The cost of the ticket. This should be pretty well correlated with the ticket class.
#  - **Cabin:** Tells where the ticket holder's cabin is. This might be more useful in higher-level analysis when certain areas had higher mortality rates then others.
#  - **Embarked:** Where the passenger embarked from. There are only three ports to consider: Cherbourg, Queenstown and Southampton

# 2. Visualise The Data
# =========================
# Before we can begin to analyse the data and form an appropriate model, we need to identify trends visually. This section looks at some common trends to see where we can develop our model.

# 2.1. Visualisation of Deaths
# ----------------------------------
# Let us start by looking at how many people died on the Titanic generally and then by category. This might provide us with some interesting insights in how your likelihood of surviving depends on certain parameters.

# In[ ]:


# First visualise the general case (i.e. no considerations)
total, survivors = df.shape[0], df[df.Survived==1].shape[0]
survival_rate = float(survivors)/float(total)*100

f, ax = plt.subplots(figsize=(7, 7))
ax.set_title("Proportion of People Who Died On The Titanic")
ax.pie(
    [survival_rate, 100-survival_rate], 
    autopct='%1.1f%%', 
    labels=['Survived', 'Died']
)
None # Removes console output


# From this visualisation you can see that only 40% of people survived on the Titanic. Thus if we assume that the testing data is similar to the training script (which is a very valid assumption) then we could simply predict that everyone died and we would still be right about 60% of the time. So already we can beat random chance! But we should be able to determine a better model then that anyway.

# The next visualisation I want to do is the likelihood of surviving based on gender and ticket class. I have a feeling that women survived more likely then men and the higher class tickets had a better survival rate then the lower classes.

# In[ ]:


sns.set_style('white')


# In[ ]:


f, ax = plt.subplots(figsize=(8, 8))
sns.barplot(
    ax=ax,
    x='Pclass',
    y='Survived',
    hue='Sex',
    data=df,
    capsize=0.05
)
ax.set_title("Survival By Gender and Ticket Class")
ax.set_ylabel("Survival (%)")
ax.set_xlabel("")
ax.set_xticklabels(["First Class", "Second Class", "Third Class"])
None # Suppress console output


# Wow! I expected some kind of a trend but nothing like this one. As you can see nearly *all* women survived in first class and the same trend is observed with second class. The women in third class survived more often then men but significantly less then the women in first and second class. The men in first class have a higher chance of surviving then the men the second and third class who have nearly identical chances.
# 
# Using this result I think I predict who died with around 70-80% alone. However, if I want a more accurate model, I'll have to keep exploring other behaviours.

# 2.2. Visualisation By Age
# ---------------------------
# I would like to see the distribution of ages on the Titanic as a purely interest thing. Perhaps it will provide some sort of insight too. I have decided to simply group the ages by year for visualisation.

# In[ ]:


sns.set_style("whitegrid")


# In[ ]:


f, ax = plt.subplots(figsize=(12, 5))
ax = sns.distplot(
    df.Age.dropna().values, bins=range(0, 81, 1), kde=False,
    axlabel='Age (Years)', ax=ax
)


# Interestingly, there seems to be a non-normal distribution for the ages. There is a small spike for young children before having a right-skewed distribution centring around 25 years. A likely explanation for this behaviour would be that families bring their children on the voyage but teenagers are expected to be old enough to care for themselves if the parents went away.

# 2.3. Visualisation By Ticket Class
# ---------------------------------
# How about the distribution by ticket class? Let us look at these values using pie charts.

# In[ ]:


total, classes_count = float(df['Pclass'].shape[0]), df['Pclass'].value_counts()
proportions = list(map(lambda x: classes_count.loc[x]/total*100, [1, 2, 3]))

f, ax = plt.subplots(figsize=(8, 8))
ax.set_title('Proportion of Passengers By Class')
ax.pie(proportions, autopct='%1.1f%%', labels=['First Class', 'Second Class', 'Third Class'])
None  # Removes console output


# 3. Building a Simple Model
# =======================
# Now that we have identified some basic trends we can begin to create a model that predicts if a person survives or dies based on data alone.
# 
# Again we are going to use the assumption that the testing data is similar to the training data such that the percentage of people who died in both sets should be identical. I will use the mean number of deaths for certain parameters as the threshold to break under a random roll to see if they survive or die. Let us try this right now.

# Each probability is stored in a dictionary with the key being a list of values for each of the columns I'm testing for. Since currently I'm just testing for sex and ticket class the key will be [Sex, Ticket Class]

# In[ ]:


def probability(df, key_list):
    """Finds the probability of surviving based on the parameters passed in key_list.
    
    The key_list input is structured like so:
        [Ticket Class, Sex]
    
    So for example, an input could be [1, 'male'].
    """
    pclass, sex = key_list
    filtered_df = df[(df.Sex == sex) & (df.Pclass == pclass)]
    return filtered_df['Survived'].mean()

##############################################################################################

sexes = df.Sex.unique()
ticket_classes = df.Pclass.unique()

probability_dict = dict()
for x in ticket_classes:
    for y in sexes:
        key = [x, y]
        probability_dict[str(key)] = probability(df, key)
        
##############################################################################################

def make_guesses(df):
    """Makes guesses on if the passengers survived or died."""
    guesses = list()
    for passenger_index, row in df.iterrows():
        # Find if the passenger survived.
        survival_key = [row.Pclass, row.Sex]
        survival_odds = probability_dict[str(survival_key)]
        survived_rolls = list(map(lambda x: random.random() <= survival_odds, range(NUM_OF_ROLLS)))
        survived = sum(survived_rolls) > NUM_OF_ROLLS/2

        # Add the result to the guesses
        guesses.append(survived)
    return guesses

##############################################################################################

df['Guess'] = make_guesses(df)
df['CorrectGuess'] = df.Guess == df.Survived
df.head()


# 4. Evaluating the Model
# ===========================
# With our first model developed, how can we be sure of its accuracy? Firstly let us compute the mean of our accuracy (which is percentage correct in this case as the result is only true or false).

# In[ ]:


df.CorrectGuess.mean()


# Our guess is alright! On average I get around 75% guess accuracy but you might be seeing another number. My model works on probability so the number of correct guesses changes when rerun.
# 
# To properly get a measure of how correct the model is, I've decided to Monte-Carlo the experiment and view the histogram. This will tell us how much of a spread the model has guessing the right answer and what on average I expect my accuracy to be.

# In[ ]:


results = list()
for ii in range(10**2):
    guesses = make_guesses(df)
    correct_guesses = (df.Survived == guesses)
    results.append(correct_guesses.mean())
sns.distplot(results, kde=False)
None


# As you can see, the model is normally distributed about 75.5% and has a spread from 70% to 80% accuracy. That means we are getting the right answer 3/4 of the time! That isn't too bad but I know we can do better.

# 4.1. How Can We Improve The Model?
# ====================================
# The model currently only uses one (or two if you don't count the AND of probabilities as one test) measure of survival rate. This is fine and good but what happens if we want to include more parameters in the model? There has to be a way of combining different likelihoods of surviving into a single measure.
# 
# My idea is as follows. The likelihood of surviving is determined by a weighted average. Each parameter or collection of parameters are given a weighting depending on how far away the prediction is to random chance and normalised so that the weightings sum up to one. I'll illustrate this with an example.
# 
# Say that there is a 40 year old women travelling in first class. The fact that she is a women in first class gives a likelihood of surviving as 90% and the fact that she is 40 years old gives a likelihood of 60%. I would assign the weighting of 80%-20% since the first parameter is 40% away from random chance while the second parameter is only 10% away. These percentages normalised to sum to 100% give 80% and 20% respectively.
# 
# I am not sure if this would work but it is worth a shot irregardless. We can tweak the model later if the result isn't consistent.

# If I am going to improve this model then I would want to remove the two columns of Guess and CorrectGuess from the dataframe. They will get re-added at the end with the new model.

# In[ ]:


df.drop('Guess', axis=1, inplace=True)
df.drop('CorrectGuess', axis=1, inplace=True)
df.head()


# 5. Improving The Model By Including Age
# ====================================
# A new factor to include in the model is the age of the passengers. I expect that there should be some kind of trend with
# the age of the passenger and their likelihood of surviving. Let us try to identify this trend by visualising the survival rate histogram overlaid with the ages histogram.

# First plot the histograms without filtering:

# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
sns.distplot(
    df.Age.dropna().values, bins=range(0, 81, 1), kde=False,
    axlabel='Age (Years)', ax=ax
)
sns.distplot(
    df[(df.Survived == 1)].Age.dropna().values, bins=range(0, 81, 1), kde=False,
    axlabel='Age (Years)', ax=ax
)
None # Suppress console output.


# Interestingly, it seems that children below 16 years have a really high chance of surviving as well as passengers above 50 years old. The worst survival rate is for passengers between 18 to 45 years.
# 
# Let us now redo this analysis but split the figure into one for males and one for females.

# In[ ]:


f, ax = plt.subplots(2, figsize=(12, 8))
# Plot both sexes on different axes
for ii, sex in enumerate(['male', 'female']):
    sns.distplot(
        df[df.Sex == sex].Age.dropna().values, bins=range(0, 81, 1), kde=False,
        axlabel='Age (Years)', ax=ax[ii]
    )
    sns.distplot(
        df[(df.Survived == 1)&(df.Sex == sex)].Age.dropna().values, bins=range(0, 81, 1), kde=False,
        axlabel='Age (Years)', ax=ax[ii]
    )

None # Suppress console output.


# This result supports what we found before, that females mostly survived over males, but it also provides some new insight. Notice that for male children their survival rate is still really high (<15 years) but is consistently low otherwise. As such you could tweak the model to say that children are much more likely to survived irregardless of gender.
# 
# Let us try to visualise the same plot again but set the bin width as 5 years.

# In[ ]:


f, ax = plt.subplots(2, figsize=(12, 8))
# Plot both sexes on different axes
for ii, sex in enumerate(['male', 'female']):
    sns.distplot(
        df[df.Sex == sex].Age.dropna().values, bins=range(0, 81, 5), kde=False,
        axlabel='Age (Years)', ax=ax[ii]
    )
    sns.distplot(
        df[(df.Survived == 1)&(df.Sex == sex)].Age.dropna().values, bins=range(0, 81, 5), kde=False,
        axlabel='Age (Years)', ax=ax[ii]
    )

None # Suppress console output.


# Our conclusion is supported! Now we have to figure out if we can include this in the model.
# 
# Let us compute the survival rate on 5 year bin-widths and use that in the final model. 

# In[ ]:


survival_rates, survival_labels = list(), list()
for x in range(0, 90+5, 5):
    aged_df = df[(x <= df.Age)&(df.Age <= x+5)]
    survival_rate = aged_df['Survived'].mean()
    survival_rate = 0.5 if (survival_rate == 0.0 or survival_rate == 1.0) else survival_rate
    
    survival_rates.append(survival_rate if (survival_rate != 0.0 or survival_rate != 1.0) else 0.5)
    survival_labels.append('(%i, %i]' % (x, x+5))

f, ax = plt.subplots(figsize=(12, 8))
ax = sns.barplot(x=survival_labels, y=survival_rates, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=50)
None # Suppress console output


# Now with these results visualised it should be easier to see. The survival rate for infants (<5 years) is quite high, while for men between 20 to 25 years it is only 35%. Anytime there is a 50% reading this is because their isn't enough information and you can only conclude that the probability matches random chance.

# 6. Bringing all of it Together
# ===========================
# Now that we have computed enough information for our model, we can begin to combine it all together and form our weighted probability of surviving.
# 
# In the end I decided that the best way to evaluate the model is through the use of ensembling as it gave a much better result when dealing with unbias and unrelated decision trees. That is to say, each parameter throws separate dice some number of times and a majority vote is taken. That way we don't have to deal with problems with weighting and can treat each parameter separately.

# In[ ]:


def getProbability(passengerId, df):
    """
    Finds the weighted probability of surviving based on the passenger's parameters.
    
    This function finds the passenger's information by looking for their id in the dataframe
    and extracting the information that it needs. Currently the probability is found using a
    weighted mean on the following parameters:
        - Pclass: Higher the ticket class the more likely they will survive.
        - Sex: Women on average had a higher chance of living.
        - Age: Infants and older people had a greater chance of living.
    """
    
    passenger = df.loc[passengerId]
    
    # Survival rate based on sex and ticket class.
    bySexAndClass = df[
        (df.Sex == passenger.Sex) & 
        (df.Pclass == passenger.Pclass)
    ].Survived.mean()
    
    # Survival rate based on sex and age.
    byAge = df[
        (df.Sex == passenger.Sex) & 
        ((df.Age//5-1)*5 <= passenger.Age) & (passenger.Age <= (df.Age//5)*5)
    ].Survived.mean()
    
    # Find the weighting for each of the rates.
    parameters = [bySexAndClass, byAge]
    rolls = [5, 4]  # Roll numbers are hardcoded until I figure out the weighting system
    
    probabilities = []
    for Nrolls, prob in zip(rolls, parameters):
        for _ in range(Nrolls):
            probabilities += [prob]
    return probabilities

##############################################################################################

def make_guesses(df):
    """Makes guesses on if the passengers survived or died."""
    guesses = list()
    for passenger_index, _row in df.iterrows():
        # Find if the passenger survived.
        survival_odds = getProbability(passenger_index, df)
        roll_outcomes = []
        for prob in survival_odds:
            roll_outcomes += [random.random() <= prob]
        survived = sum(roll_outcomes) > len(roll_outcomes)/2

        # Add the result to the guesses
        guesses.append(survived)
    return guesses

##############################################################################################

df['Guess'] = make_guesses(df)
df['CorrectGuess'] = df.Guess == df.Survived
df.head()


# In[ ]:


df.CorrectGuess.mean()


# Currently the execution time for the below cell is really long because I haven't bothered to optimise it.

# In[ ]:


results = list()
for ii in range(10**2):
    guesses = make_guesses(df)
    correct_guesses = (df.Survived == guesses)
    results.append(correct_guesses.mean())
    
    if ii % 10 == 0: print("%i/%i" % (ii, 10**2))
sns.distplot(results, kde=False)
None


# With the new ensembling method we can now reach accuracy of 77%. This prediction will only get better when we determine a proper way of weighting each of the decision trees.
# 
# My thoughts on that, actually, is that you could determine the weighting using Monte-Carlo predictions. If we treat each decision tree separately and "roll" to find the normal distribution, then we can create a weighting based on it's average correct prediction and standard deviation. That would be pretty hard to do and would come later on when we have more decision trees to work with.
