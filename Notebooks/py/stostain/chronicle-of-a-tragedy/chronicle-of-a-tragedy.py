#!/usr/bin/env python
# coding: utf-8

# # Chronicle of a tragedy: a python notebook to Titanic challenge
# *Sylvain Tostain, 2017*

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from IPython.display import display

import warnings
warnings.filterwarnings('ignore')


# # Introduction
# Dear Reader, welcome.
# 
# This is my very first Kaggle Kernel (a tradition it seems): I hope you'll find it interesting and relevant. Of course, I welcome any comment, feel free to drop a comment and I'll try my best to reply and amend the notebook ASAP if necessary.
# 
# Through this Kernel, I've tried to provide detailed guidance and explanation on every step I went through to output a cross-validated prediction, putting here an emphasis on "data munging". I know this is a bit long to read, but I hope you'll enjoy this chronicle of the Titanic tragedy as much as I enjoyed writing it. Don't worry, you'll find special sections with the outcomes of the main steps to keep you on tracks.
# 
# For sure, this is far from being perfect, I'll provide updates in the future to improve and better document some steps. Have a nice reading, I hope to reading from you soon.
# 
# # Course of action
# 
# In order to build a prediction for the educational Titanic challenge, we followed this workflow that also provides a backbone to this notebook:
# 1. Fetching the data,
# 2. Looking at the label to predict,
# 3. Exploring base features
#     * Identifying data gaps
#     * Categorical features
#     * Continuous features
#     * Unstructured data
# 4. Adding new features
# 5. Selecting features
# 6. Building a feature extractor
# 7. Building a predictor
# 8. Optimising hyperparameters and cross validating
# 9. Making predictions
# 
# <div class="alert alert-success">
# <b>Milestones</b><br><br>
# All along this notebook, we'll stop on some milestones to sum up our insights and results. These summaries will be clearly marked by this layout.</div><br>
# 
# Should you wish to discuss certain aspects of this notebook or just find it interesting, feel free to leave a comment and I'll do my best to reply in a timely manner.

# # 1.Fetching the data
# To start with, let's begin by having a look at the files made available.

# In[2]:


# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# A `train.csv` and `test.csv` dataset are supplied together with a template for prediction submissions (`gender_submission.csv`). For the time being, we'll leave aside the test dataset and template for submission file to focus on the train dataset.
# 
# Let's start by loading the train dataset at our disposal in a pandas DataFrame and then having a peek at its datatypes (the `dtypes` attribute of our newly created `data_train` pandas DataFrame).

# In[3]:


data_train = pd.read_csv('../input/train.csv', index_col=0)
data_train.info()


# There is already a lot to notice from a such simple step.
# 
# Our label is `Survived`, an `object` (status text), as we are indeed facing a classification issue.
# 
# The data is relatively thin, there are only 10 features (provided we consider the `PassengerId` as a unique identifier and not a feature as such:
# * Features `Name`, `Sex`, `Ticket`, `Cabin` and `Embarked` (5 out of 10) are texts or categorical.
# * Features `Pclass`, `SibSp`, `Parch` (3 out of 10) are integers or ordinal.
# * Features `Age` and `Fare` only are continuous real numbers.

# # 2.The label to predict: Survival
# Sadly, what we seek here is to make a prediction on wheter a given passenger has survived the sinking of the RMS Titanic or not.
# Let's have a look at the distinct values that can take our prediction label `Survived.

# In[4]:


print("Distinct values taken by Survival:", data_train['Survived'].unique())


# We are therefore to simply predict only one level: `1` if the passenger survived the tragedy, `0` if sadly he didn't. Let's have a look at the total number of labeled observations we have in our train set and how many people survived.

# In[5]:


print("Number of passengers:", data_train['Survived'].count())
print("Number of survivors:", data_train['Survived'].sum())

print("Chances of survival (baseline):",
      data_train['Survived'].sum() / data_train['Survived'].count())

sns.countplot(x='Survived', data=data_train);


# 342 out of 891 survived the sinking of RMS Titanic (38.38% of the passengers).
# Note that:
# 1. Classes are somehow imbalanced and;
# 2. We can devise a simple and stupid base predictor, that is of a 38,38% chances to survive the tragedy.

# Before moving to the features, let's stick to a good practice that is to separate the label to predict (`y`) from the features used to predict the label (`X`).

# In[6]:


X_train = data_train.drop('Survived', axis=1)
y_train = data_train.get(['Survived'])

y_train.head(5)


# # 3.Exploring the features
# Now that we have a clear view of our prediction label, we can look at the features. We can't resist here to have a glimpse at the dataset and compute some basic statistics, but as a precaution we'll refrain from making any interpretation at such an early stage.

# In[7]:


data_train.tail(5)


# In[8]:


data_train.describe()


# ## Data gaps
# To begin with, let's look if we are missing data for some observations.

# In[9]:


data_train['Survived'].count() - data_train.count()


# We are missing info on some observations and will therefore have to devise a strategy to fill in those data gaps.
# * A great number of `Cabin` data are missing. We have been told and saw from the data that this feature presents the cabin number(s) a given passenger has booked. Therefore, the absence of information here is likely to bear information in itself, reflecting the fact that a passenger does not travel in her/his own cabin. This information can be captured in a separate feature.
# * There are 2 missing values for `Embarked`, possible stowaways or crew members? These can be filled-in with the most represented value, that is also the most probable.
# * The `Age` is missing for a great number of passengers: this is obviously an issue. In the absence of other strategy, we can fill-in data gaps with medians.

# ## Categorical and ordinal features
# Let's have a look at the values taken by categorical and ordinal features.

# In[10]:


print("Values taken by Pclass:", sorted(data_train['Pclass'].unique()))
print("Values taken by Sex:", sorted(data_train['Sex'].unique()))
print("Values taken by SibSp:", sorted(data_train['SibSp'].unique()))
print("Values taken by Parch:", sorted(data_train['Parch'].unique()))
print("Values taken by Embarked:", data_train['Embarked'].unique())


# Let's check how these classes are balanced.

# In[11]:


sns.set_palette('Dark2') # A matplotlib colormap relevant for categorical values

fig1, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10))
sns.countplot(x='Pclass', data=data_train, ax=ax1)
sns.countplot(x='Sex', data=data_train, ax=ax2)
sns.countplot(x='SibSp', data=data_train, ax=ax3)
sns.countplot(x='Parch', data=data_train, ax=ax4);


# Again, there would be a lot to say here from these simple counts.
# * First of all, these classes are heavily imbalanced.
# * The vast majority of passengers (more than the half of the population actually) were traveling in 3rd class, the other passengers are almost equally split between 2nd and 1st classes.
# * There were almost twice as many men as women travelling on the titanic.
# * It seems also that a great number of passengers were traveling alone.
# 
# So, the typical RMS Titanic passenger is a single man traveling alone in 3rd class, probably seeking new horizons to relocate on the other side of the ocean. What follows also supports this assumption.

# In[12]:


sns.countplot(x="Pclass", hue="Sex", data=data_train);


# But these men traveling in 3rd class were not too fortunate... More generally, men were less likely to survive this tragedy also in 2nd and 1st class, even if the higher the class, the higher the chances to survive.
# 
# Women in first and second class mostly survived the sinking of the Titanic, whereas odds were only of 50% for women traveling in 3rd class.

# In[13]:


sns.factorplot(x='Pclass', hue='Survived', col='Sex', data=data_train, kind='count');


# In[14]:


sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=data_train);


# If we look at each of these categorical variable independently, we observe the following.

# In[15]:


g = sns.PairGrid(data=data_train,
                 y_vars="Survived",
                 x_vars=["Pclass", "Sex", "SibSp", "Parch"])
g.map(sns.pointplot);


# `Pclass` and `Sex` factors seem to be good predictors of the survival as such. In addition, there is an interaction between the two as seen on the factorplot above.
# 
# `SibSp` and `Parch` factors show something different: level `0` seems to exhibit a different behaviour as if there were a confusion with another effect. We can propose the following hypothesis for later data enrichment.
# * 0 means that a traveler has no relatives onboard the ship. These people apparently tended to behave differently from people traveling with relatives, which was not necessarily supporting their best interests.
# * non-0 values means that people traveled with relatives. These people tended to behave differently from lonely travelers which marks a possible group behaviour. It seems like the bigger the family group is, the lesser the odds of survival are. Smaller families groups of (2 to 3 people) tends to have better survival chances than the average, whereas bigger groups (more than 3 people) tends to on the contrary have lower chances of survival.

# Let's conclude this categorical features exploration with `Embarked`. There are 3 possible values, respectively `S` (for Southampton, England), `C` (for Cherbourg, Normandy) and `Q` (for Queenstown now Cobh, Ireland). There are also 2 missing values.

# In[16]:


sns.factorplot(x="Embarked", hue="Survived", col="Sex", data=data_train, kind="count");
sns.factorplot(x="Embarked", y="Survived", col="Sex", data=data_train, kind="bar");


# In[17]:


sns.pointplot(x='Embarked', y='Survived', hue='Sex', data=data_train);


# It seems that passengers embarked in Cherbourg had better chances of survival, we'll keep this feature.
# 
# We remind that two passengers do not have a value for `Embarked`. Given the fact that most of the passengers embarked in Southampton, we'll fill in these 2 gaps with `S`. 

# In[18]:


data_train[data_train['Embarked'].isnull()].drop('Embarked', axis=1)


# Enough with the categorical variables so far, we'll get back to them later. So far, we can sum up our insight.
# <div class="alert alert-success">
# <b>Exploration of base categorical features</b><br><br>
# Among the categorical variables we just explored, we'll keep in mind for future reference that:<br>
# <ul><li><b>`Pclass`</b> should definitely be retained as a feature for our predictor, as it appears obviously connected to the chances of survival.</li>
# <li><b>`Sex`</b> should be retained as well.</li>
# <li><b>`SipSp`</b> may also be a good predictor, although it probably requires feature engineering. We can for instance think about deriving a categorical variable (0 vs. non-0).</li>
# <li><b>`Parch`</b> might appear redundant with `SibSp`.</li>
# <li><b>`Embarked`</b> also provides information on chances of survival, we should keep it as well. Missing values will be replaced by the most frequent value, `S`.</li></ul>
# </div>

# ## Continuous features
# We'll address continuous features now. Basically, continuous features are:
# * `Age`: age of the passengers, note that some ages are missing.
# * `Fare`: price paid by the passenger.
# 
# At first, let's have a look at basic statistics and the distributions of our 2 features.

# In[19]:


data_train[['Age', 'Fare']].describe()


# In[20]:


sns.distplot(data_train['Age'].dropna());


# In[21]:


sns.distplot(data_train['Fare']);


# We notice that the distibution of `Fare` is far from following a normal distribution. We'll derive `logFare` the logarithm of `Fare`. We obtain a more useful distribution.

# In[22]:


data_train['logFare'] = data_train['Fare'].apply(lambda x: np.log(x + 1))
sns.distplot(data_train['logFare']);


# In both cases, the distributions of `Age` and `logFare` have relatively heavy tails on the right.
# 
# Let's see if they are somehow related.

# In[23]:


sns.jointplot('Age', 'logFare', kind="kde", size=7, space=0, data=data_train);


# There is no obvious correlation between age and fare, although there are some groups.

# In[24]:


g = sns.FacetGrid(data_train, hue='Survived', size=6)
g.map(plt.scatter, 'Age', 'logFare', s=20, alpha=.7, linewidth=.5, edgecolor="white")
g.add_legend();


# There is no clear pattern, also it seems like passengers paying more seems to have better chances of survival. Passenger travelling for free (crew members ?) weren't very lucky: it might worth deriving a feature on this condition.
# 
# Another fact that worth, being noted: traveling was more expensive for young passengers (especially below 15). This is another feature that we can derive, especially if we intend to use tree based algorithms.

# In[25]:


g = sns.lmplot(x='Age', y='logFare', hue='Pclass',
               truncate=True, size=6, data=data_train[data_train.Fare != 0])


# From the graphs below, we also notice that:
# * Younger and older passengers had better chances of survival.
# * Passengers traveling for free had lower chances of survival.

# In[26]:


g = sns.PairGrid(data_train,
                 x_vars=['Survived'],
                 y_vars=['Age', 'logFare'],
                 size=5)
g.map(sns.violinplot, palette='Dark2');


# Before concluding our overview of continuous features, we still have an issue to address: the great amount of missing data for `Age`.

# In[27]:


data_train[data_train['Age'].isnull()]    .drop('Age', axis=1)    .drop('logFare', axis=1)    .tail(5)


# Filling missing values for `Age` is not that simple, we can think about several strategies:
# 1. The simple one: replace missing values by the median. Why not the mean? Since we have seen that the distribution of `Age` is not normal and therefore, the mean is not a good estimator of the central value of the dataset: the median is usually a more reliable estimator in such cases. This is probably a simple and prudent approach in the first place.
# 2. Taking the median of the sample of population to whom the passenger belong, based on one or more features (e.g. `Pclass`, `Sex`, `Embarked`,... of course not `Survived` as this is our prediction label. We should be careful, as pushing too far such a strategy could introduce bias in our predictions.
# 3. Taking a random value, which is probably not a good option as it would bring unnecessary white noise and variance to our data.
# 4. Simply dropping the data, which would lead us to drop a significant amount of lines and increase bias and variance of our prediction. This is for sure not advisable.
# 
# We'll go for option 2 based upon `Sex` and `Pclass` for this attempt, as `Embarked` is too much imbalanced.

# In[28]:


age_fill = data_train[['Sex', 'Pclass', 'Age']].groupby(['Sex', 'Pclass']).median()
age_fill


# Example of how to get these values at a later stage.

# In[29]:


age_fill.get_value(('female',1),'Age')


# We've finished our tour regarding continuous variables so far, we'll get back to them later.
# <div class="alert alert-success">
# <b>Exploration of base continuous features</b><br><br>
# Among the continuous variables we just explored, we'll keep in mind for future reference that:<br>
# <ul><li><b>`Age`</b> should be retained as a predictor. Nevertheless, it might be relevant to derive a distinct categorical feature based upon the age to make a distinction between adults and children (e.g. below 15).</li>
# <li><b>`Fare`</b> is too far from a normal distribution, we derived `logFare` instead.</li>
# <li><b>`logFare`</b> will be kept as a potential predictor. We can also derive another feature to make a distinction between passengers traveling for free and others.</li>
# <li><b>Gaps in `Age`</b> will be filled in based upon the median of groups defined by `Sex` and `Pclass`.</li>
# </ul>
# </div>

# ## Unstructured data
# The dataset also comes with poorly structured data under the form of more or less structured strings, `Name`, `Ticket` and `Cabin`. These data are unusable as such, but are likely to provide additional features if we are able to derive relevant data from them.

# In[30]:


data_train[['Name','Ticket','Cabin']].head(15)


# ### Ticket
# Ticket values are alphanumerical codes. Some ticket numbers are purely numerical where some others exhibit letters or special characters: there is no universal pattern.
# 
# Strangely enough, the ticket number is not unique: some passengers seems to share the same ticket numbers. Still, the number of unique values remain too high to be treated conveniently as a categorical variable.

# In[31]:


len(data_train['Ticket'].unique())


# In[32]:


data_train['Ticket'].value_counts().head(20)


# The value `'LINE'` retain our attention. It may capture the crew members of the Titanic... unfortunately this is incomplete (4 people). Generally speaking, passengers sharing the same `Tickets` are of the same family, see the example above.
# 
# We also notice that the number of members of a family is actually the result of `SibSp + Parch`. We'll have to create this feature as well.

# In[33]:


data_train[data_train['Ticket']=='347082'].drop('logFare', axis=1)


# <div class="alert alert-danger">
# <b>Tickets</b><br><br>
# Unfortunately, it's a long shot: we'll simply drop the `Ticket` without further investigations.<br>Nevertheless, we keep the idea of deriving a feature with the total number of family members from the example of family Andersson traveling with ticket nr. 347082.</div><br>

# ### Name
# There would be a lot to say and to try regarding the name, justifying a dedicated notebook on this topic. At this stage, this is beyond the scope of this notebook, so we'll very briefly address the `Name` using a simplistic approach, without spending too much time on it.
# <div class="alert alert-warning">
# <b>Name tokenization and further treatments: off-topic</b><br><br>
# There would be a lot to say about engineering the `Name`, advocating for a detailed treatment of this data, but we won't do it here: we simply list here a couple of ideas that could be implemented to address the issue<br>
# <ul><li>Tokenization could be tempted on this string, in order to separate words and strip spaces and special characters.</li>
# <li>A detection of most recurrent words could be implemented in order to derive titles that could be dummified, this has been done elsewhere several times and is well documented.</li>
# <li>There seems to be a structure that might allow for the construction of full families and why not even graphs, since we could extract the last name, and possibly maiden name that are very often put in brackets at the end of the name. This would therefore require a significant treatment.</li>
# </ul></div><br>
# Instead, we make this simple observation: longer names is correlated to better chances of survival.

# In[34]:


sns.violinplot(x=data_train.Survived,y=data_train['Name'].apply(len));


# <div class="alert alert-success">
# <b>Name</b><br><br>
# In this first attempt, we'll derive the length of the name as a prediction variable and drop the name.
# </div><br>

# ### Cabin
# The `Cabin` feature contains one or more cabins reserved by the passenger, if any. A great number of passengers do not have a cabin and probably sleep in a dormitory.

# In[35]:


data_train['Cabin'].unique()


# A cabin number is (relatively) standard. There are some exceptions but cabins are composed of a letter comprised between `A` and `F` followed by a number. The number is probably useless but the letter is the deck where the cabin was located.
# 
# We can derive a feature from this deck letter and add a separate level for passengers not having a cabin. As the decks are ordered from A to F, we can probably make this an ordinal variable encoded as an integer.

# In[36]:


data_train['Deck'] = data_train['Cabin'].str.extract('(^[A-Z]{1})', expand=True).fillna('G')
data_train['Deck'] = data_train['Deck']         .replace(['A','B','C','D','E','F','G','T'],[1,2,3,4,5,6,7,8])
    
sns.countplot(data=data_train, x='Deck');


# In[37]:


sns.pointplot(x='Deck', y='Survived', hue='Sex', data=data_train);


# We see that actually, the fact to have a cabin or not (level 7) is probably more significant on chances of survival than the `Deck` variable.
# 
# <div class="alert alert-success">
# <b>Cabin</b><br><br>
# In this first attempt, we'll extract the deck of the first cabin, if any (first letter of the string) and complete the dataset with an additional level for missing values (dormitories). Then, we'll recode this on integers as the decks are ordered. Alternatively, it might simply be as efficient to derive a variable based upon the fact to have a cabin or not. 
# </div><br>

# # 4.Adding new features
# As we finished the exploration of the base dataset, we saw that a number of features can easily be derived from the base dataset, respectively:
# 
# |New feature|Derived from|Legend                                            |
# |-----------------|-------------------|----------------------------------------------|
# |Fsize             |SibSp, Parch |Family size                                      |
# |Adult            |Age                 |If a passenger is older than 15   |
# |logFare        |Fare                |A logarithm of the Fare (scaling)|
# |lenName     |Name             |Length of the name                       |
# |hasCabin     |Cabin             |Passenger has a cabin                  |

# In[44]:


data_train = pd.read_csv('../input/train.csv', index_col=0)

# Generating Fsize
data_train['Fsize'] = data_train['SibSp'] + data_train['Parch']

# Generating Adult
data_train['Adult'] = (data_train['Age'] >= 15).astype(int)

# Generating Alone, suggested by Reinhard given the correlation structure
data_train['Alone'] = (data_train['Fsize'] == 0).astype(int) 

# Generating logFare
data_train['logFare'] = data_train['Fare'].apply(lambda x: np.log(x + 1))

# Generating lenName
data_train['lenName'] = data_train['Name'].apply(len)

# Generating hasCabin
data_train['hasCabin'] = data_train['Cabin'].isnull().astype(int)

# Generating Deck
data_train['Deck'] = data_train['Cabin'].str.extract('(^[A-Z]{1})', expand=True).fillna('G')
data_train['Deck'] = data_train['Deck']             .replace(['A','B','C','D','E','F','G','T'],[1,2,3,4,5,6,7,8])

# Drop features no longer useful
data_train = data_train.drop(['SibSp'], axis=1)                       .drop(['Parch'], axis=1)                       .drop(['Ticket'], axis=1)                       .drop(['Fare'], axis=1)                       .drop(['Name'], axis=1)                       .drop(['Cabin'], axis=1)

data_train.head(5)


# ## Filling in data gaps
# We just need now to fill-in the data gaps as discussed above:
# * Filling in empty values for `Embarked` with the most frequent (and probable) value: `S`.
# * Filling in empty values for `Age` with the median age.

# In[45]:


# Filling in empty Age values.
data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())

# Filling in empty values for Embarked
data_train['Embarked'] = data_train['Embarked'].fillna('S')


# <div class="alert alert-success">
# <b>Data munging</b><br><br>
# At this stage, we have finished the preparation of our dataset (cleaning, adressing data gaps, enriching the dataset and dropping useless features).  We are now ready to proceed with modelling.
# </div><br>

# # 5.Selecting features
# We'll now have to selet the features we wish to use with our model.
# 
# Let's have a look at the clean and enriched data we have at this stage:

# In[46]:


data_train.head(5)


# We start by computing the correlation matrix of our dataset.

# In[47]:


corr = data_train.corr()


# And visualise a heatmap of half of this correlation matrix.
# 
# Remember that a correlation matrix is symetric, therefore half of the information corresponding to one size of the diagonal is redundant. As a good practice, we want to favor readability of our visualisation so we should stick to the "keep it simple stupid" (KISS) principle. I then follow the advice of my old six-sigma guru and therefore drop one half of the matrix before displaying the heatmap.

# In[48]:


# The following excerpt is inspired from the seaborn documentation :
#http://seaborn.pydata.org/examples/many_pairwise_correlations.html

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, vmax=.3, center=0,
            square=True, cmap=cm.seismic, linewidths=.5, cbar_kws={"shrink": .5});


# We observe that there is some correlation structures in this dataset, the better correlations are around 0.6.
# 
# We notice that the label `Survived` is relatively well correlated to `Pclass` and to a lower extent to `Deck` and `hasCabin`. `Age`, `Fsize` and `Adult` are less correlated. On the contrary, `logFare` and `lenName` have a negative correlation with `Survived`.
# 
# In addition, we also notice that some of the features of this dataset are highly correlated, like `logFare` with `Pclass`, `hasCabin` and `Deck`. It is therefore possible that the information from `logFare` is redundant somehow with the information from the 3 other features.
# 
# 
# Classical statistic models are sensitive to this issue: bringing several times the same effect into a model can be detrimental to its quality, so let's keep this in mind that dropping `logFare` can be an option if the model quality suffers from high variance.

# <div class="alert alert-success">
# <b>Correlations</b><br><br>
# We had a look at the correlation within our dataset and noticed that some featuresare likely to be good predictors (`Pclass`, `Deck`, `hasCabin`, `lenName`, `logFare`.<br>
# We also notice correlations within our features which should be kept in mind for feature selection (avoid using highly correlated features together). For instance `logFare` is correlated to `Pclass`, `hasCabin` and `Deck`, and `hasCabin` and `Deck` are correlated.
# </div><br>

# # 6. Building a feature extractor
# We are now ready to start a model!
# 
# But as we are lazy people, let's do this seriously and properly so that it is easy to maintain and use: we are going to create a pipeline to handle our predictions and make our lives easy with feature preparation, training and cross validations.
# 
# Let's start by creating a __feature extractor__.

# ## Feature extractor
# We will build a python class `FeatureExtractor` that we'll use to process our raw datasets in order to output the nice and clean dataset we saw a couple of lines before. This feature extractor will go through each and every step we consider relevant for data preparation and output a clean dataset, ready for training. We'll add also a one-hot encoding for categorical features.
# 
# This somehow boring step will save us time later for the maintenance of our pipeline and the handling of train, cross-validation and handling of test datasets.
# 
# This is (fortunately) not the case here, but such a feature extractor will allow us to build an out-of-the-box pipeline to put into production a predictor on fresh new data, should it be necessary. The feature extractor we are about to create can be used into any python environment where pandas is available.
# 
# We'll make use of objects made available by the guys from <a href=http://scikit-learn.org>scikit-learn</a> to build this pipeline and would like to thank the <a href=https://ramp.studio> RAMP Studio</a> team for providing guidance on this good practice.

# In[49]:


import numpy as np
import pandas as pd

class FeatureExtractor(object):
    def __init__(self):
        # Nothing special to address in the constructor
        pass
    
    def fit(self, Xraw_df, y_array):
        # No fitting action required, this is a feature extractor
        pass
    
    def transform(self, Xraw_df, return_df=False):
        # Feeding the raw data
        Xtrf_df = Xraw_df

        # Generating new features
        Xtrf_df['Fsize'] = Xtrf_df['SibSp'] + Xtrf_df['Parch']
        Xtrf_df['Alone'] = (Xtrf_df['Fsize'] == 0).astype(int)
        Xtrf_df['Adult'] = (Xtrf_df['Age'] >= 15).astype(int)
        Xtrf_df['logFare'] = Xtrf_df['Fare'].apply(lambda x: np.log(x + 1))
        Xtrf_df['lenName'] = Xtrf_df['Name'].apply(len)
        Xtrf_df['hasCabin'] = Xtrf_df['Cabin'].isnull().astype(int)
        Xtrf_df['Deck'] = Xtrf_df['Cabin'].str.extract('(^[A-Z]{1})', expand=True).fillna('G')
        Xtrf_df['Deck'] = Xtrf_df['Deck']                          .replace(['A','B','C','D','E','F','G','T'],[1,2,3,4,5,6,7,8])
        
        # Filling in empty values.
        Xtrf_df['Age'] = Xtrf_df['Age'].fillna(Xtrf_df['Age'].median())
        Xtrf_df['Embarked'] = Xtrf_df['Embarked'].fillna('S')
        
        # One-hot encoding of categorical features
        Xtrf_df = pd.concat([Xtrf_df,
                            pd.get_dummies(Xtrf_df['Sex'], prefix='Sex', drop_first=True),
                            pd.get_dummies(Xtrf_df['Embarked'], prefix='Emb')],
                            axis=1)
        
        # Droping features no longer useful
        Xtrf_df = Xtrf_df.drop(['SibSp'], axis=1)                         .drop(['Parch'], axis=1)                         .drop(['Ticket'], axis=1)                         .drop(['Fare'], axis=1)                         .drop(['Name'], axis=1)                         .drop(['Cabin'], axis=1)                         .drop(['Sex'], axis=1)                         .drop(['Embarked'], axis=1)
        
        # Returning the output, numpy array is default unless specified otherwise.
        if return_df==True:
            return Xtrf_df
        else:
            return Xtrf_df.values


# Nothing more than the wrap-up of what we did so far, let's test it now.

# In[50]:


feature_extractor = FeatureExtractor()

data_train = pd.read_csv('../input/train.csv', index_col=0)
X_df = feature_extractor.transform(data_train.drop(['Survived'], axis=1), return_df=True)
X_df.head(5)


# In[51]:


X_train = feature_extractor.transform(data_train.drop(['Survived'], axis=1))
y_train = data_train['Survived'].values


# We are now ready for training models.
# # 7.Predictor
# ### Choice of a predictor
# We are challenged to predict __the survival of the passengers (`Survived`), which is 0 or 1__. The quality of our model will be assessed for its __accuracy__.
# 
# We need to build a __classifier__.
# 
# In a first attempt, let's try a high performance gradient boosting decision tree based classifier, provided by <a href=https://github.com/Microsoft/LightGBM>lightGBM</a>. The <a href=http://lightgbm.readthedocs.io/en/latest/index.html>documentation for Light GBM</a> is quite comprehensive. LightGBM provides fast and often very efficient classifiers, that performs quite well in many cases. We'll use the API provided for <a href=http://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api>Scikit-learn</a>.
# 
# A first default classifier is quite simple to implement.

# In[52]:


import lightgbm as lgbm

clf = lgbm.sklearn.LGBMClassifier()
clf.fit(X_train, y_train)
clf


# Making prediction with Scikit-learn is easy.

# In[53]:


y_pred = clf.predict(X_train)


# As well as scoring our model.

# In[54]:


from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_pred)


# Nice ! But wait... it's not that easy!
# 
# <div class="alert alert-danger">
# <b>Malpractice</b><br><br>
# It is not good practice to rely on a model evaluation based upon its training dataset, since it's not "playing fair": the model knows already the data used to evaluate its performance. Therefore, it will not be possible to detect <i>overfitting</i> and validate if our model is indeed a good quality predictor, or if it just stupidly "interpolated" the data and does not perform well on fresh new data.<br>
# We need to go through __cross validation__ to train and check if our model performs well on fresh data not used for training.
# </div><br>

# # 8. Cross validation and optimisation of hyper parameters
# At this stage, we'll go iteratively through __hyperparameters optimisation__ and __cross validation (cv)__.
# 
# There are strategies to test hyperparameters, amongst them:
# * we can test each combination, strategy proposed by GridSearchCV from Scikit-learn, or
# * we can try to test only a limited number of combinations, strategy proposed by the package <a href=https://hyperopt.github.io/hyperopt/>Hyperopt</a> (random or guided strategies).
# 
# There's no unique or perfect strategy: here, we'll try to follow the path proposed by `Hyperopt`.
# ## Defining a strategy for cross validation.
# We need to define a strategy for cross validation: how will we split our train dataset to save some data to evaluate the performances of our model in termes of accuracy. 
# * We'll use KFold to make and test 5 splits of our train dataset,
# * We'll use cross_val_score to compute the accuracy on our 5 splits from the train dataset (X_train, y_train).
# 
# ## Building a predictor to optimize.
# We need to build an object `objective` that defines a predictor that will allow us to test and optimise the hyperparameters of a `LGBMClassifier`, see the documentation of this predictor <a href=http://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.LGBMClassifier>here</a>.

# In[55]:


from sklearn.model_selection import cross_val_score, KFold

cv = KFold(n_splits=5, random_state=42)

i = 0
def objective(params):
    global i
    i += 1
    print(params)
    clf = lgbm.sklearn.LGBMClassifier(num_leaves = int(params['num_leaves']),
                                     max_depth = int(params['max_depth']),
                                     learning_rate = float(params['learning_rate']),
                                     n_estimators = int(params['n_estimators']),
                                     random_state = 42)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=3)
    score = np.mean(scores)
    print("Accuracy: %s" % score)
    print('-----------------')
    df_result_hyperopt.loc[i, ['score'] + list(params.keys())] = [score] + list(params.values())
    return {'loss': 1. - score, 'status': STATUS_OK}


# In[ ]:


from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

space = {'num_leaves': hp.quniform('num_leaves', 5, 50, 5),
        'max_depth': hp.quniform('max_depth', 1, 10, 2),
        'learning_rate': hp.quniform('learning_rate', 0.05, 0.1, 0.05),
        'n_estimators': hp.quniform('n_estimators', 50, 300, 50)}

df_result_hyperopt = pd.DataFrame(columns = ['score'] + list(space.keys()))

i = 0
trials = Trials()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=3,
            trials=trials)

print("Best: %s" % best)


# Etc.
# 
# This remains a relatively time consuming and iterative process, the above described example is for illustration only, it is shortened due to the low value of `max_eval`. Feel free to raise the value of `max_eval` to allow for more iterations (better, but longer).

# ### Final predictor
# Based upon the above optimisation, we'll keep the following predictor.

# In[56]:


import lightgbm as lgbm

clf = lgbm.sklearn.LGBMClassifier(num_leaves = 27,
                                  max_depth = 10,
                                  learning_rate = 0.05,
                                  n_estimators = 200,
                                  random_state = 42)
clf


# # 8.Making predictions
# We're almost done, we'll now feed the pipeline we built to make our predictions for the challenge.
# 
# Now, let's wrap-up the whole thing.

# In[57]:


# Creating the pipeline
feature_extractor = FeatureExtractor()

# Feeding the pipeline with training data
data_train = pd.read_csv('../input/train.csv', index_col=0)
X_train = feature_extractor.transform(data_train.drop(['Survived'], axis=1))
y_train = data_train['Survived'].values

# Training the model
clf.fit(X_train, y_train)

# Feeding the pipeline with test data and making predictions
data_test = pd.read_csv('../input/test.csv', index_col=0)
X_test = feature_extractor.transform(data_test)
y_pred = clf.predict(X_test)

# Reconstructing the output dataframe
pred_df = pd.DataFrame()
pred_df['PassengerId'] = data_test.index
pred_df['Survived'] = y_pred
pred_df.head(10)


# We just have to serialise our final output `pred_df`

# In[58]:


pred_df.to_csv("../working/predictions.csv", index=False)


# We just have to submit it now...
# 
# # That's all folks !
# I thank you for reading thus far, and hope you enjoyed the reading. Feel free to leave a comment.
# 
# I've tried to credit every source of inspiration all along, but should you identify some missing credits I will be happy to include the references.
# 
# Special thanks to all the fellows and the great teachers from X EXED DSSP 7 for their inestimable support and input. 
